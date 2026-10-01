#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Local CPU dry run: exercise the whole pipeline on the tiny random-weight LLaVA.

Nothing here needs a GPU, a download or more than a few hundred MB of RAM: the model is
the real HuggingFace ``LlavaForConditionalGeneration`` with miniature dimensions, and the
corpus is synthetic images. Every stage the fit script and the analysis tools use runs on
the real code paths — manifest -> encode (placeholder expansion + image fusion) ->
position masks -> masked Jacobian fit (incl. checkpoint/resume) -> artifact save/load ->
lens readout -> hold-out scoring -> residual intervention — and the script asserts the
shapes and invariants at each step.

    python scripts/dry_run.py                       # fast (56px images, 16 image tokens)
    python scripts/dry_run.py --image-size 336      # production-shaped 576 image tokens
    python scripts/dry_run.py --shape-check         # + offline check of the real config

``--shape-check`` loads the real LLaVA-1.5 config/processor (no weights) and verifies the
production token accounting; it uses local cache only unless ``--allow-download`` is set.
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import torch
from PIL import Image

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vlm_lens.artifacts import load_lens_set, save_lens_set  # noqa: E402
from vlm_lens.data.dummy import build_dummy_manifest  # noqa: E402
from vlm_lens.data.manifest import manifest_meta, read_manifest  # noqa: E402
from vlm_lens.evaluate import format_scores, score_lens  # noqa: E402
from vlm_lens.fitting import fit_masked  # noqa: E402
from vlm_lens.interventions import ResidualEdit, generate_with_edits, lens_vectors  # noqa: E402
from vlm_lens.models.llava import LlavaLensModel  # noqa: E402
from vlm_lens.models.tiny_llava import TinyLlavaConfig, build_tiny_llava  # noqa: E402
from vlm_lens.positions import build_position_masks, mask_summary  # noqa: E402
from vlm_lens.readout import lens_readout  # noqa: E402

PROMPT = "USER: <image>\nDescribe this image.\nASSISTANT:"


def section(title: str) -> None:
    print(f"\n== {title} ==")


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"DRY RUN FAILED: {message}")
    print(f"  ok: {message}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default="runs/dry-run", help="run directory")
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=56, help="56 = fast, 336 = production token count")
    parser.add_argument("--dim-batch", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shape-check",
        nargs="?",
        const="llava-hf/llava-1.5-7b-hf",
        metavar="MODEL_ID",
        help="also verify the real LLaVA-1.5 config/processor (no weights)",
    )
    parser.add_argument("--allow-download", action="store_true", help="let --shape-check hit the Hub")
    return parser.parse_args(argv)


def check_production_shapes(model_id: str, *, allow_download: bool) -> None:
    """Verify the real checkpoint's token accounting without loading any weights."""
    from transformers import AutoConfig, AutoProcessor, LlavaForConditionalGeneration

    section(f"production shape check ({model_id}, config + processor only)")
    local_files_only = not allow_download
    config = AutoConfig.from_pretrained(model_id, local_files_only=local_files_only)
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
    with torch.device("meta"):
        hf_model = LlavaForConditionalGeneration(config)
    model = LlavaLensModel(hf_model, processor)
    print(
        f"  n_layers={model.n_layers} d_model={model.d_model} "
        f"image_token_id={model.image_token_id} image_seq_length={model.image_seq_length}"
    )

    inputs = processor(
        images=Image.new("RGB", (336, 336)),
        text=PROMPT,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"]
    n_image_tokens = int((input_ids == model.image_token_id).sum())
    pixel_values = inputs["pixel_values"]
    expect(
        n_image_tokens == model.image_seq_length == 576,
        f"placeholder expands to {n_image_tokens} image tokens (expected 576)",
    )
    n_text_tokens = int(input_ids.shape[1]) - n_image_tokens
    expect(
        n_text_tokens >= 5,
        f"prompt tokenizes to {tuple(input_ids.shape)} (576 image + {n_text_tokens} text tokens)",
    )
    expect(
        tuple(pixel_values.shape) == (1, 3, 336, 336),
        f"pixel_values {tuple(pixel_values.shape)}",
    )
    expect(
        model.d_model == int(config.text_config.hidden_size),
        f"d_model matches text_config.hidden_size ({model.d_model})",
    )
    expect(
        len(model.layers) == int(config.text_config.num_hidden_layers),
        f"{len(model.layers)} residual blocks (LLaMA-7B has 32)",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    if args.shape_check:
        check_production_shapes(args.shape_check, allow_download=args.allow_download)
        print()

    section("tiny model (real HF classes, random weights, CPU)")
    config = TinyLlavaConfig(image_size=args.image_size, seed=args.seed)
    hf_model, processor = build_tiny_llava(config)
    model = LlavaLensModel(hf_model, processor)
    expected_image_tokens = (args.image_size // config.patch_size) ** 2
    print(
        f"  n_layers={model.n_layers} d_model={model.d_model} "
        f"image_token_id={model.image_token_id} image_seq_length={model.image_seq_length}"
    )
    expect(model.image_seq_length == expected_image_tokens, f"image_seq_length={expected_image_tokens}")

    section("synthetic corpus -> manifest")
    manifest_path = build_dummy_manifest(
        out_dir / "dummy-corpus",
        n_samples=args.n_samples,
        image_size=args.image_size,
        seed=args.seed,
    )
    samples = read_manifest(manifest_path)
    expect(len(samples) == args.n_samples, f"manifest round-trip: {len(samples)} samples")
    expect(all(Path(sample.images[0]).exists() for sample in samples), "image files exist")
    print(f"  header: {manifest_meta(manifest_path)}")

    section("encode (placeholder expansion + image fusion)")
    batch = model.encode_mm(samples[0].text, samples[0].images[0])
    assert batch.pixel_values is not None
    print(
        f"  input_ids={tuple(batch.input_ids.shape)} attention_mask={tuple(batch.attention_mask.shape)} "
        f"pixel_values={tuple(batch.pixel_values.shape)} ({batch.pixel_values.dtype}, "
        f"{batch.pixel_values.device})"
    )
    expect(batch.seq_len == batch.n_image_tokens + 7, f"{batch.n_image_tokens} image tokens + 7 text tokens")
    expect(batch.n_image_tokens == model.image_seq_length, "image-token count matches config")
    expect(batch.pixel_values.shape[1] == 3, "pixel_values are RGB")

    section("source masks")
    masks = build_position_masks(batch.input_ids, model.image_token_id)
    summary = mask_summary(masks, input_ids=batch.input_ids, image_token_id=model.image_token_id)
    print(f"  {summary}")
    expect(bool((masks["text"] | masks["image"]).eq(masks["all"]).all()), "text | image == all")
    expect(int(masks["image"].sum()) == model.image_seq_length, "image mask spans the image block")
    expect(not bool((masks["text"] & masks["image"]).any()), "text and image masks are disjoint")

    section("masked Jacobian fit (+ checkpoint resume)")
    result = fit_masked(
        model,
        samples,
        dim_batch=args.dim_batch,
        skip_first=1,
        masks=("text", "image", "all"),
        checkpoint_path=out_dir / "checkpoint.pt",
        log_every=0,
    )
    expect(set(result.lenses) == {"text", "image", "all"}, f"fitted masks: {sorted(result.lenses)}")
    for mask, lens in sorted(result.lenses.items()):
        expect(
            all(tuple(J.shape) == (model.d_model, model.d_model) for J in lens.jacobians.values()),
            f"{mask}: {len(lens.jacobians)} Jacobians of "
            f"[{model.d_model}, {model.d_model}] at layers {lens.source_layers}",
        )
        expect(lens.n_prompts == args.n_samples, f"{mask}: n_prompts={lens.n_prompts}")
    resumed = fit_masked(
        model,
        samples,
        dim_batch=args.dim_batch,
        skip_first=1,
        masks=("text", "image", "all"),
        checkpoint_path=out_dir / "checkpoint.pt",
        log_every=0,
    )
    expect(not resumed.history, "resume from checkpoint did not refit any sample")
    expect(
        {mask: lens.n_prompts for mask, lens in resumed.lenses.items()}
        == {mask: lens.n_prompts for mask, lens in result.lenses.items()},
        "resumed lenses keep n_prompts",
    )

    section("artifacts (save / load / upstream interop)")
    written = save_lens_set(
        out_dir / "artifacts",
        result.lenses,
        provenance={"note": "dry run", "n_prompts": result.n_prompts},
    )
    print(f"  wrote {sorted(path.name for path in written.values())}")
    loaded, provenance = load_lens_set(out_dir / "artifacts")
    expect(set(loaded) == set(result.lenses), "lens set round-trips")
    for mask in loaded:
        for layer, J in loaded[mask].jacobians.items():
            expect(
                bool(torch.allclose(J, result.lenses[mask].jacobians[layer], atol=1e-3)),
                f"{mask}: layer {layer} matches before/after fp16 round-trip",
            )
        from jlens.lens import JacobianLens  # noqa: PLC0415  (vendor path installed by vlm_lens)

        upstream = JacobianLens.load(str(written[mask]))
        expect(
            upstream.source_layers == loaded[mask].source_layers and upstream.n_prompts == loaded[mask].n_prompts,
            f"{mask}: upstream jlens.JacobianLens.load accepts the file",
        )
    expect("note" in provenance, "provenance sidecar readable")

    section("lens readout")
    readout = lens_readout(model, result.lenses["text"], batch, layers=[0], positions=[-1])
    vocab_size = hf_model.config.text_config.vocab_size
    expect(
        tuple(readout.lens_logits[0].shape) == (1, vocab_size),
        f"lens logits [n_positions, vocab] = {tuple(readout.lens_logits[0].shape)}",
    )
    top_tokens = readout.top_k(k=3)[0][0]
    print(f"  layer 0 top-3 at the last position: {top_tokens}")
    model_only = lens_readout(
        model, result.lenses["text"], batch, layers=[model.n_layers - 1], positions=[-1], use_jacobian=False
    )
    expect(
        bool(torch.allclose(model_only.lens_logits[model.n_layers - 1], model_only.model_logits, atol=1e-5)),
        "final-layer readout equals the model's own logits (identity transport)",
    )

    section("hold-out fidelity scoring")
    scores = score_lens(model, result.lenses["text"], samples, tags=("text", "image"))
    print(format_scores(scores))
    final_rows = [score for score in scores if score.layer == model.n_layers - 1]
    expect(bool(final_rows), "final-layer sanity row present")
    expect(
        all(row.top1_agreement == 1.0 and row.mean_kl < 1e-6 for row in final_rows),
        "final layer agrees with the model exactly (agreement 1.0, KL 0)",
    )
    expect(all(0.0 <= row.top1_agreement <= 1.0 for row in scores), "agreements in [0, 1]")

    section("residual intervention via a lens vector")
    direction = lens_vectors(model, result.lenses["text"], ["t42"], layers=[0])[0][0]
    edit = ResidualEdit(layer=0, mode="add", alpha=1.0 / float(direction.norm()), token="t42", positions="all")
    baseline = generate_with_edits(model, None, batch, max_new_tokens=args.max_new_tokens)
    edited = generate_with_edits(
        model, result.lenses["text"], batch, edits=[edit], max_new_tokens=args.max_new_tokens
    )
    print(f"  baseline tokens: {baseline['token_ids']}   edited: {edited['token_ids']}")
    expect(edited["n_edit_forwards"] > 0, "edits ran inside generation")
    expect(len(edited["token_ids"]) <= args.max_new_tokens, "edited generation respects max_new_tokens")

    section("summary")
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  peak RSS: {peak_mb:.0f} MiB   wall: {time.perf_counter() - started:.1f}s   out: {out_dir}")
    print("\nDRY RUN OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
