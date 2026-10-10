#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""X13: the causal cure test — training-free hallucination removal via mean replacement.

For each selected COCO val2014 image the model captions greedily (baseline). The
caption's category mentions are matched against that image's ground-truth categories
from ``instances_val2014.json``: a mentioned category with no annotation is a
HALLUCINATION, a mentioned category with an annotation is GROUNDED. For every
hallucinated category and every edit layer we regenerate greedily under a
mean-replacement ablation of that category's J-lens direction — ``apply_edit`` with
``mode="ablate"`` and ``mean=m``: ``x' = x + P_v(m - x)`` (Belrose App. D), i.e. the
residual's ``v``-component is swapped for the mean residual's, so at ``alpha=1`` the
``(x - m)`` projection along ``v`` is exactly zero.

The mean per layer comes from the SAME baseline pass: the baseline greedy decode is
replayed teacher-forced in one forward — the generated ids are appended to the prompt
ids at the token level (nothing is re-tokenized) and an ActivationRecorder on the edit
layers captures the block outputs; the replay runs on ``inputs_embeds`` with the
prompt's image features merged once, mirroring the decode steps exactly (see
:func:`mean_residuals`). The model is causal and decoding greedy, so the replay
carries the baseline pass's own residuals (identical math; only GEMM-shape
reassociation can differ in low-order bits).

Edited positions (mirroring what the editor edits): ``positions="last"`` edits the
residual at the last prompt position during the prefill forward and at each
single-token decode step during generation — in the teacher-forced replay exactly the
slice ``[prompt_len - 1 : prompt_len + n_new]``. The per-layer mean is the average of
the recorded residuals over that slice.

Per hallucination x layer arm we score: ``removed`` (the hallucinated category word is
absent from the edited caption), grounded-category retention (fraction of the image's
grounded categories still present), the token-length ratio edited/baseline, and a
clean-cure flag (removed with no grounded category lost). A CONTROL arm runs the same
mean-replacement ablation on a GROUNDED category's direction (first in sorted order):
if the cure is concept-specific, ablating a real category's direction should damage the
caption (that category drops out and retention drops) where ablating a hallucinated one
cures without collateral loss.

Matching details: category names match as word-boundary phrases with an optional plural
suffix (``dog`` ~ ``dogs``; irregular plurals like "people"/"mice" are a known miss);
matches are resolved longest-first and each match masks its text span, so "a hot dog"
counts as ``hot dog`` and NOT as ``dog`` (COCO carries both, plus ``bear``/``teddy
bear``). Multi-word names map to the final sub-token's lens vector (the edit path's
``_token_id`` convention). Token strings keep the caption's leading space.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

import vlm_lens  # noqa: E402, F401  # installs the vendored jlens path: must precede jlens
from jlens.hooks import ActivationRecorder  # noqa: E402
from vlm_lens.artifacts import load_lens_set  # noqa: E402
from vlm_lens.data.captions import (  # noqa: E402
    PROMPT_TEMPLATE,
    list_coco_images,
    prompt_template_hash,
    prompt_text,
    select_images,
)
from vlm_lens.interventions import ResidualEdit, generate_with_edits  # noqa: E402
from vlm_lens.models.llava import LlavaLensModel, MultimodalBatch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lens-dir", default="/data/anhnq/vlm-lens-out/validation/s2-merged/artifacts"
    )
    parser.add_argument("--images-dir", default="/data/baodq/coco2014/val2014")
    parser.add_argument(
        "--annotations",
        default="/data/baodq/coco2014/annotations/instances_val2014.json",
        help="instances_val2014.json: per-image ground-truth categories + the 80 names",
    )
    parser.add_argument("--n-images", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0, help="image-selection seed")
    parser.add_argument("--question", default="Describe this image in detail.")
    parser.add_argument(
        "--layers", default="16,24,28,30", help="edit layers (one arm per layer)"
    )
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument(
        "--backend", choices=("hf-llava", "tiny"), default="hf-llava",
        help="model backend: 'hf-llava' (default, CUDA) or the tiny CPU smoke fixture "
        "(tiny has 3 layers, so pass e.g. --layers 1,2)",
    )
    parser.add_argument("--json", required=True)
    return parser.parse_args()


def load_ground_truth(
    annotations_path: str | Path,
) -> tuple[dict[int, str], dict[int, set[int]]]:
    """Category id -> name, and image id -> set of annotated category ids.

    ``instances_val2014.json`` is loaded wholesale (the segmentation polygons dominate
    its size); only the two small maps are kept.
    """
    data = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
    names = {int(cat["id"]): str(cat["name"]) for cat in data["categories"]}
    by_image: dict[int, set[int]] = {}
    for ann in data["annotations"]:
        by_image.setdefault(int(ann["image_id"]), set()).add(int(ann["category_id"]))
    return names, by_image


def image_id_of(path: Path) -> int:
    """``COCO_val2014_000000123456.jpg`` -> ``123456`` (the last digit run in the stem)."""
    digits = re.findall(r"\d+", path.stem)
    if not digits:
        raise ValueError(f"cannot read a COCO image id from {path.name}")
    return int(digits[-1])


def category_in_caption(name: str, text: str) -> bool:
    """Word-boundary match on lowercased text, plural suffix allowed (``dog`` ~ ``dogs``)."""
    return re.search(rf"\b{re.escape(name.strip())}(?:s|es)?\b", text.lower()) is not None


def detect_mentions(names: dict[int, str], text: str) -> list[str]:
    """Mentioned COCO category names, longest first, each match masking its span.

    Longest-first masking keeps "hot dog" from also counting as a ``dog`` mention (and
    "teddy bear" as a ``bear``) while still matching a genuinely separate word elsewhere
    in the caption.
    """
    masked = text.lower()
    mentioned: list[str] = []
    for name in sorted({n.strip() for n in names.values()}, key=len, reverse=True):
        pattern = rf"\b{re.escape(name)}(?:s|es)?\b"
        if re.search(pattern, masked) is not None:
            mentioned.append(name)
            masked = re.sub(pattern, " ", masked)
    return sorted(mentioned)


def mean_residuals(
    model: LlavaLensModel,
    batch: MultimodalBatch,
    new_token_ids: Sequence[int],
    layers: Sequence[int],
) -> dict[int, torch.Tensor]:
    """Per-layer mean residual at the edited positions, replaying the baseline pass.

    The baseline greedy decode is replayed teacher-forced: the generated ids are
    appended to the prompt ids at the token level (no re-tokenization) and one forward
    is recorded with an ActivationRecorder on ``layers``. Causal model + greedy decode
    means the recorded residuals are the baseline pass's own residuals.

    The replay runs on ``inputs_embeds``: the image features are merged into the
    prompt's placeholder positions once via the model's own vision path (the prefill's
    ``masked_scatter``), while generated positions keep their raw token embeddings —
    exactly what the decode steps saw, since ``generate`` passes ``pixel_values`` only
    at prefill. Re-passing ``pixel_values`` instead would let a generated id equal to
    the image-token id trip HF's placeholder-count check.

    Edited positions: ``positions="last"`` edits the last prompt position at the prefill
    forward and each single-token decode step during generation — here exactly the
    slice ``[prompt_len - 1 : prompt_len + n_new]``. The mean is over that slice.
    """
    prompt_len = batch.seq_len
    n_new = len(new_token_ids)
    device = batch.input_ids.device
    new_ids = torch.as_tensor(new_token_ids, dtype=batch.input_ids.dtype, device=device)
    input_ids = torch.cat([batch.input_ids, new_ids.unsqueeze(0)], dim=1)
    attention_mask = torch.cat(
        [batch.attention_mask, torch.ones(1, n_new, dtype=torch.long, device=device)], dim=1
    )

    mm = model.hf_model.model
    inputs_embeds = mm.get_input_embeddings()(input_ids)
    if batch.pixel_values is not None:
        pooler = mm.get_image_features(pixel_values=batch.pixel_values, return_dict=True)
        features = pooler.pooler_output
        if isinstance(features, list):  # per-image list on current transformers
            features = torch.cat(features, dim=0)
        features = features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        prompt_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        prompt_mask[:, :prompt_len] = batch.image_token_mask
        inputs_embeds = inputs_embeds.masked_scatter(prompt_mask.unsqueeze(-1), features)

    with torch.no_grad(), ActivationRecorder(model.layers, at=layers) as recorder:
        mm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False)
    means: dict[int, torch.Tensor] = {}
    for layer in layers:
        residual = recorder.activations[layer]  # [1, prompt_len + n_new, d]
        means[layer] = residual[0, prompt_len - 1 :, :].float().mean(dim=0).cpu()
    return means


def arm_row(
    kind: str,
    name: str,
    layer: int,
    spec: ResidualEdit,
    out: dict,
    baseline_text: str,
    baseline_ids: list[int],
    mentioned: Sequence[str],
    grounded: Sequence[str],
) -> dict:
    """One generation arm's record + scores (shared by hallucination and control arms)."""
    text = out["text"]
    removed_categories = [c for c in mentioned if not category_in_caption(c, text)]
    retention = (
        sum(1 for c in grounded if category_in_caption(c, text)) / len(grounded)
        if grounded
        else None
    )
    return {
        "kind": kind,
        "category": name,
        "layer": layer,
        "edit": spec.to_json(),
        "text": text,
        "changed": out["token_ids"] != baseline_ids,
        "n_tokens": len(out["token_ids"]),
        "token_length_ratio": len(out["token_ids"]) / max(len(baseline_ids), 1),
        "removed": not category_in_caption(name, text),
        "removed_categories": removed_categories,
        "grounded_retention": retention,
        "n_edit_forwards": out["n_edit_forwards"],
    }


def _mean(values: Sequence[float | bool]) -> float | None:
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)


def per_layer_summary(samples: Sequence[dict], layers: Sequence[int]) -> dict[str, dict]:
    """Cure/control aggregates per layer over all hallucination/control arms."""
    arms = [arm for sample in samples for arm in sample["arms"]]
    summary: dict[str, dict] = {}
    for layer in layers:
        hall = [a for a in arms if a["layer"] == layer and a["kind"] == "hallucination"]
        ctrl = [a for a in arms if a["layer"] == layer and a["kind"] == "control"]
        clean = [
            a["removed"]
            and (a["grounded_retention"] is None or a["grounded_retention"] == 1.0)
            for a in hall
        ]
        summary[str(layer)] = {
            "n_halluc_arms": len(hall),
            "removed_rate": _mean([a["removed"] for a in hall]),
            "grounded_retention": _mean(
                [a["grounded_retention"] for a in hall if a["grounded_retention"] is not None]
            ),
            "token_length_ratio": _mean([a["token_length_ratio"] for a in hall]),
            "changed_rate": _mean([a["changed"] for a in hall]),
            "clean_cure_rate": _mean(clean),
            "control": {
                "n_arms": len(ctrl),
                "removed_rate": _mean([a["removed"] for a in ctrl]),
                "grounded_retention": _mean(
                    [a["grounded_retention"] for a in ctrl if a["grounded_retention"] is not None]
                ),
                "token_length_ratio": _mean([a["token_length_ratio"] for a in ctrl]),
                "changed_rate": _mean([a["changed"] for a in ctrl]),
            },
        }
    return summary


def main() -> int:
    args = parse_args()
    layers = [int(part) for part in args.layers.split(",")]

    lenses, _ = load_lens_set(args.lens_dir)
    lens = lenses["text"]
    if args.backend == "tiny":
        from vlm_lens.models.tiny_llava import TinyLlavaConfig, build_tiny_llava

        hf_model, processor = build_tiny_llava(TinyLlavaConfig())
        model = LlavaLensModel(hf_model, processor)
    else:
        model = LlavaLensModel.from_pretrained(
            dtype=torch.bfloat16, device="cuda", local_files_only=True
        )

    names, gt_by_image = load_ground_truth(args.annotations)
    images = select_images(list_coco_images(args.images_dir), args.n_images, seed=args.seed)

    report: dict[str, object] = {
        "experiment": "x13_cure_test",
        "lens_dir": args.lens_dir,
        "lens_mask": "text",
        "n_prompts": int(lens.n_prompts),
        "backend": args.backend,
        "images_dir": args.images_dir,
        "annotations": args.annotations,
        "n_categories": len(names),
        "n_images": len(images),
        "seed": args.seed,
        "question": args.question,
        "prompt_template": PROMPT_TEMPLATE,
        "prompt_template_hash": prompt_template_hash(),
        "layers": layers,
        "max_new_tokens": args.max_new_tokens,
        "mean_positions": (
            "positions='last' edits the residual at the last prompt position during the "
            "prefill forward and at each single-token decode step during generation; the "
            "per-layer mean is the average block output over the same slice "
            "[prompt_len - 1 : prompt_len + n_new] of the baseline pass replayed "
            "teacher-forced on inputs_embeds (generated ids appended at the token "
            "level, prompt image features merged once, ActivationRecorder on the edit "
            "layers)"
        ),
    }
    print(f"== X13 cure test: {len(images)} images, layers={layers}, q={args.question!r} ==")

    samples: list[dict] = []
    for path in images:
        batch = model.encode_mm(prompt_text(args.question), path, max_length=1536)
        baseline = generate_with_edits(model, None, batch, max_new_tokens=args.max_new_tokens)
        baseline_text = baseline["text"]
        baseline_ids = baseline["token_ids"]

        image_id = image_id_of(path)
        gt_ids = gt_by_image.get(image_id)
        if gt_ids is None:
            raise ValueError(f"image {path.name} (id {image_id}) is missing from the annotations")
        gt_names = {names[cid] for cid in gt_ids}
        mentioned = detect_mentions(names, baseline_text)
        hallucinated = [name for name in mentioned if name not in gt_names]
        grounded = [name for name in mentioned if name in gt_names]

        # The means come from the same baseline pass (see mean_residuals for the
        # position semantics); they are shared by every arm of this sample.
        means = mean_residuals(model, batch, baseline_ids, layers)

        sample_record: dict[str, object] = {
            "image": path.name,
            "image_id": image_id,
            "question": args.question,
            "baseline_text": baseline_text,
            "baseline_n_tokens": len(baseline_ids),
            "n_gt_categories": len(gt_names),
            "mentioned": mentioned,
            "hallucinated": hallucinated,
            "grounded": grounded,
            "mean_norms": {str(layer): float(means[layer].norm()) for layer in layers},
            "arms": [],
        }
        arms: list[dict] = sample_record["arms"]
        for name in hallucinated:  # sorted: deterministic arm order
            for layer in layers:
                spec = ResidualEdit(
                    layer=layer, mode="ablate", token=f" {name}", alpha=1.0, mean=means[layer]
                )
                out = generate_with_edits(
                    model, lens, batch, edits=[spec], max_new_tokens=args.max_new_tokens
                )
                arms.append(
                    arm_row(
                        "hallucination", name, layer, spec, out, baseline_text, baseline_ids,
                        mentioned, grounded,
                    )
                )
        if grounded:  # control: same treatment on a real category's direction
            control_name = grounded[0]
            for layer in layers:
                spec = ResidualEdit(
                    layer=layer, mode="ablate", token=f" {control_name}", alpha=1.0,
                    mean=means[layer],
                )
                out = generate_with_edits(
                    model, lens, batch, edits=[spec], max_new_tokens=args.max_new_tokens
                )
                arms.append(
                    arm_row(
                        "control", control_name, layer, spec, out, baseline_text, baseline_ids,
                        mentioned, grounded,
                    )
                )
        samples.append(sample_record)
        print(
            f"  {path.name}: gt={len(gt_names)} mentioned={len(mentioned)} "
            f"halluc={hallucinated or '-'} grounded={grounded or '-'}"
        )

    report["samples"] = samples
    report["per_layer"] = per_layer_summary(samples, layers)
    n_halluc = sum(len(sample["hallucinated"]) for sample in samples)
    report["summary"] = {
        "n_images": len(samples),
        "n_with_hallucination": sum(1 for sample in samples if sample["hallucinated"]),
        "n_hallucinations": n_halluc,
        "n_grounded_mentions": sum(len(sample["grounded"]) for sample in samples),
        "halluc_per_caption": n_halluc / max(len(samples), 1),
    }

    summary = report["summary"]
    print(
        f"\n== digest: {summary['n_images']} images, "
        f"{summary['n_with_hallucination']} with a hallucination, "
        f"{summary['n_hallucinations']} hallucinated / "
        f"{summary['n_grounded_mentions']} grounded mentions =="
    )

    def fmt(value: float | None) -> str:
        return f"{value:>10.3f}" if value is not None else f"{'-':>10}"

    print(
        f"{'layer':>6}{'n_hall':>7}{'removed':>10}{'retent':>10}{'len_rat':>10}"
        f"{'clean':>10} | {'ctrl_n':>7}{'c_rem':>10}{'c_ret':>10}{'c_len':>10}"
    )
    for layer in layers:
        cell = report["per_layer"][str(layer)]
        control = cell["control"]
        print(
            f"{layer:>6}{cell['n_halluc_arms']:>7}"
            f"{fmt(cell['removed_rate'])}"
            f"{fmt(cell['grounded_retention'])}"
            f"{fmt(cell['token_length_ratio'])}"
            f"{fmt(cell['clean_cure_rate'])} | "
            f"{control['n_arms']:>7}"
            f"{fmt(control['removed_rate'])}"
            f"{fmt(control['grounded_retention'])}"
            f"{fmt(control['token_length_ratio'])}"
        )

    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.json} ({len(samples)} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
