#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""X12: cross-image residual patching — localize where visual content commits.

Question: at which residual layer has the model already committed to an image's
objects? Zero fitting and no lens involved: for COCO image pairs (A, B) whose
annotated category sets are disjoint, cache every swept block's output from both
images' forward passes (ActivationRecorder), then regenerate A's caption with a
forward hook on ``model.layers[l]`` that REPLACES the block-output rows at
selected positions with B's cached rows. If B's object words start appearing in
A's caption (and A's vanish), the layer-``l`` residual at the patched positions
is where A's visual content had committed.

Variants — which rows get replaced (donor rows come from the same positions of
B's cached run; the prompt text is identical and each run carries one image, so
positions align 1:1 and are asserted to):
  image       all IMAGE-token positions: the fused visual patch states.
  image-last  only the LAST image-token position: the visual state closest to
              the following text.
  text        only TEXT (non-image) positions — the control. Same prompt, same
              positions; any effect must be B's visual context leaking through.

The patch fires on the prefill forward only: with ``use_cache=True`` decode steps
carry a single token each (the patched positions no longer exist), while the
replaced rows propagate through attention via the KV cache — the exact semantics
of "patch the residual stream at layer l". The prefill patch count is asserted
== 1 per generation.

Metrics per (pair, layer, variant): caption_changed (token-level), first-diff
token index, A-object-disappearance rate (A categories present in A's baseline
caption but absent after patching, over those present in the baseline; 0.0 when
the baseline mentions none), and B-object-appearance rate (B categories present
after patching, over all of B's annotated categories). As a magnitude diagnostic
each row also records ``row_rel_delta`` — the mean relative L2 distance between
A's and B's cached rows at the patched positions — so a null effect with
far-apart rows is a real finding and one with near-identical rows is expected.
word-boundary on the COCO category names, no stemming ("dog" does not match
"dogs"; COCO lists "ski" and "skis" separately, so exact whole words are the safe
convention).

Output: JSON {meta, pairs, results[], digest} plus a printed digest table of
means per layer x variant. ``--backend tiny`` runs the whole pipeline on the tiny
CPU fixture (pass ``--layers 0,1,2`` — the fixture has fewer layers than the
default sweep).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
import vlm_lens  # noqa: E402, F401  # installs the vendored jlens path: must precede jlens
from jlens.hooks import ActivationRecorder  # noqa: E402

from vlm_lens.data.captions import (  # noqa: E402
    DEFAULT_QUESTIONS,
    PROMPT_TEMPLATE,
    list_coco_images,
    prompt_text,
)
from vlm_lens.interventions import generate_with_edits  # noqa: E402
from vlm_lens.models.llava import LlavaLensModel, MultimodalBatch  # noqa: E402

#: Which block-output rows the patch replaces.
VARIANTS: tuple[str, ...] = ("image", "image-last", "text")
#: Default residual-layer sweep.
DEFAULT_LAYERS: tuple[int, ...] = (0, 8, 12, 16, 20, 24, 28, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir", default="/data/baodq/coco2014/val2014",
        help="directory of COCO images (val2014 *.jpg)",
    )
    parser.add_argument(
        "--instances-json", default="/data/baodq/coco2014/annotations/instances_val2014.json",
        help="COCO instance annotations (category ground truth per image)",
    )
    parser.add_argument("--n-pairs", type=int, default=8, help="number of disjoint (A, B) pairs")
    parser.add_argument(
        "--layers", default=",".join(str(layer) for layer in DEFAULT_LAYERS),
        help="comma-separated residual layers to patch",
    )
    parser.add_argument(
        "--variants", default=",".join(VARIANTS),
        help="comma-separated subset of: " + ",".join(VARIANTS),
    )
    parser.add_argument(
        "--question", default=DEFAULT_QUESTIONS[0],
        help="question substituted into the corpus prompt template",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--backend", choices=("hf-llava", "tiny"), default="hf-llava",
        help="model backend: 'hf-llava' (default, CUDA) or the tiny CPU smoke fixture",
    )
    parser.add_argument("--dtype", default="float32", help="torch dtype name for the model weights")
    parser.add_argument("--json", required=True, help="output JSON path (meta, pairs, results, digest)")
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> LlavaLensModel:
    """The patching model: the HF checkpoint (default, CUDA) or the tiny CPU fixture."""
    if args.backend == "tiny":
        from vlm_lens.models.tiny_llava import TinyLlavaConfig, build_tiny_llava

        hf_model, processor = build_tiny_llava(TinyLlavaConfig())
        return LlavaLensModel(hf_model, processor)
    return LlavaLensModel.from_pretrained(
        dtype=getattr(torch, args.dtype), device="cuda", local_files_only=True
    )


# --------------------------------------------------------------------------- COCO pairs


def load_image_categories(instances_json: str | Path) -> dict[str, set[str]]:
    """``file_name`` -> the COCO category names annotated on that image."""
    data = json.loads(Path(instances_json).read_text(encoding="utf-8"))
    names = {int(category["id"]): str(category["name"]) for category in data["categories"]}
    id_to_file = {int(image["id"]): str(image["file_name"]) for image in data["images"]}
    out: dict[str, set[str]] = {file: set() for file in id_to_file.values()}
    for ann in data["annotations"]:
        file = id_to_file.get(int(ann["image_id"]))
        name = names.get(int(ann["category_id"]))
        if file is not None and name is not None:
            out[file].add(name)
    return out


def select_pairs(
    files: list[Path], cats_by_file: dict[str, set[str]], n_pairs: int
) -> list[tuple[Path, Path]]:
    """Greedy first-disjoint pairing in sorted file order (deterministic, no seed).

    Walk the sorted file list; for each not-yet-paired image A with a non-empty
    category set, pair it with the first not-yet-paired later image B whose
    category set is disjoint from A's. Images without annotations never enter a
    pair (their rates would be trivial). Returns at most ``n_pairs`` pairs.
    """
    used = [False] * len(files)
    pairs: list[tuple[Path, Path]] = []
    for index_a, path_a in enumerate(files):
        if used[index_a]:
            continue
        cats_a = cats_by_file.get(path_a.name)
        if not cats_a:
            continue
        for index_b in range(index_a + 1, len(files)):
            if used[index_b]:
                continue
            cats_b = cats_by_file.get(files[index_b].name)
            if not cats_b or not cats_a.isdisjoint(cats_b):
                continue
            pairs.append((path_a, files[index_b]))
            used[index_a] = used[index_b] = True
            break
        if len(pairs) >= n_pairs:
            break
    return pairs


def category_in_text(name: str, text: str) -> bool:
    """Lowercase word-boundary match of a COCO category name in ``text``.

    The documented matcher: ``\\b``-bounded whole words only, no stemming — a
    substring like "cat" inside "category" must not count as the "cat" category.
    """
    return re.search(r"\b" + re.escape(name.lower()) + r"\b", text.lower()) is not None


def mentioned(categories: set[str], text: str) -> list[str]:
    """Sorted category names matched in ``text``."""
    return sorted(name for name in categories if category_in_text(name, text))


# --------------------------------------------------------------------------- patching


def assert_aligned(batch_a: MultimodalBatch, batch_b: MultimodalBatch) -> None:
    """The patch is row-indexed, so both runs must agree on the sequence layout."""
    if batch_a.seq_len != batch_b.seq_len:
        raise ValueError(
            f"sequence lengths diverge: A={batch_a.seq_len} B={batch_b.seq_len}; "
            "the same prompt with n_images=1 must tokenize identically"
        )
    if not torch.equal(batch_a.image_token_mask, batch_b.image_token_mask):
        raise ValueError("image-token positions differ between the A and B runs")


def variant_positions(variant: str, batch: MultimodalBatch) -> torch.Tensor:
    """Boolean ``[seq_len]`` selector of the rows the patch replaces."""
    image_mask = batch.image_token_mask[0]
    if variant == "image":
        return image_mask
    if variant == "image-last":
        if not bool(image_mask.any()):
            raise ValueError("no image tokens in the batch; 'image-last' is undefined")
        positions = torch.zeros_like(image_mask)
        positions[int(torch.nonzero(image_mask).max())] = True
        return positions
    if variant == "text":
        return ~image_mask
    raise ValueError(f"unknown variant {variant!r}; choose from {list(VARIANTS)}")


def cache_block_outputs(
    model: LlavaLensModel, batch: MultimodalBatch, layers: list[int]
) -> dict[int, torch.Tensor]:
    """One recorded no-grad forward per image: the swept blocks' output rows.

    Tensors are detached and cloned so nothing downstream keeps the forward's
    graph alive; dtype and device are exactly the block's own.
    """
    with torch.no_grad(), ActivationRecorder(model.layers, at=layers) as recorder:
        model.forward_mm(batch)
    return {layer: recorder.activations[layer].detach().clone() for layer in layers}


def row_rel_delta(rows_a: torch.Tensor, rows_b: torch.Tensor) -> float:
    """Mean relative L2 distance between A's and B's cached rows (same positions).

    Magnitude context for a (near-)null patch effect: with ``row_rel_delta``
    near 0 the donor rows were indistinguishable from A's, so the caption could
    not have changed; with it large, a null effect is a real finding.
    """
    delta = (rows_b - rows_a).norm(dim=-1).mean()
    reference = rows_a.norm(dim=-1).mean().clamp_min(1e-12)
    return float(delta / reference)


class BlockPatcher:
    """Forward hook on one block: replace prompt-position rows with donor rows.

    Fires on the prefill forward only (sequence length == the prompt length);
    decode steps carry one token each and return the output untouched. The
    replaced rows reach later steps through the KV cache.
    """

    def __init__(
        self,
        model: LlavaLensModel,
        layer: int,
        donor: torch.Tensor,
        positions: torch.Tensor,
        prompt_len: int,
    ) -> None:
        self._block = model.layers[layer]
        self._donor = donor
        self._positions = positions
        self._prompt_len = prompt_len
        self._handle: Any = None
        self.n_prefill_patches = 0

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        tensor = output if torch.is_tensor(output) else output[0]
        if int(tensor.shape[1]) != self._prompt_len:
            return output
        patched = tensor.clone()
        patched[0, self._positions] = self._donor[0, self._positions]
        self.n_prefill_patches += 1
        if torch.is_tensor(output):
            return patched
        return (patched, *output[1:])

    def __enter__(self) -> BlockPatcher:
        self._handle = self._block.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc: Any) -> None:
        self._handle.remove()
        self._handle = None
        return None


@torch.no_grad()
def generate_patched(
    model: LlavaLensModel,
    batch: MultimodalBatch,
    layer: int,
    donor: torch.Tensor,
    positions: torch.Tensor,
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Greedy caption for ``batch`` (image A) with layer-``l`` rows replaced by ``donor``."""
    patcher = BlockPatcher(model, layer, donor, positions, batch.seq_len)
    with patcher:
        generated = model.hf_model.generate(
            **batch.hf_kwargs(),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    if patcher.n_prefill_patches != 1:
        raise RuntimeError(
            f"expected the patch to fire on exactly one prefill forward, got "
            f"{patcher.n_prefill_patches} (layer={layer})"
        )
    new_tokens = generated[0, batch.seq_len :].detach().cpu()
    return {
        "text": model._decode(new_tokens),
        "token_ids": [int(token) for token in new_tokens],
    }


# --------------------------------------------------------------------------- reporting


def build_digest(
    results: list[dict[str, Any]], layers: list[int], variants: list[str]
) -> dict[str, dict[str, dict[str, Any]]]:
    """Mean metrics per layer x variant over the pair rows."""
    digest: dict[str, dict[str, dict[str, Any]]] = {}
    for layer in layers:
        per_variant: dict[str, dict[str, Any]] = {}
        for variant in variants:
            rows = [row for row in results if row["layer"] == layer and row["variant"] == variant]
            if not rows:
                continue
            total = len(rows)
            diffs = [
                row["first_diff_token"]
                for row in rows
                if row["caption_changed"] and row["first_diff_token"] is not None
            ]
            per_variant[variant] = {
                "n": total,
                "changed_rate": sum(1 for row in rows if row["caption_changed"]) / total,
                "mean_first_diff_token": sum(diffs) / len(diffs) if diffs else None,
                "a_disappearance_rate": sum(
                    float(row["a_disappearance_rate"]) for row in rows
                ) / total,
                "b_appearance_rate": sum(float(row["b_appearance_rate"]) for row in rows) / total,
                "row_rel_delta": sum(float(row["row_rel_delta"]) for row in rows) / total,
            }
        digest[str(layer)] = per_variant
    return digest


def print_digest(digest: dict[str, dict[str, dict[str, Any]]], n_pairs: int) -> None:
    print(f"\n== X12 digest: means over {n_pairs} pairs (layer x variant) ==")
    header = (
        f"{'layer':>5}  {'variant':<11}  {'n':>3}  {'changed':>7}  "
        f"{'first_diff':>10}  {'a_vanish':>8}  {'b_appear':>8}  {'row_delta':>9}"
    )
    print(header)
    for layer, per_variant in digest.items():
        for variant, means in per_variant.items():
            first_diff = (
                "-"
                if means["mean_first_diff_token"] is None
                else f"{means['mean_first_diff_token']:.1f}"
            )
            print(
                f"{layer:>5}  {variant:<11}  {means['n']:>3}  {means['changed_rate']:>7.3f}  "
                f"{first_diff:>10}  {means['a_disappearance_rate']:>8.3f}  "
                f"{means['b_appearance_rate']:>8.3f}  {means['row_rel_delta']:>9.3f}"
            )


def main() -> int:
    args = parse_args()
    layers = [int(part) for part in args.layers.split(",") if part.strip()]
    variants = [part for part in args.variants.split(",") if part.strip()]
    unknown = [variant for variant in variants if variant not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; choose from {list(VARIANTS)}")

    model = load_model(args)
    invalid = [layer for layer in layers if not 0 <= layer < model.n_layers]
    if invalid:
        raise ValueError(
            f"layers {invalid} outside 0..{model.n_layers - 1} (backend {args.backend!r}, "
            f"n_layers={model.n_layers}); pass --layers within range"
        )

    files = list_coco_images(args.images_dir)
    cats_by_file = load_image_categories(args.instances_json)
    pairs = select_pairs(files, cats_by_file, args.n_pairs)
    if not pairs:
        raise ValueError(
            f"no disjoint-category pairs among {len(files)} images in {args.images_dir}"
        )
    if len(pairs) < args.n_pairs:
        print(f"warning: only {len(pairs)} disjoint pairs found (requested {args.n_pairs})")

    prompt = prompt_text(args.question)
    print(
        f"model ready: layers={model.n_layers} d_model={model.d_model} "
        f"image_seq_length={model.image_seq_length}; sweep={layers} variants={variants}"
    )
    print(f"prompt: {prompt!r}")

    report: dict[str, Any] = {
        "script": Path(__file__).name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": args.backend,
        "dtype": args.dtype,
        "images_dir": str(args.images_dir),
        "instances_json": str(args.instances_json),
        "n_pairs": len(pairs),
        "layers": layers,
        "variants": variants,
        "question": args.question,
        "prompt": prompt,
        "prompt_template": PROMPT_TEMPLATE,
        "max_new_tokens": args.max_new_tokens,
        "pair_selection": "greedy first-disjoint pairing in sorted file order (deterministic)",
        "matcher": "lowercase word-boundary regex on COCO category names, no stemming",
        "pairs": [],
        "results": [],
    }

    for pair_index, (path_a, path_b) in enumerate(pairs):
        cats_a = cats_by_file[path_a.name]
        cats_b = cats_by_file[path_b.name]
        batch_a = model.encode_mm(prompt, path_a)
        batch_b = model.encode_mm(prompt, path_b)
        assert_aligned(batch_a, batch_b)

        # One recorded forward per image; the sweep is cached once and reused for
        # every (layer, variant) patch of this pair.
        cache_a = cache_block_outputs(model, batch_a, layers)
        cache_b = cache_block_outputs(model, batch_b, layers)

        baseline_a = generate_with_edits(model, None, batch_a, max_new_tokens=args.max_new_tokens)
        baseline_b = generate_with_edits(model, None, batch_b, max_new_tokens=args.max_new_tokens)
        a_baseline = mentioned(cats_a, baseline_a["text"])

        report["pairs"].append(
            {
                "pair_id": pair_index,
                "image_a": {"file": path_a.name, "categories": sorted(cats_a)},
                "image_b": {"file": path_b.name, "categories": sorted(cats_b)},
                "seq_len": batch_a.seq_len,
                "n_image_tokens": batch_a.n_image_tokens,
                "baseline_a": {"text": baseline_a["text"], "token_ids": baseline_a["token_ids"]},
                "baseline_b": {"text": baseline_b["text"], "token_ids": baseline_b["token_ids"]},
                "baseline_a_mentioned": a_baseline,
                "baseline_b_mentioned": mentioned(cats_b, baseline_b["text"]),
            }
        )

        for layer in layers:
            donor = cache_b[layer]
            rows_a = cache_a[layer]
            for variant in variants:
                positions = variant_positions(variant, batch_a)
                out = generate_patched(
                    model, batch_a, layer, donor, positions, max_new_tokens=args.max_new_tokens
                )
                tokens = out["token_ids"]
                a_patched = mentioned(cats_a, out["text"])
                b_patched = mentioned(cats_b, out["text"])
                vanished = [name for name in a_baseline if name not in set(a_patched)]
                first_diff = next(
                    (
                        index
                        for index, (x, y) in enumerate(
                            zip(baseline_a["token_ids"], tokens, strict=False)
                        )
                        if x != y
                    ),
                    None,
                )
                report["results"].append(
                    {
                        "pair_id": pair_index,
                        "image_a": path_a.name,
                        "image_b": path_b.name,
                        "layer": layer,
                        "variant": variant,
                        "n_patched_positions": int(positions.sum()),
                        "row_rel_delta": row_rel_delta(rows_a[0, positions], donor[0, positions]),
                        "text": out["text"],
                        "caption_changed": tokens != baseline_a["token_ids"],
                        "first_diff_token": first_diff,
                        "a_objects_baseline": a_baseline,
                        "a_objects_patched": a_patched,
                        "b_objects_patched": b_patched,
                        "a_disappearance_rate": (
                            len(vanished) / len(a_baseline) if a_baseline else 0.0
                        ),
                        "b_appearance_rate": len(b_patched) / len(cats_b) if cats_b else 0.0,
                    }
                )
        print(
            f"  pair {pair_index}: A={path_a.name} B={path_b.name}; "
            f"A baseline={baseline_a['text'][:60]!r}"
        )

    report["digest"] = build_digest(report["results"], layers, variants)
    print_digest(report["digest"], len(pairs))

    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
