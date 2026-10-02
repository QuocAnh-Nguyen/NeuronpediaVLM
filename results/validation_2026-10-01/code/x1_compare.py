#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""X1: does the target-mask variant (text-only targets) change the caption lens rows?

Two parts, because the ALD estimator conjoins them:

1. **Bit-level (decisive)** — one real sample, on the causally-disjoint *subset* (D3):
   text sources after the image block cannot causally reach image targets, and the
   estimator seeds cotangents by position, so their rows must be *bit-identical* between
   ``target_mask=all`` and ``target_mask=text``; the ``image`` rows must differ. Checked
   per layer with ``torch.equal``. The library's ``text`` mask also covers the ``USER:``
   tokens *before* the placeholder, which do reach image targets and legitimately change
   the aggregate row (D3), so the check trims the prompt to start at ``<image>``.
2. **Lens-level** — the same 20-sample shard fitted both ways; per mask, relative
   Frobenius distance, cosine and the best-fit scale, to quantify how much the *image*
   rows move (the text rows must reproduce the bit-level result at the aggregate).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

import vlm_lens  # noqa: E402, F401  # installs the vendored jlens path: must precede jlens
from vlm_lens._batch import as_batch  # noqa: E402
from vlm_lens.artifacts import load_lens_set  # noqa: E402
from vlm_lens.data.manifest import read_manifest  # noqa: E402
from vlm_lens.fitting import jacobian_for_sample  # noqa: E402
from vlm_lens.models.llava import IMAGE_PLACEHOLDER, LlavaLensModel  # noqa: E402
from vlm_lens.positions import build_position_masks  # noqa: E402

MASKS = ("text", "image", "all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir-all", required=True, help="artifacts fitted with target_mask=all")
    parser.add_argument("--dir-text", required=True, help="artifacts fitted with target_mask=text")
    parser.add_argument("--manifest", required=True, help="the shared shard manifest")
    parser.add_argument("--sample-index", type=int, default=0, help="sample for the bit-level check")
    parser.add_argument("--json", required=True)
    parser.add_argument("--layers", default="0,8,16,24,30")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument(
        "--backend", choices=("hf-llava", "tiny"), default="hf-llava",
        help="model backend: 'hf-llava' (default, CUDA) or the tiny CPU smoke fixture",
    )
    return parser.parse_args()


def _compare(A: torch.Tensor, B: torch.Tensor) -> dict[str, float | bool]:
    A, B = A.float(), B.float()
    scale = float((A * B).sum() / B.pow(2).sum().clamp_min(1e-12))
    return {
        "bit_identical": bool(torch.equal(A, B)),
        "max_abs_diff": float((A - B).abs().max()),
        "rel_fro": float((A - B).norm() / B.norm().clamp_min(1e-12)),
        "cosine": float(torch.nn.functional.cosine_similarity(A.reshape(-1), B.reshape(-1), dim=0)),
        "scale": scale,
        "rel_fro_after_scale": float((A - scale * B).norm() / B.norm().clamp_min(1e-12)),
    }


def bit_check(model, sample, layers, max_seq_len: int) -> dict[str, object]:
    """Per-sample Jacobians under both target masks; pass the causally-disjoint prompt."""

    def compute(target_mask: str) -> dict[int, torch.Tensor]:
        jacobians, _info = jacobian_for_sample(
            model,
            sample,
            layers,
            masks=MASKS,
            target_mask=target_mask,
            skip_first=1,
            max_seq_len=max_seq_len,
        )
        return {layer: jacobians[mask][layer] for mask in MASKS for layer in layers if mask in jacobians}

    jac_all = jacobian_for_sample(
        model, sample, layers, masks=MASKS, target_mask="all", skip_first=1, max_seq_len=max_seq_len
    )[0]
    jac_text = jacobian_for_sample(
        model, sample, layers, masks=MASKS, target_mask="text", skip_first=1, max_seq_len=max_seq_len
    )[0]
    per_mask: dict[str, dict[str, dict]] = {}
    for mask in MASKS:
        if mask not in jac_all or mask not in jac_text:
            continue
        per_mask[mask] = {
            str(layer): _compare(jac_all[mask][layer], jac_text[mask][layer]) for layer in layers
        }
    return {
        "per_mask": per_mask,
        "text_rows_bit_identical": all(
            entry["bit_identical"] for entry in per_mask.get("text", {}).values()
        ),
        "image_rows_bit_identical": all(
            entry["bit_identical"] for entry in per_mask.get("image", {}).values()
        ),
    }


def main() -> int:
    args = parse_args()
    layers = [int(part) for part in args.layers.split(",")]
    lens_all, prov_all = load_lens_set(args.dir_all)
    lens_text, prov_text = load_lens_set(args.dir_text)
    report: dict[str, object] = {
        "dir_all": args.dir_all,
        "dir_text": args.dir_text,
        "n_prompts": {
            "all": {mask: int(lens.n_prompts) for mask, lens in lens_all.items()},
            "text": {mask: int(lens.n_prompts) for mask, lens in lens_text.items()},
        },
        "provenance_target_mask": {
            "all": {
                mask: prov_all.get("fit_config", {}).get("target_mask") for mask in lens_all
            },
            "text": {
                mask: prov_text.get("fit_config", {}).get("target_mask") for mask in lens_text
            },
        },
    }

    samples = read_manifest(args.manifest)
    sample = samples[args.sample_index]
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    if args.backend == "tiny":
        from vlm_lens.models.tiny_llava import TinyLlavaConfig, build_tiny_llava

        hf_model, processor = build_tiny_llava(TinyLlavaConfig())
        model = LlavaLensModel(hf_model, processor)
    else:
        model = LlavaLensModel.from_pretrained(dtype=dtype, device="cuda", local_files_only=True)
    full_batch = as_batch(model, sample, args.max_seq_len)
    full_masks = build_position_masks(
        full_batch.input_ids, model.image_token_id, skip_first=1, masks=MASKS
    )
    first_image = int(
        (full_batch.input_ids[0] == model.image_token_id).nonzero(as_tuple=True)[0][0]
    )
    pre_image_text = int(full_masks["text"][:first_image].sum())

    # D3: trim the prompt to start at the placeholder so the ``text`` mask is exactly the
    # causally-disjoint block after the image; the pre-image ``USER:`` tokens would
    # otherwise contaminate the aggregate row (they can reach the image targets).
    marker = sample.text.find(IMAGE_PLACEHOLDER)
    causal_sample = replace(sample, text=sample.text[marker:]) if marker > 0 else sample
    causal_masks = build_position_masks(
        as_batch(model, causal_sample, args.max_seq_len).input_ids,
        model.image_token_id,
        skip_first=1,
        masks=MASKS,
    )
    report["sample_geometry"] = {
        "sample": sample.sample_id,
        "question": sample.meta.get("question"),
        "mask_positions": {name: int(mask.sum()) for name, mask in causal_masks.items()},
        "full_prompt_mask_positions": {
            name: int(mask.sum()) for name, mask in full_masks.items()
        },
        "pre_image_text_positions": pre_image_text,
        "prefix_chars_dropped": max(marker, 0),
    }
    check = bit_check(model, causal_sample, layers, args.max_seq_len)
    report["bit_check"] = check
    print("== X1 bit-level (one real sample, both target masks) ==")
    for mask, per_layer in check["per_mask"].items():
        for layer, entry in per_layer.items():
            print(
                f"{mask:<6} L{layer:<3} equal={entry['bit_identical']} "
                f"max_abs={entry['max_abs_diff']:.3e} rel_fro={entry['rel_fro']:.3e} "
                f"cos={entry['cosine']:.6f}"
            )
    print(
        f"VERDICT (causally-disjoint text subset; {pre_image_text} pre-image text "
        "positions dropped):",
        "text rows bit-identical" if check["text_rows_bit_identical"] else "TEXT ROWS DIFFER",
        "|",
        "image rows bit-identical" if check["image_rows_bit_identical"] else "image rows differ",
    )

    lens_compare: dict[str, dict] = {}
    for mask in MASKS:
        if mask not in lens_all or mask not in lens_text:
            continue
        lens_compare[mask] = {
            str(layer): _compare(lens_all[mask].jacobians[layer], lens_text[mask].jacobians[layer])
            for layer in layers
            if layer in lens_all[mask].jacobians and layer in lens_text[mask].jacobians
        }
    report["lens_comparison"] = lens_compare
    print("\n== X1 lens-level (same 20-image shard, two target masks) ==")
    for mask, per_layer in lens_compare.items():
        for layer, entry in per_layer.items():
            print(
                f"{mask:<6} L{layer:<3} equal={entry['bit_identical']} "
                f"rel_fro={entry['rel_fro']:.3e} cos={entry['cosine']:.6f} "
                f"scale={entry['scale']:.4f} rel_after_scale={entry['rel_fro_after_scale']:.3e}"
            )

    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
