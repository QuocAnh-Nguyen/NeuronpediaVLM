#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Affine-correction census: per-layer bias/scale for the J-lens readout, from a corpus.

D37 measured that the purely-linear lens (``unembed(J_l @ h_l)``) degrades in the L17-24
region - a property of the fitted average-Jacobian map, not the model. The tuned-lens-style
fix is a per-layer affine correction,

    lens_l(h) = unembed(s_l * (J_l @ h + b_l)),

estimated forward-only from a corpus: with the same ActivationRecorder pattern the readout
uses, accumulate the layer-``l`` residuals ``h_l``, the final-layer pre-norm residuals
``h_final`` (same mask positions; the final residual does not depend on the layer, so its
moments are accumulated once), and, where ``J_l`` exists, the transported vectors
``t = J_l @ h_l``. Then per layer

    b_l = mean(h_final) - J_l @ mean(h_l)      (the affine map hits the corpus mean)
    s_l = mean(<t, h_final>) / mean(||t||^2)   (least-squares scalar, guarded)

with ``s_l = 1.0`` when ``mean(||t||^2) < 1e-12`` (a dead transport must not be scaled).
The census never backprops, so it runs anywhere the forward pass runs; the saved file feeds
``readout.lens_readout(bias=..., scale=...)`` and ``s2_eval --bias-dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402, I001  # the vlm_lens import must precede jlens: it installs the path
import vlm_lens  # noqa: E402, F401
from jlens.hooks import ActivationRecorder  # noqa: E402

from vlm_lens._batch import as_batch  # noqa: E402
from vlm_lens.artifacts import load_lens_set, save_bias  # noqa: E402
from vlm_lens.data.manifest import read_manifest  # noqa: E402
from vlm_lens.models.llava import LlavaLensModel  # noqa: E402
from vlm_lens.positions import build_position_masks  # noqa: E402

ESTIMATOR = (
    "b_l = mean(h_final) - J_l @ mean(h_l); "
    "s_l = mean(<J_l @ h_l, h_final>) / mean(||J_l @ h_l||^2), "
    "s_l = 1.0 when mean(||J_l @ h_l||^2) < 1e-12"
)


@dataclass
class _LayerMoments:
    """Per-layer moment accumulators: device sums while sampling, fp64 CPU at the end."""

    d_model: int
    h_sum: torch.Tensor  # [d_model] float64 CPU sum of h_l over the mask positions
    h_norm_sum: float  # sum of per-position ||h_l||
    count: int  # positions accumulated
    t_sum: torch.Tensor | None  # [d_model] float64 CPU sum of J_l @ h_l (J layers only)
    dot_sum: float | None  # sum of <t_p, h_final_p> over positions (J layers only)
    sq_sum: float | None  # sum of ||t_p||^2 over positions (J layers only)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lens-dir", required=True, help="the J source (lens directory or single lens-<mask>.pt)")
    parser.add_argument("--mask", default="text", help="which lens file (and which positions) to census")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True, help="the bias-<mask>.pt path to write")
    parser.add_argument("--json", required=True, help="summary path (the table numbers, tensors excluded)")
    parser.add_argument(
        "--backend", choices=("hf-llava", "tiny"), default="hf-llava",
        help="model backend: 'hf-llava' (default, CUDA) or the tiny CPU smoke fixture",
    )
    parser.add_argument("--dtype", default="float32", help="torch dtype name for the model weights")
    parser.add_argument("--skip-first", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--limit", type=int, default=None, help="census only the first N samples (smoke)")
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> LlavaLensModel:
    """The census model: the HF checkpoint (default, CUDA) or the tiny CPU fixture."""
    if args.backend == "tiny":
        from vlm_lens.models.tiny_llava import TinyLlavaConfig, build_tiny_llava

        hf_model, processor = build_tiny_llava(TinyLlavaConfig())
        return LlavaLensModel(hf_model, processor)
    return LlavaLensModel.from_pretrained(
        dtype=getattr(torch, args.dtype), device="cuda", local_files_only=True
    )


def main() -> int:
    args = parse_args()
    samples = read_manifest(args.manifest)
    if args.limit:
        samples = samples[: args.limit]
    lenses, _ = load_lens_set(args.lens_dir)
    if args.mask not in lenses:
        raise ValueError(f"mask {args.mask!r} not in {args.lens_dir} (found {sorted(lenses)})")
    lens = lenses[args.mask]
    j_layers = set(lens.source_layers)
    model = load_model(args)
    print(
        f"model ready: layers={model.n_layers} d_model={model.d_model} "
        f"image_seq_length={model.image_seq_length}; fitted J at {sorted(j_layers)}"
    )

    layers = list(range(model.n_layers))
    final_layer = model.n_layers - 1
    moments = {
        layer: _LayerMoments(
            d_model=model.d_model,
            h_sum=torch.zeros(model.d_model, dtype=torch.float64),
            h_norm_sum=0.0,
            count=0,
            t_sum=torch.zeros(model.d_model, dtype=torch.float64) if layer in j_layers else None,
            dot_sum=0.0 if layer in j_layers else None,
            sq_sum=0.0 if layer in j_layers else None,
        )
        for layer in layers
    }
    hf_sum = torch.zeros(model.d_model, dtype=torch.float64)
    hf_count = 0
    n_used = 0
    n_skipped = 0

    with torch.no_grad():
        for index, sample in enumerate(samples):
            batch = as_batch(model, sample, args.max_seq_len)
            masks = build_position_masks(
                batch.input_ids, model.image_token_id, skip_first=args.skip_first,
                masks=(args.mask,),
            )
            mask_positions = masks[args.mask]
            if not bool(mask_positions.any()):
                n_skipped += 1
                continue
            # The readout's capture pattern: block outputs at every layer, each block's
            # pre-norm residual; the final-layer entry is the pre-norm residual h_final.
            with ActivationRecorder(model.layers, at=layers) as recorder:
                model.forward_mm(batch)
                activations = {layer: recorder.activations[layer].detach() for layer in layers}
            final_act = activations[final_layer][0]  # [seq_len, d_model]
            hf = final_act[mask_positions.to(final_act.device)].float()
            hf_sum += hf.sum(dim=0).cpu().double()
            hf_count += int(hf.shape[0])
            for layer in layers:
                act = activations[layer][0]
                h = act[mask_positions.to(act.device)].float()
                moments[layer].h_sum += h.sum(dim=0).cpu().double()
                moments[layer].h_norm_sum += float(h.norm(dim=-1).sum())
                moments[layer].count += int(h.shape[0])
                if layer in j_layers:
                    t = lens.transport(h, layer)
                    moments[layer].t_sum += t.sum(dim=0).cpu().double()
                    moments[layer].dot_sum += float((t * hf).sum())
                    moments[layer].sq_sum += float((t * t).sum())
            n_used += 1
            if (index + 1) % 10 == 0:
                print(f"  {index + 1}/{len(samples)} samples", flush=True)

    if n_used == 0:
        raise ValueError(f"no samples with valid {args.mask!r} mask positions in {args.manifest}")

    mean_final = hf_sum / hf_count
    bias_payload: dict[str, object] = {}
    layer_rows: dict[str, dict[str, float | int]] = {}
    for layer in layers:
        m = moments[layer]
        row: dict[str, float | int] = {
            "n_positions": m.count,
            "mean_h_norm": round(m.h_norm_sum / m.count, 6),
        }
        if layer in j_layers:
            j64 = lens.jacobians[layer].to(torch.float64)
            bias = mean_final - j64 @ (m.h_sum / m.count)
            mean_dot = m.dot_sum / m.count
            mean_sq = m.sq_sum / m.count
            scale = mean_dot / mean_sq if mean_sq >= 1e-12 else 1.0
            bias_payload[str(layer)] = {"bias": bias.to(torch.float32), "scale": float(scale)}
            row["bias_norm"] = round(float(bias.norm()), 6)
            row["scale"] = round(float(scale), 6)
        layer_rows[str(layer)] = row

    meta = {
        "lens_dir": str(args.lens_dir),
        "mask": args.mask,
        "n_samples_used": n_used,
        "skip_first": args.skip_first,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "estimator": ESTIMATOR,
    }
    save_bias(args.out, {**bias_payload, "meta": meta})

    print(f"\ncensus: {n_used} samples used, {n_skipped} skipped (no valid {args.mask!r} positions)")
    print("layer      ||b_l||        s_l   mean ||h_l||")
    for layer in layers:
        row = layer_rows[str(layer)]
        bias_norm = f"{row['bias_norm']:>10.3f}" if "bias_norm" in row else "          -"
        scale = f"{row['scale']:>9.3f}" if "scale" in row else "         -"
        print(f"{layer:<6}  {bias_norm}  {scale}  {row['mean_h_norm']:>12.3f}")

    summary = {
        "lens_dir": str(args.lens_dir),
        "mask": args.mask,
        "n_samples_used": n_used,
        "n_samples_skipped": n_skipped,
        "skip_first": args.skip_first,
        "estimator": ESTIMATOR,
        "created_utc": meta["created_utc"],
        "layers": layer_rows,
    }
    Path(args.json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
