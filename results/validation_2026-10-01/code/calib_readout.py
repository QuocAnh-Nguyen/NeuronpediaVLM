#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Tuned-lens-style calibration of the J-lens readout: per-layer temperature + scale.

The census readout ``unembed(s_l * (J_l @ h + b_l))`` fixed the mid-layer degradation
(D37), but its two per-layer knobs were fitted in residual space or left at a default:
the least-squares scale ``s_l`` never saw the output distribution, and the softmax
temperature is 1.0 by fiat. This script estimates both post-hoc, tuned-lens-style: per
layer, minimize

    KL(softmax(z) || softmax(z_model)),   z = unembed((s_l * m) * (J_l @ h + b_l)) / T,

over a grid of scale multipliers ``m`` and output temperatures ``T`` on a calibration
manifest (a FIT split; ``write_calibrated_bias.py`` bakes the winners into the
``bias-<mask>.pt`` the standard scorer consumes). The target is the model's own
final-layer logits ``z_model = unembed(h_final)`` - never corrected. Positions are
selected exactly as the text-tag scoring in ``s2_eval.score_table_fast``: mask nonzero,
a next token exists, and (the default mode, not ``include_placeholders``) the next token
is not the image placeholder. Layers missing from the bias file fall back to the
scorer's semantics (``s_l = 1.0``, ``b_l = 0``) instead of crashing.

Forward-only (no backprop), so it runs anywhere the forward pass runs; the tiny backend
exercises the whole path on CPU. The KL is computed and accumulated in float64 (the grid
compares coarse cells, so reduction noise must stay invisible), vectorized over position
chunks; the transport is one batched matmul per (sample, layer). Outputs: the per-layer
winners + full grid to ``--out``, and the mean logit gap ``mean_p(z_base - z_model)``
per vocab entry - ``z_base`` is the uncalibrated census readout - as
``<out stem>_logitgap.pt``.

Redundancy note (R2). With no logit bias the grid parameterizes ONE effective scalar per
layer, ``c = s_l * m / T`` (``s_l`` the loaded census scale): the unembed's final norm is
scale-invariant up to its eps (RMSNorm), so the residual-side multiplier ``s_l * m`` and
the logit-side division by ``T`` do not separate - the reported ``best_temp`` alone is
NOT a temperature, it is ``1/(c/s_l)`` in readout units. The per-layer ``c_effective``
and ``c_grid`` columns make the collapse explicit; the KL grid should be read as nearly
flat in ``m`` (noise-level differences across the scale multipliers).
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
from vlm_lens.artifacts import load_bias, load_lens_set  # noqa: E402
from vlm_lens.data.manifest import read_manifest  # noqa: E402
from vlm_lens.models.llava import LlavaLensModel  # noqa: E402
from vlm_lens.positions import build_position_masks  # noqa: E402

CALIB_ESTIMATOR = (
    "per (layer, cell): z(m, T) = unembed((s_l * m) * (J_l @ h_l + b_l)) / T; "
    "KL(m, T) = mean_p KL(softmax(z) || softmax(unembed(h_final))) in float64; "
    "best cell = grid argmin; z_base = unembed(s_l * (J_l @ h_l + b_l)) is the "
    "uncalibrated census readout (grid cell m=1.0, T=1.0, so kl_best <= kl_base); "
    "c = s_l * m / T is the one effective scalar per layer (the (m, T) split is a "
    "redundant reparameterization of it)"
)


@dataclass
class _LayerCalib:
    """Per-layer calibration accumulators: device sums while sampling, reduced at the end."""

    kl_base_sum: float  # sum of KL(softmax(z_base) || softmax(z_model)) over positions
    gap_sum: torch.Tensor  # [vocab] float64 sum of (z_base - z_model) over positions
    grid_kl: dict[tuple[float, float], float]  # (scale_mult, temp) -> summed KL
    count: int  # positions scored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lens-dir", required=True, help="the J source (lens directory or single lens-<mask>.pt)"
    )
    parser.add_argument(
        "--bias-dir", required=True,
        help="directory of moment-census bias-<mask>.pt affine-correction files",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mask", default="text", help="which lens file (and which positions) to calibrate")
    parser.add_argument("--limit", type=int, default=40, help="calibrate only the first N samples")
    parser.add_argument(
        "--scale-mults", default="0.5,0.75,1.0,1.25,1.5",
        help="comma-separated scale multipliers m applied to s_l (the refined scale grid)",
    )
    parser.add_argument(
        "--temps", default="0.5,0.7,1.0,1.4,2.0,3.0",
        help="comma-separated output temperatures T (must all be > 0)",
    )
    parser.add_argument("--out", default="calib.json", help="summary path (the grid numbers, tensors excluded)")
    parser.add_argument(
        "--backend", choices=("hf-llava", "tiny"), default="hf-llava",
        help="model backend: 'hf-llava' (default, CUDA) or the tiny CPU smoke fixture",
    )
    parser.add_argument("--dtype", default="bfloat16", help="torch dtype name for the model weights")
    parser.add_argument("--skip-first", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=1536)
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> LlavaLensModel:
    """The calibration model: the HF checkpoint (default, CUDA) or the tiny CPU fixture."""
    if args.backend == "tiny":
        from vlm_lens.models.tiny_llava import TinyLlavaConfig, build_tiny_llava

        hf_model, processor = build_tiny_llava(TinyLlavaConfig())
        return LlavaLensModel(hf_model, processor)
    return LlavaLensModel.from_pretrained(
        dtype=getattr(torch, args.dtype), device="cuda", local_files_only=True
    )


def parse_floats(spec: str, flag: str) -> tuple[float, ...]:
    """Split a comma-separated float grid; empty or non-positive temperatures are CLI errors."""
    values = tuple(float(part) for part in spec.split(",") if part.strip())
    if not values:
        raise SystemExit(f"{flag} must be a comma-separated float list, got {spec!r}")
    if flag == "--temps" and any(value <= 0 for value in values):
        raise SystemExit(f"{flag} must all be > 0 (a non-positive temperature is degenerate), got {spec!r}")
    return values


def _kl_sum(lens_logits: torch.Tensor, model_logits: torch.Tensor, chunk_size: int) -> float:
    """KL(softmax(lens) || softmax(model)) summed over positions, in float64.

    ``KL(P || Q) = sum_v p_v * (log p_v - log q_v)`` with ``P`` the lens softmax and
    ``Q`` the model softmax. Vectorized over position chunks of ``chunk_size`` rows;
    each chunk's logits are promoted to float64 before the log-softmaxes and the chunk
    sums accumulate in float64, so the reduction noise cannot reach the grid comparison.
    """
    total = 0.0
    for start in range(0, lens_logits.shape[0], chunk_size):
        lens_chunk = lens_logits[start : start + chunk_size].to(torch.float64)
        model_chunk = model_logits[start : start + chunk_size].to(torch.float64)
        log_p = torch.log_softmax(lens_chunk, dim=-1)
        log_q = torch.log_softmax(model_chunk, dim=-1)
        total += float((log_p.exp() * (log_p - log_q)).sum())
    return total


def _base_sums(
    z_base: torch.Tensor, z_model: torch.Tensor, chunk_size: int
) -> tuple[float, torch.Tensor]:
    """KL(z_base || z_model) plus the per-vocab logit-gap sums over positions.

    One chunk loop computes both reductions (the [chunk, vocab] float64 promotion is the
    only large transient); the gap sums are a [vocab] float64 tensor on z_base's device,
    the running total of ``(z_base - z_model)`` over positions.
    """
    gap = torch.zeros(z_base.shape[1], dtype=torch.float64, device=z_base.device)
    kl = 0.0
    for start in range(0, z_base.shape[0], chunk_size):
        base_chunk = z_base[start : start + chunk_size].to(torch.float64)
        model_chunk = z_model[start : start + chunk_size].to(torch.float64)
        log_p = torch.log_softmax(base_chunk, dim=-1)
        log_q = torch.log_softmax(model_chunk, dim=-1)
        kl += float((log_p.exp() * (log_p - log_q)).sum())
        gap += (base_chunk - model_chunk).sum(dim=0)
    return kl, gap


def main() -> int:
    args = parse_args()
    scale_mults = parse_floats(args.scale_mults, "--scale-mults")
    temps = parse_floats(args.temps, "--temps")
    samples = read_manifest(args.manifest)
    if args.limit:
        samples = samples[: args.limit]
    lenses, _ = load_lens_set(args.lens_dir)
    if args.mask not in lenses:
        raise ValueError(f"mask {args.mask!r} not in {args.lens_dir} (found {sorted(lenses)})")
    lens = lenses[args.mask]
    bias_path = Path(args.bias_dir) / f"bias-{args.mask}.pt"
    if not bias_path.is_file():
        raise FileNotFoundError(
            f"no bias-{args.mask}.pt under {args.bias_dir}; the calibration needs the census bias"
        )
    bias_payload, _ = load_bias(bias_path)
    # s2_eval's extraction; layers missing from the bias fall back to the scorer's
    # no-correction semantics (s_l = 1.0, b_l = 0) instead of crashing.
    bias_by_layer = {int(layer): entry["bias"] for layer, entry in bias_payload.items()}
    scale_by_layer = {int(layer): float(entry["scale"]) for layer, entry in bias_payload.items()}
    model = load_model(args)
    print(
        f"model ready: layers={model.n_layers} d_model={model.d_model} "
        f"image_seq_length={model.image_seq_length}; fitted J at {sorted(lens.source_layers)}"
    )

    final_layer = model.n_layers - 1
    # s2_eval scores the final-layer row as the model's own logits (never corrected), so
    # the calibration grid covers the lens layers except a final layer.
    calib_layers = [layer for layer in sorted(set(lens.source_layers)) if layer != final_layer]
    if not calib_layers:
        raise ValueError("no lens layers below the final layer to calibrate")
    # The readout's capture pattern: block outputs at the lens layers plus the final
    # layer, each block's pre-norm residual.
    record_at = sorted(set(calib_layers) | {final_layer})
    unembed_device = model.unembed_weight().device  # z lives here: unembed forces this device
    vocab_size = int(model.unembed_weight().shape[0])
    accum = {
        layer: _LayerCalib(
            kl_base_sum=0.0,
            gap_sum=torch.zeros(vocab_size, dtype=torch.float64, device=unembed_device),
            grid_kl={(sm, temp): 0.0 for sm in scale_mults for temp in temps},
            count=0,
        )
        for layer in calib_layers
    }
    n_used = 0
    n_skipped = 0
    n_positions = 0
    chunk_size = 256

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
            # s2_eval's text-tag scoring positions: mask nonzero, a next token exists,
            # and the next token is not the image placeholder (the V5 default mode).
            input_ids = batch.input_ids[0].detach().cpu()
            seq_len = int(input_ids.numel())
            positions = [
                int(p) for p in mask_positions.nonzero(as_tuple=True)[0] if int(p) + 1 < seq_len
            ]
            positions = [p for p in positions if int(input_ids[p + 1]) != model.image_token_id]
            if not positions:
                n_skipped += 1
                continue
            pos_idx = torch.tensor(positions, dtype=torch.long)
            with ActivationRecorder(model.layers, at=record_at) as recorder:
                model.forward_mm(batch)
                activations = {layer: recorder.activations[layer].detach() for layer in record_at}
            final_act = activations[final_layer][0]  # [seq_len, d_model]
            hf = final_act[pos_idx.to(final_act.device)].float()
            z_model = model.unembed(hf).float()  # [n_pos, vocab]; the model's own logits
            n_pos = len(positions)
            for layer in calib_layers:
                act = activations[layer][0]
                h = act[pos_idx.to(act.device)].float()
                t = lens.transport(h, layer)  # one batched matmul per (sample, layer)
                # readout.py's composition, exactly: transport, += bias, *= scale.
                entry_bias = bias_by_layer.get(layer)
                x = t
                if entry_bias is not None:
                    x = t + entry_bias.to(t.device)
                s = scale_by_layer.get(layer, 1.0)
                z_base = model.unembed(x * s).float()
                kl_base, gap = _base_sums(z_base, z_model, chunk_size)
                accum[layer].kl_base_sum += kl_base
                accum[layer].gap_sum += gap
                for sm in scale_mults:
                    z_raw = model.unembed(x * (s * sm)).float()
                    for temp in temps:
                        accum[layer].grid_kl[(sm, temp)] += _kl_sum(
                            z_raw / temp, z_model, chunk_size
                        )
                accum[layer].count += n_pos
            n_used += 1
            n_positions += n_pos
            if (index + 1) % 10 == 0:
                print(f"  {index + 1}/{len(samples)} samples", flush=True)

    if n_used == 0:
        raise ValueError(
            f"no samples with valid {args.mask!r} scoring positions in {args.manifest}"
        )

    # The mean logit gap per layer: mean over positions of (z_base - z_model), per vocab
    # entry, as float32 - the direction the unembedding over- and under-weights.
    gap_path = Path(args.out).with_name(f"{Path(args.out).stem}_logitgap.pt")
    gap_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            str(layer): (accum[layer].gap_sum / accum[layer].count).to(torch.float32).cpu()
            for layer in calib_layers
        },
        gap_path,
    )

    layer_rows: dict[str, dict[str, object]] = {}
    for layer in calib_layers:
        a = accum[layer]
        # Ties keep the first grid cell in (scale_mult, temp) iteration order.
        best_sm, best_temp = min(
            ((sm, temp) for sm in scale_mults for temp in temps), key=lambda cell: a.grid_kl[cell]
        )
        s_loaded = scale_by_layer.get(layer, 1.0)  # the census s_l the grid refines
        layer_rows[str(layer)] = {
            "kl_base": round(a.kl_base_sum / a.count, 6),
            "best_scale_mult": best_sm,
            "best_temp": best_temp,
            "c_effective": round(s_loaded * best_sm / best_temp, 6),
            "kl_best": round(a.grid_kl[(best_sm, best_temp)] / a.count, 6),
            "grid": {
                str(sm): {str(temp): round(a.grid_kl[(sm, temp)] / a.count, 6) for temp in temps}
                for sm in scale_mults
            },
            "c_grid": {
                str(sm): {str(temp): round(s_loaded * sm / temp, 6) for temp in temps}
                for sm in scale_mults
            },
        }

    meta = {
        "lens_dir": str(args.lens_dir),
        "bias_dir": str(args.bias_dir),
        "manifest": args.manifest,
        "mask": args.mask,
        "n_samples_used": n_used,
        "n_samples_skipped": n_skipped,
        "n_positions": n_positions,
        "skip_first": args.skip_first,
        "scale_mults": list(scale_mults),
        "temps": list(temps),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "estimator": CALIB_ESTIMATOR,
    }
    report = {
        "mask": args.mask,
        "n_samples": n_used,
        "n_positions": n_positions,
        "layers": layer_rows,
        "meta": meta,
    }

    print(
        f"\ncalibration: {n_used} samples used, {n_skipped} skipped "
        f"(no valid {args.mask!r} scoring positions)"
    )
    print("layer      kl_base   best_sm   best_T      kl_best")
    for layer in calib_layers:
        row = layer_rows[str(layer)]
        print(
            f"{layer:<6}  {row['kl_base']:>10.3f}  {row['best_scale_mult']:>7.2f}  "
            f"{row['best_temp']:>7.2f}  {row['kl_best']:>10.3f}"
        )

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {gap_path}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
