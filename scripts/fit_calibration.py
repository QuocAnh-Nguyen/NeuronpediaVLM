#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Fit the per-layer logit-space calibration (bias + scale) of a fitted J lens.

The tuned-lens completion: for every fitted source layer, fit
``softmax(scale * x + bias)`` against the model's own teacher-forced next-token
distribution at the mask's source positions (``x`` = the lens logits exactly as
``readout.lens_readout`` produces them), with a sample-level hold-out and early
stopping. Writes ``bias-<mask>.pt`` (the additive ``logit_bias``/``logit_scale``
keys on the standard affine-correction schema, provenance incl. the base-lens
fingerprint in ``meta``) and prints the per-layer pre->post gain table.

Example::

    python scripts/fit_calibration.py --model llava-7b --backend hf-llava \
        --lens-dir fits/llava-1.5-7b-hf/artifacts --mask text \
        --manifest manifest-coco.jsonl --out-dir fits/llava-1.5-7b-hf/calibration
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vlm_lens.artifacts import load_lens_set, save_bias  # noqa: E402
from vlm_lens.calibrate import CalibrationResult, bias_payload, fit_calibration  # noqa: E402
from vlm_lens.data.manifest import manifest_meta, read_manifest, sha256_file  # noqa: E402
from vlm_lens.models.registry import MODEL_SPECS  # noqa: E402

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lens-dir", required=True, help="fitted lens directory (load_lens_set)")
    parser.add_argument("--mask", default="text", help="which fitted lens and position mask")
    parser.add_argument("--manifest", required=True, help="calibration manifest (read_manifest)")
    parser.add_argument("--out-dir", required=True, help="where bias-<mask>.pt is written")
    parser.add_argument(
        "--layers",
        default="all",
        help="'all' (every fitted source layer) or comma-separated layer indices",
    )
    parser.add_argument("--holdout-frac", type=float, default=0.2, help="sample-level hold-out")
    parser.add_argument("--device", default=None, help="default: cuda when available else cpu")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default=None, help="model loading dtype")
    parser.add_argument("--epochs", type=int, default=8, help="Adam epochs (early-stopped)")
    parser.add_argument("--lr", type=float, default=0.05, help="Adam learning rate")
    parser.add_argument("--batch-positions", type=int, default=256, help="position mini-batch size")
    parser.add_argument("--patience", type=int, default=2, help="early-stop patience (0 = off)")
    parser.add_argument("--skip-first", type=int, default=1, help="leading positions excluded")
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N samples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--notes", default=None, help="free-text note stored in the bias meta")

    model = parser.add_argument_group("model")
    model.add_argument("--backend", choices=("hf-llava", "tiny"), default="hf-llava")
    model.add_argument(
        "--model",
        choices=tuple(MODEL_SPECS),
        default="llava-7b",
        help="checkpoint family: llava-7b (default), llava-13b or qwen2vl",
    )
    model.add_argument("--attn-implementation", default=None)
    model.add_argument("--compile", action="store_true", help="torch.compile each residual block")
    model.add_argument("--local-files-only", action="store_true")
    model.add_argument("--tiny-image-size", type=int, default=336, help="tiny backend only")
    return parser.parse_args(argv)


def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device:
        return torch.device(args.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(args: argparse.Namespace, device: torch.device) -> torch.dtype:
    if args.dtype:
        return DTYPES[args.dtype]
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def build_model(args: argparse.Namespace) -> object:
    """Wrapper for the ``--model`` family: tiny CPU fixture or the real checkpoint."""
    spec = MODEL_SPECS[args.model]
    if args.backend == "tiny":
        hf_model, processor = spec.build_tiny(image_size=args.tiny_image_size)
        return spec.lens_model_cls()(hf_model, processor)
    device = resolve_device(args)
    dtype = resolve_dtype(args, device)
    print(f"loading {spec.hf_id} (device={device}, dtype={dtype})")
    return spec.lens_model_cls().from_pretrained(
        spec.hf_id,
        dtype=dtype,
        device=device,
        attn_implementation=args.attn_implementation,
        compile=args.compile,
        local_files_only=args.local_files_only,
    )


def parse_layers(resolved: str) -> list[int] | None:
    """``'all'`` -> every fitted source layer, else the comma-separated subset."""
    if resolved.strip().lower() == "all":
        return None
    try:
        return [int(part) for part in resolved.split(",") if part.strip()]
    except ValueError as error:
        raise SystemExit(f"cannot parse --layers {resolved!r}: {error}") from error


def format_gains(result: CalibrationResult, *, digits: int = 3) -> str:
    """Fixed-width pre->post gain table (``evaluate.format_scores`` style)."""
    header = (
        f"{'layer':>5}  {'scale':>7}  {'rank_true':>17}  {'top1_agree':>17}  "
        f"{'KL(model||lens)':>23}  {'CE':>17}"
    )
    lines = [header, "-" * len(header)]
    for layer, calibration in sorted(result.layers.items()):
        pre, post = calibration.pre, calibration.post

        def pair(key: str) -> str:
            if key not in pre or key not in post:
                return "--"
            return f"{pre[key]:>{digits + 4}.{digits}f} -> {post[key]:>{digits + 4}.{digits}f}"

        lines.append(
            f"{layer:>5}  {calibration.scale:>7.4f}  {pair('mean_rank_true'):>17}  "
            f"{pair('top1_agreement'):>17}  {pair('mean_kl'):>23}  {pair('holdout_ce'):>17}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    started = time.perf_counter()

    lenses, lens_provenance = load_lens_set(args.lens_dir)
    if args.mask not in lenses:
        raise SystemExit(
            f"mask {args.mask!r} not fitted in {args.lens_dir}; available: {sorted(lenses)}"
        )
    lens = lenses[args.mask]
    samples = read_manifest(args.manifest)
    if args.limit is not None:
        samples = samples[: args.limit]
    model = build_model(args)
    print(
        f"calibrating lens {args.lens_dir!r} (mask={args.mask}, "
        f"source_layers={lens.source_layers}) on {len(samples)} samples"
    )

    result = fit_calibration(
        model,
        lens,
        samples,
        mask=args.mask,
        layers=parse_layers(args.layers),
        holdout_frac=args.holdout_frac,
        epochs=args.epochs,
        lr=args.lr,
        batch_positions=args.batch_positions,
        patience=args.patience,
        skip_first=args.skip_first,
        max_seq_len=args.max_seq_len,
        device=args.device,
        seed=args.seed,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = bias_payload(result)
    payload["meta"] = {
        **payload["meta"],
        "backend": args.backend,
        "model_id": (
            MODEL_SPECS[args.model].tiny_id if args.backend == "tiny" else MODEL_SPECS[args.model].hf_id
        ),
        "lens_dir": str(args.lens_dir),
        "lens_provenance_created_utc": lens_provenance.get("created_utc"),
        "manifest_path": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_meta": manifest_meta(args.manifest),
        "wall_seconds": round(time.perf_counter() - started, 1),
        "notes": args.notes,
    }
    written = save_bias(out_dir / f"bias-{args.mask}.pt", payload)

    print(f"\nfitted {len(result.layers)} layer calibration(s) in {time.perf_counter() - started:.1f}s")
    print(format_gains(result))
    print(f"\nbias file: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
