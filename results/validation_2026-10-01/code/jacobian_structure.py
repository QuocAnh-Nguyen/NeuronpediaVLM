#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Structural dissection of a fitted J-lens: scaled-identity / diagonal / low-rank share per layer.

The next campaign question is which synthetic readout variants are worth scoring. Arbitrary
[d, d] matrices can be injected as lenses with NO new fitting (synthetic lens artifacts), so
replacing parts of the fitted average Jacobian ``J_l`` with cheaper structure - a scaled
identity, a per-column rescale, a rank-k truncation - is a free experiment, but only where
``J_l`` is actually close to that structure. This script dissects each fitted ``J_l`` on CPU
(no model, no prompts, no GPU):

* ``alpha_ls = trace(J_l)/d`` is the least-squares scaled-identity coefficient, and
  ``residual_ratio = ||J_l - alpha_ls*I||_F / ||J_l||_F`` is what a scaled-identity synthetic
  readout would miss (0 = ``J_l`` IS a scaled identity);
* ``diag_ratio = sum|diag| / sum|offdiag|`` sizes the per-column-rescale variant, and
  ``symmetry_ratio = ||J_l - J_l^T||_F / ||J_l||_F`` flags asymmetric (directed) maps;
* a truncated SVD gives the top singular values (``s1``/``s8``/``s_k``), the
  ``energy_top_k = sum(s^2, top-k) / ||J_l||_F^2`` that a rank-k truncation keeps (the
  denominator is the full spectrum in exact arithmetic, so no full SVD is needed), and the
  participation ratio ``PR = (sum s, top-k)^2 / sum(s^2, top-k)`` of the returned set - the
  low-rank variant is worth scoring where ``energy_top_k`` is high at small ``k``.

The truncated SVD is the slow part: ``torch.svd_lowrank`` (torch has no ``svds``) with
``niter=16`` power iterations costs ~1 s per layer at d=4096 on the campaign workstation
(measured, flat and decayed spectra alike), so all 31 layers dissect in well under a minute -
far inside the ~30-min budget. Cheap scalar reductions accumulate in float64; the d<k guard
is ``k = min(svd_k, d)``; a dead (all-zero) ``J_l`` reports ``None`` ratios/svals rather than
0/0. One layer is held at a time (popped from the loaded lens set as processed) to keep the
CPU footprint flat.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402, I001  # the vlm_lens import must precede jlens: it installs the path
import vlm_lens  # noqa: E402, F401
from vlm_lens.artifacts import load_lens_set  # noqa: E402

#: Power iterations for the randomized SVD; 16 gives <0.2% s1 error on flat spectra at
#: ~1 s/layer (d=4096, measured) - accurate enough for the energy/PR decision, still cheap.
SVD_NITER = 16

ESTIMATOR = (
    "alpha_ls = trace(J_l)/d; "
    "residual_ratio = ||J_l - alpha_ls*I||_F / ||J_l||_F; "
    "diag_ratio = sum|diag(J_l)| / sum|offdiag(J_l)| (None when J_l is pure diagonal); "
    "symmetry_ratio = ||J_l - J_l^T||_F / ||J_l||_F; "
    "svd = torch.svd_lowrank(J_l, q=min(svd_k, d), niter=16) (torch has no torch.svds); "
    "energy_top_k = sum(s^2, top-k) / ||J_l||_F^2; "
    "participation_ratio = (sum s, top-k)^2 / sum(s^2, top-k) (k-truncated estimate); "
    "col/row_norm = mean/std (population) of the d column/row L2 norms"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lens-dir", required=True, help="the J source (lens directory or single lens-<mask>.pt)")
    parser.add_argument("--mask", default="text", help="which lens file to dissect")
    parser.add_argument("--json", required=True, help="summary path (the table numbers, tensors excluded)")
    parser.add_argument("--digest", required=True, help="digest path (the per-layer table; also printed)")
    parser.add_argument("--layers", default=None, help="comma-separated layers to dissect, e.g. 0,8,16,20,24,30 (default: all fitted)")
    parser.add_argument("--svd-k", type=int, default=64, help="singular values requested (d<k guard: k = min(svd_k, d))")
    parser.add_argument("--svd-layers", default=None, help="comma-separated layers to run the SVD on (default = --layers)")
    parser.add_argument("--device", default="cpu", help="torch device (CPU by default; this script needs no GPU)")
    return parser.parse_args()


def parse_layer_list(spec: str | None, flag: str) -> list[int] | None:
    """Parse a comma-separated layer list ("0,8,16"); ``None`` means "use the default"."""
    if spec is None:
        return None
    if not spec.strip():
        raise SystemExit(f"{flag} must be comma-separated ints, got {spec!r}")
    try:
        layers = sorted({int(part) for part in spec.split(",") if part.strip()})
    except ValueError:
        raise SystemExit(f"{flag} must be comma-separated ints, got {spec!r}") from None
    return layers


def dissect_layer(J: torch.Tensor, *, svd: bool, k: int) -> dict[str, float | int | None]:
    """Structural metrics for one ``J_l`` = [d, d].

    Cheap scalar reductions run in float64 (``J64``); the truncated SVD runs on the fp32
    ``J`` (``svd_lowrank`` requires it). ``k`` is already the d<k-guarded rank. A dead
    (all-zero) ``J_l`` reports ``None`` ratios/svals rather than 0/0.
    """
    if J.dim() != 2 or J.shape[0] != J.shape[1]:
        raise ValueError(f"J_l must be square [d, d], got {tuple(J.shape)}")
    d = int(J.shape[0])
    J64 = J.to(torch.float64)
    fro = float(torch.linalg.matrix_norm(J64))
    trace = float(J64.trace())
    alpha_ls = trace / d
    eye = torch.eye(d, dtype=J64.dtype, device=J64.device)
    residual_ratio = float((J64 - alpha_ls * eye).norm()) / fro if fro > 0.0 else None
    diag_abs = float(J64.diagonal().abs().sum())
    offdiag_abs = float(J64.abs().sum()) - diag_abs
    diag_ratio = diag_abs / offdiag_abs if offdiag_abs > 0.0 else None
    symmetry_ratio = float((J64 - J64.transpose(-2, -1)).norm()) / fro if fro > 0.0 else None

    s1 = s8 = s_last = energy = pr = None
    if svd and fro > 0.0:
        kk = min(k, d)
        _, S, _ = torch.svd_lowrank(J, q=kk, niter=SVD_NITER)
        s64 = S.to(torch.float64)
        s1 = float(S[0])
        s8 = float(S[7]) if kk >= 8 else None
        s_last = float(S[-1])
        s_sq = float((s64**2).sum())
        energy = s_sq / (fro * fro)
        pr = float(s64.sum()) ** 2 / s_sq

    col_norms = J64.norm(dim=0)
    row_norms = J64.norm(dim=1)
    return {
        "fro_norm": round(fro, 6),
        "trace": round(trace, 6),
        "alpha_ls": round(alpha_ls, 6),
        "residual_ratio": None if residual_ratio is None else round(residual_ratio, 6),
        "diag_ratio": None if diag_ratio is None else round(diag_ratio, 6),
        "symmetry_ratio": None if symmetry_ratio is None else round(symmetry_ratio, 6),
        "s1": None if s1 is None else round(s1, 6),
        "s8": None if s8 is None else round(s8, 6),
        "s_k": None if s_last is None else round(s_last, 6),
        "energy_top_k": None if energy is None else round(energy, 6),
        "participation_ratio": None if pr is None else round(pr, 6),
        "col_norm_mean": round(float(col_norms.mean()), 6),
        "col_norm_std": round(float(col_norms.std(correction=0)), 6),
        "row_norm_mean": round(float(row_norms.mean()), 6),
        "row_norm_std": round(float(row_norms.std(correction=0)), 6),
    }


def _cell(value: float | int | None, spec: str, width: int) -> str:
    """Fixed-width digest cell; an undefined metric (``None``) renders as '-'."""
    return f"{'-' if value is None else format(value, spec):>{width}}"


def digest_table(layer_rows: dict[str, dict[str, float | int | None]], layers: list[int]) -> str:
    """Compact fixed-width per-layer table (printed to stdout and the --digest file)."""
    header = (
        f"{'layer':>5}  {'||J||_F':>10}  {'trace':>10}  {'alpha_ls':>10}  {'res_r':>7}  "
        f"{'diag_r':>7}  {'sym_r':>7}  {'s1':>9}  {'s8':>9}  {'s_k':>9}  {'en_k':>7}  "
        f"{'PR':>9}  {'col_n mean±std':>17}  {'row_n mean±std':>17}"
    )
    lines = [header]
    for layer in layers:
        row = layer_rows[str(layer)]
        col = f"{row['col_norm_mean']:.3f}±{row['col_norm_std']:.3f}"
        rown = f"{row['row_norm_mean']:.3f}±{row['row_norm_std']:.3f}"
        lines.append(
            f"{layer:>5}  {_cell(row['fro_norm'], '.3f', 10)}  {_cell(row['trace'], '.3f', 10)}  "
            f"{_cell(row['alpha_ls'], '.6f', 10)}  {_cell(row['residual_ratio'], '.4f', 7)}  "
            f"{_cell(row['diag_ratio'], '.4f', 7)}  {_cell(row['symmetry_ratio'], '.4f', 7)}  "
            f"{_cell(row['s1'], '.3f', 9)}  {_cell(row['s8'], '.3f', 9)}  {_cell(row['s_k'], '.3f', 9)}  "
            f"{_cell(row['energy_top_k'], '.4f', 7)}  {_cell(row['participation_ratio'], '.1f', 9)}  "
            f"{col:>17}  {rown:>17}"
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.svd_k < 1:
        raise SystemExit("--svd-k must be >= 1")
    device = torch.device(args.device)
    lenses, _ = load_lens_set(args.lens_dir)
    if args.mask not in lenses:
        raise ValueError(f"mask {args.mask!r} not in {args.lens_dir} (found {sorted(lenses)})")
    lens = lenses[args.mask]
    source_layers = list(lens.source_layers)
    d_model = lens.d_model

    requested = parse_layer_list(args.layers, "--layers")
    if requested is None:
        requested = source_layers
    fitted = set(source_layers)
    missing = [layer for layer in requested if layer not in fitted]
    selected = [layer for layer in requested if layer in fitted]
    if not selected:
        raise SystemExit(f"no requested layer is fitted (fitted {source_layers}; requested {requested})")

    svd_requested = parse_layer_list(args.svd_layers, "--svd-layers")
    if svd_requested is None:
        svd_requested = selected
    svd_set = sorted(set(svd_requested) & set(selected))
    if set(svd_requested) - set(selected):
        skipped = sorted(set(svd_requested) - set(selected))
        print(f"note: --svd-layers outside the dissected set are ignored: {skipped}")

    k = min(args.svd_k, d_model)
    print(
        f"dissection: mask={args.mask} d_model={d_model} n_prompts={lens.n_prompts} "
        f"device={device}; fitted J at {source_layers}"
    )
    print(f"dissecting {selected}; svd k={k} at {svd_set} (niter={SVD_NITER})")
    if missing:
        print(f"note: requested layers not fitted, skipped: {missing}")

    layer_rows: dict[str, dict[str, float | int | None]] = {}
    with torch.no_grad():
        for layer in selected:
            # One layer at a time: pop it from the loaded set so the CPU footprint stays
            # flat; fp32 cast keeps svd_lowrank happy even for an fp16-saved lens file.
            J = lens.jacobians.pop(layer).to(device).float()
            layer_rows[str(layer)] = dissect_layer(J, svd=layer in set(svd_set), k=k)
            del J

    text = digest_table(layer_rows, selected)
    print()
    print(text)
    print(f"\ndissected {len(selected)} layers; svd on {len(svd_set)} layers")

    meta = {
        "lens_dir": str(args.lens_dir),
        "n_prompts": lens.n_prompts,
        "svd_k": args.svd_k,
        "svd_k_effective": k,
        "svd_niter": SVD_NITER,
        "device": str(device),
        "layers_dissected": selected,
        "layers_skipped": missing,
        "svd_layers": svd_set,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "estimator": ESTIMATOR,
    }
    summary = {
        "mask": args.mask,
        "source_layers": source_layers,
        "d_model": d_model,
        "layers": layer_rows,
        "meta": meta,
    }
    Path(args.digest).write_text(text + "\n", encoding="utf-8")
    Path(args.json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {args.digest}")
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
