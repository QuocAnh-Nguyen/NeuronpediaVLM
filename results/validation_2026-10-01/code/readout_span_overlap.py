#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Readout-span overlap: how much of each fitted J_l lives in W_U's dominant directions.

The x9 alpha sweep found concept-directed J-lens edits are rare (~1-3% of transported
vectors move at alpha<=4); the hypothesis is that a concept subspace is nearly orthogonal
to the dominant readout span of the unembedding (Yuan et al. 2026, arXiv 2609.39263:
across 26 models a random unit vector keeps only 0.38-0.80% of its energy in the top-10
readout directions). This script quantifies that on the fitted transport itself - zero
fit, no model, no prompts, CPU by default:

* the rows of W_U span the readout directions. A randomized SVD of the shape-oriented
  W_U ([vocab, d]) gives the top-``r`` d-space readout directions ``V_r = Vh[:r]`` and
  the projector ``P_span = V_r^T V_r``; ``readout_energy = mean_i (v_i^T P_span v_i)``
  over ``J_l``'s top ``svd_k`` right singular vectors ``v_i`` is the share of the
  transport's dominant directions the readout can even see (P_span is applied as
  ``||V_r v_i||^2``, never materialised). The isotropic random baseline is
  ``trace(P_span)/d`` - a readout-aligned transport scores ~1, a random direction the
  baseline;
* ``cross_layer_overlap = mean_i cos^2(v_i^(l), v_i^(L))`` over rank-matched pairs
  against the FINAL fitted layer's right singular vectors sizes the mid-layer basis
  drift the J-lens assumes away by transporting every layer into one target basis
  (1.0 at ``l = L``, near the baseline for unrelated bases).

Per layer the spectrum comes from ``torch.svd_lowrank`` (torch has no ``svds``) with
``niter=16`` power iterations and a fixed seed, like the other campaign SVDs. Cheap
scalar reductions accumulate in float64. Edge cases: the W_U orientation is detected by
shape (the dim equal to the lens d_model is the d side; a square tensor is read as
``[vocab, d]``, the repo's ``unembed_weight`` layout), ``k``/``r`` are guarded to
``min(requested, d)`` / ``min(requested, vocab, d)``, and a dead (all-zero) ``J_l``
reports ``None`` metrics. One layer is held at a time (popped from the loaded lens set
as processed) to keep the CPU footprint flat; the W_U SVD runs once.
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
#: ~1 s/layer (d=4096, measured) - accurate enough for the energy/overlap decisions.
SVD_NITER = 16
#: Seed for the randomized SVD starts, recorded in meta so runs reproduce.
SVD_SEED = 0
#: ``--backend tiny`` builds its W_U with the same config as tests/conftest.py's
#: ``tiny_model`` fixture (d_model=16, vocab=64), so a conftest-fitted tiny lens pairs
#: with it out of the box.
TINY_CONFIG_KWARGS = dict(
    d_model=16,
    n_layers=3,
    n_heads=2,
    vision_hidden=16,
    vision_layers=1,
    image_size=56,
    patch_size=14,
    vocab_size=64,
    image_token_id=50,
    seed=0,
)
#: Per-layer metric keys, in digest order; a dead (all-zero) ``J_l`` reports all-None.
ROW_KEYS = (
    "readout_energy",
    "cross_layer_overlap",
    "sv_s1",
    "sv_s8",
    "sv_sk",
    "participation_ratio",
)

ESTIMATOR = (
    "svd = torch.svd_lowrank(M, q=guarded, niter=16) fixed seed (torch has no torch.svds); "
    "W_U oriented to [vocab, d] by shape (d = lens d_model); readout directions = rows of "
    "Vh[:r], P_span = V_r^T V_r applied as ||V_r v_i||^2 (never materialised); "
    "readout_energy = mean_i (v_i^T P_span v_i) over J_l's top svd-k right vectors; "
    "cross_layer_overlap = mean_i cos^2(v_i^(l), v_i^(final fitted)), rank-matched pairs "
    "(exactly 1.0 for the final fitted layer); random baseline = trace(P_span)/d; "
    "participation_ratio = (sum s, top-k)^2 / sum(s^2, top-k) (k-truncated estimate)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lens-dir", required=True,
        help="the J source (lens directory or single lens-<mask>.pt)",
    )
    parser.add_argument("--mask", default="text", help="which lens file to dissect")
    parser.add_argument(
        "--unembed", default=None,
        help="saved W_U tensor .pt, [vocab, d] float32; orientation detected by shape "
        "([d, vocab] is transposed) (required unless --backend tiny)",
    )
    parser.add_argument(
        "--out", required=True,
        help="summary path (the per-layer numbers, tensors excluded)",
    )
    parser.add_argument(
        "--digest", required=True,
        help="digest path (the per-layer table; also printed)",
    )
    parser.add_argument(
        "--svd-k", type=int, default=64,
        help="singular values requested (d<k guard: k = min(svd_k, d))",
    )
    parser.add_argument(
        "--r-span", type=int, default=128,
        help="readout directions kept (guard: r = min(r_span, vocab, d))",
    )
    parser.add_argument(
        "--backend", choices=("unembed", "tiny"), default="unembed",
        help="W_U source: the --unembed tensor (default) or the tiny fixture built like "
        "tests/conftest.py's tiny model",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="torch device (CPU by default; this script needs no GPU)",
    )
    return parser.parse_args()


def orient_unembed(W: torch.Tensor, d_model: int, source: str) -> torch.Tensor:
    """``W_U`` as ``[vocab, d_model]`` with the rows the d-space readout directions.

    Orientation is detected by shape: the dim equal to the lens ``d_model`` is the d side
    and the other is the vocabulary, so a saved ``[d, vocab]`` tensor is transposed. A
    square tensor cannot be disambiguated by shape and is read as ``[vocab, d]`` (the
    repo's ``unembed_weight`` layout).
    """
    if W.dim() != 2:
        raise ValueError(f"W_U from {source} must be 2-D [vocab, d], got {tuple(W.shape)}")
    if W.shape[1] == d_model:
        return W
    if W.shape[0] == d_model:
        print(f"note: W_U from {source} saved as {tuple(W.shape)}; transposing to [vocab, d]")
        return W.transpose(0, 1).contiguous()
    raise ValueError(
        f"neither dim of W_U from {source} matches the lens d_model={d_model}: {tuple(W.shape)}"
    )


def load_unembed(args: argparse.Namespace, d_model: int) -> torch.Tensor:
    """The run's W_U source: ``[vocab, d_model]`` fp32 CPU, rows the d-space directions.

    ``--backend tiny`` grabs W_U from the tiny fixture (same config as tests/conftest.py's
    ``tiny_model``, so a conftest-fitted tiny lens pairs with it); otherwise the saved
    ``--unembed`` tensor is loaded with the ``weights_only`` unpickler.
    """
    if args.backend == "tiny":
        from vlm_lens.models.llava import LlavaLensModel
        from vlm_lens.models.tiny_llava import TinyLlavaConfig, build_tiny_llava

        hf_model, processor = build_tiny_llava(TinyLlavaConfig(**TINY_CONFIG_KWARGS))
        W = LlavaLensModel(hf_model, processor).unembed_weight()
        source = "tiny fixture"
    else:
        if not args.unembed:
            raise SystemExit("--unembed is required unless --backend tiny")
        source = str(args.unembed)
        W = torch.load(source, map_location="cpu", weights_only=True)
    W = orient_unembed(W, d_model, source).detach().to(torch.float32).cpu()
    print(f"W_U: {source} as {tuple(W.shape)} (vocab, d_model), rows are the d-space directions")
    return W


def spectrum(J: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(S, V)`` of the top-k singular pairs of ``J_l`` = [d, d]; ``None`` when dead.

    ``V`` is [d, k]: column ``i`` is the d-space right singular vector ``v_i``. The
    randomized SVD is seeded per call, so results do not depend on call order.
    """
    if float(torch.linalg.matrix_norm(J.to(torch.float64))) == 0.0:
        return None
    kk = min(k, int(J.shape[0]))
    torch.manual_seed(SVD_SEED)
    _, S, V = torch.svd_lowrank(J, q=kk, niter=SVD_NITER)
    return S, V


def overlap_row(
    S: torch.Tensor | None,
    V: torch.Tensor | None,
    *,
    V_r: torch.Tensor,
    V_ref: torch.Tensor | None,
    is_final: bool,
) -> dict[str, float | None]:
    """Per-layer metrics from the layer's ``(S, V)``; a ``None`` pair is a dead ``J_l``.

    ``readout_energy`` is the mean quadratic form ``v_i^T P_span v_i`` with
    ``P_span = V_r^T V_r`` applied as ``||V_r v_i||^2`` (column ``i`` of ``V_r @ V``);
    ``cross_layer_overlap`` is the mean rank-matched ``cos^2`` against the final fitted
    layer's vectors - exactly 1.0 for the final layer itself, ``None`` while no reference
    spectrum exists. Scalar reductions accumulate in float64.
    """
    if S is None or V is None:
        return dict.fromkeys(ROW_KEYS)
    s64 = S.to(torch.float64)
    kk = int(S.numel())
    sv_s1 = float(S[0])
    sv_s8 = float(S[7]) if kk >= 8 else None
    s_sq = float((s64**2).sum())
    pr = float(s64.sum()) ** 2 / s_sq

    # v^T P_span v = ||V_r v||^2: one [r, k] matmul; P_span ([d, d]) never materialised.
    proj = (V_r @ V).to(torch.float64)
    readout_energy = float((proj**2).sum(dim=0).mean())
    if is_final:
        cross: float | None = 1.0
    elif V_ref is None:
        cross = None
    else:
        # Rank-matched pairs: column i of V against column i of the reference. Singular
        # vectors are unit-norm; the explicit norms guard float error (cos^2 is sign-free).
        dots = (V_ref * V).sum(dim=0).to(torch.float64)
        norms = V_ref.norm(dim=0).to(torch.float64) * V.norm(dim=0).to(torch.float64)
        cross = float(((dots / norms) ** 2).mean())
    return {
        "readout_energy": round(readout_energy, 6),
        "cross_layer_overlap": None if cross is None else round(cross, 6),
        "sv_s1": round(sv_s1, 6),
        "sv_s8": None if sv_s8 is None else round(sv_s8, 6),
        "sv_sk": round(float(S[-1]), 6),
        "participation_ratio": round(pr, 6),
    }


def _cell(value: float | None, spec: str, width: int) -> str:
    """Fixed-width digest cell; an undefined metric (``None``) renders as '-'."""
    return f"{'-' if value is None else format(value, spec):>{width}}"


def digest_table(
    layer_rows: dict[str, dict[str, float | None]], layers: list[int], *, baseline: float
) -> str:
    """Compact fixed-width per-layer table (printed to stdout and the --digest file)."""
    lines = [f"# readout-span overlap; random baseline trace(P_span)/d = {baseline:.6f}"]
    lines.append(
        f"{'layer':>5}  {'en_ro':>8}  {'x_fin':>8}  {'sv_s1':>10}  {'sv_s8':>10}  "
        f"{'sv_sk':>10}  {'PR':>10}"
    )
    for layer in layers:
        row = layer_rows[str(layer)]
        lines.append(
            f"{layer:>5}  {_cell(row['readout_energy'], '.4f', 8)}  "
            f"{_cell(row['cross_layer_overlap'], '.4f', 8)}  "
            f"{_cell(row['sv_s1'], '.3f', 10)}  {_cell(row['sv_s8'], '.3f', 10)}  "
            f"{_cell(row['sv_sk'], '.3f', 10)}  {_cell(row['participation_ratio'], '.1f', 10)}"
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.svd_k < 1:
        raise SystemExit("--svd-k must be >= 1")
    if args.r_span < 1:
        raise SystemExit("--r-span must be >= 1")
    if args.backend == "tiny" and args.unembed:
        print("note: --unembed is ignored with --backend tiny")
    device = torch.device(args.device)

    lenses, _ = load_lens_set(args.lens_dir)
    if args.mask not in lenses:
        raise ValueError(f"mask {args.mask!r} not in {args.lens_dir} (found {sorted(lenses)})")
    lens = lenses[args.mask]
    source_layers = list(lens.source_layers)
    if not source_layers:
        raise SystemExit(f"no fitted layers in {args.lens_dir} for mask {args.mask!r}")
    d_model = lens.d_model
    final_fitted = source_layers[-1]
    k = min(args.svd_k, d_model)

    W_U = load_unembed(args, d_model)
    vocab = int(W_U.shape[0])
    r = min(args.r_span, vocab, d_model)

    # The readout span, one randomized SVD: W_U's top-r right singular directions. The
    # projector P_span = V_r^T V_r is never materialised; its quadratic form and trace
    # are applied through V_r directly (trace(P_span) = ||V_r||_F^2 for orthonormal rows).
    torch.manual_seed(SVD_SEED)
    _, _, Vw = torch.svd_lowrank(W_U, q=r, niter=SVD_NITER)
    V_r = Vw.transpose(0, 1).contiguous()  # [r, d]: rows are Vh[:r]
    baseline = float((V_r.to(torch.float64) ** 2).sum()) / d_model  # trace(P_span)/d

    print(
        f"readout span: mask={args.mask} d_model={d_model} n_prompts={lens.n_prompts} "
        f"device={device}; fitted J at {source_layers}; final fitted layer {final_fitted}"
    )
    print(
        f"W_U svd k={r} (niter={SVD_NITER}, seed={SVD_SEED}), J svd k={k}; "
        f"random baseline trace(P_span)/d = {baseline:.6f}"
    )

    layer_rows: dict[str, dict[str, float | None]] = {}
    with torch.no_grad():
        # The final fitted layer's vectors are the cross-layer reference: pop and SVD it
        # first, then walk the remaining layers one at a time (popped as processed) so the
        # CPU footprint stays flat; the fp32 cast keeps svd_lowrank happy even for an
        # fp16-saved lens file.
        J_final = lens.jacobians.pop(final_fitted).to(device).float()
        spec_final = spectrum(J_final, k)
        del J_final
        S_final, V_final = spec_final if spec_final is not None else (None, None)
        for layer in source_layers:
            if layer == final_fitted:
                layer_rows[str(layer)] = overlap_row(
                    S_final, V_final, V_r=V_r, V_ref=V_final, is_final=True
                )
                continue
            J = lens.jacobians.pop(layer).to(device).float()
            spec = spectrum(J, k)
            del J
            S, V = spec if spec is not None else (None, None)
            layer_rows[str(layer)] = overlap_row(S, V, V_r=V_r, V_ref=V_final, is_final=False)

    text = digest_table(layer_rows, source_layers, baseline=baseline)
    print()
    print(text)
    print(f"\nprocessed {len(source_layers)} layers; svd k={k}, r={r} (niter={SVD_NITER})")

    meta = {
        "lens_dir": str(args.lens_dir),
        "unembed": "tiny fixture" if args.backend == "tiny" else str(args.unembed),
        "w_u_shape": [vocab, d_model],
        "n_prompts": lens.n_prompts,
        "r_span": args.r_span,
        "r_span_effective": r,
        "svd_k": args.svd_k,
        "svd_k_effective": k,
        "svd_niter": SVD_NITER,
        "svd_seed": SVD_SEED,
        "device": str(device),
        "final_fitted_layer": final_fitted,
        "readout_energy_baseline": round(baseline, 6),
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
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {args.digest}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
