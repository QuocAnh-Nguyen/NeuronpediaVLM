#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Synthetic lens variants: rewrite a fitted lens dir into readout-hypothesis controls.

The zoo evaluator (``lens_zoo_eval.py`` -> ``s2_eval.score_table_fast``) consumes a lens
directory as ``--lens name=DIR --mask text`` and loads ``artifacts/lens-text.pt``, so a
synthetic variant is just another zoo entry: this script derives variant ``J'_l`` matrices
from ONE fitted lens dir (zero new fits, no model load) and writes
``OUT_ROOT/<variant>/artifacts/lens-text.pt`` plus ``provenance.json`` per variant, and
``OUT_ROOT/variants.json`` with the per-layer relative-Frobenius distance to the source.

Per fitted layer ``l`` (``J'_l`` in ``[d, d]`` float32, contiguous):

* ``alphaI`` - ``J'_l = alpha_l * I`` with ``alpha_l = trace(J_l) / d`` (scaled-identity
  control: how much of the readout is just a per-layer scalar?);
* ``diag`` - ``J'_l = diag(J_l)`` as a dense matrix (off-diagonal transport dropped);
* ``rank64`` / ``rank256`` - truncated randomized SVD, ``J'_l = U_k diag(s_k) Vh_k``
  (``torch.svd_lowrank``; ``torch.svds`` no longer exists in torch) - how much rank does
  the transport need?;
* ``shiftP<N>`` / ``shiftM<N>`` - ``J'_l = J_clamp(l±N, min_layer, max_layer)`` for any
  N >= 1 (nearest fitted layer; separates source-state vs transport quality).

Each variant set is ~2 GB at d=4096 (32 layers x 64 MiB per [4096, 4096] float32 matrix);
the source J is held in memory once and every variant writes from it. Provenance follows
the ``fit_llava.py run_merge`` pattern: the source identity is echoed, ``created_utc`` is
fresh, and every value stays a weights_only-safe plain primitive.

Example:
    python synth_lenses.py --src-lens-dir RUN/s2-merged/artifacts --out-root RUN/synth
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
from jlens.lens import JacobianLens  # noqa: E402

from vlm_lens.artifacts import MASK_FILENAME, load_lens_set, save_lens_set  # noqa: E402

#: Every synthetic variant this script knows; ``--variants`` picks a subset.
ALL_VARIANTS = ("alphaI", "diag", "rank64", "rank256", "shiftP4", "shiftM4")
#: The randomized-SVD variants and their target ranks (clamped to d_model).
RANK_VARIANTS = {"rank64": 64, "rank256": 256}
#: Seed for the randomized SVD start, recorded in provenance so artifacts reproduce.
SVD_SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src-lens-dir", required=True,
        help="fitted lens directory (or a single lens-<mask>.pt file) to derive variants from",
    )
    parser.add_argument("--mask", default="text", help="which source lens file to transform")
    parser.add_argument(
        "--out-root", required=True, help="variants are written to OUT_ROOT/<variant>/artifacts/"
    )
    parser.add_argument(
        "--variants", default=",".join(ALL_VARIANTS),
        help=f"comma-separated subset of {','.join(ALL_VARIANTS)} (default: all)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="torch device for the randomized-SVD transforms (the other variants are cheap CPU "
        "ops); artifacts always save CPU-side",
    )
    return parser.parse_args()


def _alpha_identity(jacobians: dict[int, torch.Tensor], d_model: int) -> dict[int, torch.Tensor]:
    """``J'_l = alpha_l * I`` with ``alpha_l = trace(J_l) / d`` (scaled-identity control)."""
    eye = torch.eye(d_model, dtype=torch.float32)
    out: dict[int, torch.Tensor] = {}
    for layer, J in jacobians.items():
        # fp64 diagonal sum: the trace drives the whole variant, so do not lose it to
        # float32 accumulation over d_model terms.
        alpha = float(torch.diagonal(J).to(torch.float64).sum().item()) / d_model
        out[layer] = (alpha * eye).contiguous()
    return out


def _diagonal_only(jacobians: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    """``J'_l = diag(J_l)`` as a dense matrix (off-diagonal transport dropped)."""
    return {layer: torch.diag(torch.diagonal(J)).contiguous() for layer, J in jacobians.items()}


def _lowrank(
    jacobians: dict[int, torch.Tensor], rank: int, device: torch.device
) -> dict[int, torch.Tensor]:
    """``J'_l = U_k diag(s_k) Vh_k``: truncated randomized SVD to rank ``k``.

    ``torch.svds`` no longer exists in torch; ``torch.svd_lowrank`` is the randomized
    truncated SVD (seeded by the caller for reproducible artifacts). Seven power
    iterations land within ~0.2% of the exact rank-k (Eckart-Young) optimum even on the
    flat spectra a fitted average-Jacobian tends to have; the default two do not.
    """
    out: dict[int, torch.Tensor] = {}
    for layer, J in jacobians.items():
        Jd = J.to(device)
        k = min(rank, Jd.shape[0])
        U, S, V = torch.svd_lowrank(Jd, q=k, niter=7)
        recon = U @ torch.diag(S) @ V.T
        out[layer] = recon.to(torch.float32).contiguous().cpu()
    return out


def _shift(jacobians: dict[int, torch.Tensor], delta: int) -> dict[int, torch.Tensor]:
    """``J'_l = J_clamp(l+delta, min_layer, max_layer)``: the nearest fitted layer.

    When the shifted index is itself fitted it is used as-is; otherwise the nearest
    fitted layer wins (ties resolve to the lower index, deterministically).
    """
    layers = sorted(jacobians)
    lo, hi = layers[0], layers[-1]
    out: dict[int, torch.Tensor] = {}
    for layer in layers:
        target = min(max(layer + delta, lo), hi)
        nearest = min(layers, key=lambda fitted: (abs(fitted - target), fitted))
        out[layer] = jacobians[nearest].contiguous()
    return out


def _is_shift(name: str) -> bool:
    """``shiftP<N>`` / ``shiftM<N>`` for any N >= 1."""
    return (
        len(name) > 6
        and name.startswith("shift")
        and name[5] in ("P", "M")
        and name[6:].isdigit()
    )


def _shift_delta(name: str) -> int:
    """+N for ``shiftP<N>``, -N for ``shiftM<N>``; KeyError on anything else."""
    magnitude = int(name[6:])
    return magnitude if name[5] == "P" else -magnitude


def variant_description(name: str, d_model: int) -> str:
    """One-line description of the variant, stored in provenance and variants.json."""
    if name == "alphaI":
        return "scaled identity: J'_l = alpha_l * I, alpha_l = trace(J_l)/d"
    if name == "diag":
        return "diagonal-only: J'_l = diag(J_l) as a dense matrix (off-diagonal transport dropped)"
    if name in RANK_VARIANTS:
        k = min(RANK_VARIANTS[name], d_model)
        return f"truncated randomized SVD to rank {k}: J'_l = U_{k} diag(s_{k}) Vh_{k} (seeded)"
    if _is_shift(name):
        delta = _shift_delta(name)
        sign = "+" if delta > 0 else "-"
        return (
            f"layer shift {sign}{abs(delta)}: J'_l = J_clamp(l{sign}{abs(delta)}, min_layer, "
            "max_layer), nearest fitted layer"
        )
    raise KeyError(name)


def build_variant(
    name: str,
    jacobians: dict[int, torch.Tensor],
    d_model: int,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """The variant's ``{layer: J'_l}`` float32 contiguous dict for one variant name."""
    if name == "alphaI":
        return _alpha_identity(jacobians, d_model)
    if name == "diag":
        return _diagonal_only(jacobians)
    if name in RANK_VARIANTS:
        torch.manual_seed(SVD_SEED)  # per variant: rank64/rank256 reproduce independently
        return _lowrank(jacobians, RANK_VARIANTS[name], device)
    if _is_shift(name):
        return _shift(jacobians, _shift_delta(name))
    raise KeyError(name)


def main() -> int:
    args = parse_args()
    variants = [part.strip() for part in args.variants.split(",") if part.strip()]
    unknown = sorted(name for name in variants if not (name in ALL_VARIANTS or _is_shift(name)))
    if unknown:
        raise SystemExit(
            f"unknown variants {unknown}; choose from {list(ALL_VARIANTS)} or shiftP<N>/shiftM<N>"
        )
    if not variants:
        raise SystemExit("--variants parsed to nothing")
    device = torch.device(args.device)

    src_dir = Path(args.src_lens_dir)
    lenses, src_prov = load_lens_set(src_dir)
    if args.mask not in lenses:
        raise ValueError(f"mask {args.mask!r} not in {src_dir} (found {sorted(lenses)})")
    lens = lenses[args.mask]
    jacobians = lens.jacobians  # {layer: [d,d] float32} - the fitted transport being transformed
    d_model = lens.d_model
    src_sha = ((src_prov.get("artifacts") or {}).get(args.mask) or {}).get("sha256")
    print(
        f"source: {src_dir} mask={args.mask} n_prompts={lens.n_prompts} d_model={d_model} "
        f"layers={len(lens.source_layers)} device={device}"
    )

    out_root = Path(args.out_root)
    report: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Source provenance identity: the zoo consumer reads n_prompts/d_model/source_layers
        # from the lens itself; the rest identifies where the fitted J came from.
        "src": {
            "lens_dir": str(src_dir),
            "mask": args.mask,
            "n_prompts": int(lens.n_prompts),
            "d_model": int(d_model),
            "source_layers": list(lens.source_layers),
            "estimator": src_prov.get("estimator"),
            "created_utc": src_prov.get("created_utc"),
            "model": src_prov.get("model"),
            "corpus": src_prov.get("corpus"),
            "src_lens_sha256": src_sha,
        },
        "variants": {},
    }

    for name in variants:
        desc = variant_description(name, d_model)
        variant_jacobians = build_variant(name, jacobians, d_model, device)
        out_dir = out_root / name / "artifacts"
        # run_merge provenance pattern: the source identity is echoed, created_utc is fresh,
        # and the stale source files-meta is dropped (save_lens_set writes its own
        # artifacts key); everything stays JSON-plain for the weights_only unpickler.
        provenance = {
            **{key: value for key, value in src_prov.items() if key != "artifacts"},
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_prompts": {args.mask: int(lens.n_prompts)},
            "synthetic": {
                "generator": str(Path(__file__).resolve()),
                "variant": name,
                "description": desc,
                "derived_from": {"lens_dir": str(src_dir), "mask": args.mask, "sha256": src_sha},
                "device": str(device),
                **({"svd_seed": SVD_SEED} if name in RANK_VARIANTS else {}),
            },
        }
        # float32 on disk (save_lens_set defaults to float16): the variant schema stays
        # upstream-compatible, so the zoo reads it with no special casing.
        save_lens_set(
            out_dir,
            {args.mask: JacobianLens(variant_jacobians, n_prompts=lens.n_prompts, d_model=d_model)},
            provenance=provenance,
            dtype=torch.float32,
        )

        # ||J'-J||_F / ||J||_F per layer: how far the variant sits from the fitted transport.
        rel_fro: dict[str, float] = {}
        for layer in lens.source_layers:
            J = jacobians[layer]
            num = (variant_jacobians[layer] - J).norm().item()
            den = J.norm().item()
            rel_fro[str(layer)] = num / den if den > 0.0 else (0.0 if num == 0.0 else float("inf"))
        mean_rel = sum(rel_fro.values()) / len(rel_fro)
        report["variants"][name] = {"dir": str(out_dir), "description": desc, "rel_fro": rel_fro}
        print(
            f"[{name}] wrote {out_dir / MASK_FILENAME.format(mask=args.mask)} "
            f"n_prompts={lens.n_prompts} rel_fro={mean_rel:.3f}"
        )

    variants_path = out_root / "variants.json"
    variants_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {variants_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
