#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Compose a re-paired lens: per-layer ``J'_l = J_{l + delta_l}`` with per-layer deltas.

The synth zoo found the (map, state) pairing suboptimal: the FORWARD layer shift (+4) beats the
correct pairing on the held-out (LQS +0.524 vs -0.102, both uncorrected) while the backward shift
hurts - later maps transport earlier states better. This script selects the best shift per layer
from a fit-split shift sweep (the phase5c zoo on half_a; no held-out leakage), composes the
re-paired Jacobian, and writes a loadable lens dir the standard zoo evaluator scores.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from vlm_lens.artifacts import load_lens_set, save_lens_set  # noqa: E402

__all__ = ["main", "parse_args"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-lens-dir", required=True, help="fitted lens dir (the J source)")
    parser.add_argument("--selection", required=True, help="fit-split shift zoo JSON (the winners)")
    parser.add_argument("--out", required=True, help="output lens dir (artifacts/ + provenance)")
    parser.add_argument("--mask", default="text", help="which lens file to re-pair")
    return parser.parse_args()


def _winner_deltas(
    selection_path: Path, layers: list[int], tag: str
) -> dict[int, dict[str, object]]:
    """Per-layer best shift name + delta from the fit-split zoo rows (lowest mean rank)."""
    zoo = json.loads(selection_path.read_text(encoding="utf-8"))["zoo"]
    rows_by_shift = {
        name: {(row["layer"], row["tag"]): row for row in entry["rows"]}
        for name, entry in zoo.items()
    }
    deltas: dict[int, dict[str, object]] = {}
    for layer in layers:
        best_name, best_rank = None, None
        for name, rows in rows_by_shift.items():
            cell = rows.get((layer, tag))
            if cell is None:
                continue
            if best_rank is None or cell["mean_rank_true"] < best_rank:
                best_name, best_rank = name, cell["mean_rank_true"]
        delta = 0
        if best_name is not None and best_name.startswith("shift") and len(best_name) > 6:
            magnitude = int(best_name[6:])
            delta = magnitude if best_name[5] == "P" else -magnitude
        deltas[layer] = {"shift": best_name, "delta": delta, "fit_rank": best_rank}
    return deltas


def _clamp_target(index: int, fitted: set[int]) -> int:
    """Nearest fitted layer to ``index`` (ties resolve to the lower index, deterministic)."""
    if index in fitted:
        return index
    lower = [x for x in fitted if x < index]
    upper = [x for x in fitted if x > index]
    if lower and (not upper or index - max(lower) <= min(upper) - index):
        return max(lower)
    return min(upper) if upper else max(fitted)


def main() -> int:
    args = parse_args()
    lenses, src_prov = load_lens_set(args.src_lens_dir)
    lens = lenses[args.mask]
    jacobians = lens.jacobians
    layers = sorted(jacobians)
    deltas = _winner_deltas(Path(args.selection), layers, args.mask)

    fitted = set(layers)
    out_jac: dict[int, torch.Tensor] = {}
    for layer in layers:
        delta = int(deltas[layer]["delta"])
        out_jac[layer] = jacobians[_clamp_target(layer + delta, fitted)].contiguous().float()

    composed = type(lens)(jacobians=out_jac, n_prompts=lens.n_prompts, d_model=lens.d_model)
    provenance = {
        key: value
        for key, value in src_prov.items()
        if key not in ("files", "artifacts")
    }
    provenance["repaired"] = {
        "selection": str(Path(args.selection).resolve()),
        "per_layer_shift": {str(layer): deltas[layer]["shift"] for layer in layers},
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "note": "J'_l = J_clamp(l + delta_l); deltas selected on the fit split (no held-out leakage)",
    }
    out_dir = Path(args.out)
    written = save_lens_set(out_dir / "artifacts", {args.mask: composed}, provenance=provenance)
    for mask, path in sorted(written.items()):
        print(f"  lens-{mask}.pt  n_prompts={composed.n_prompts}  {path}")

    winners: dict[str, int] = {}
    for layer in layers:
        name = str(deltas[layer]["shift"])
        winners[name] = winners.get(name, 0) + 1
    print(f"per-layer winners: {winners}")
    print(f"wrote {out_dir / 'artifacts'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
