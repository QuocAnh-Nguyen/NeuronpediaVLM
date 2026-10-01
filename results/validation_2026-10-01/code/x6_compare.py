#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""X6: bf16 vs fp32 — per-layer relative Frobenius difference of the merged Jacobian.

Rule (register X6): if the median relative difference across layers exceeds 3 %, or the
maximum exceeds 10 %, later fits use fp32 (with ``dim_batch`` reduced as needed); else
bfloat16 stands. Each layer row also reports the best-fit scale and the cosine similarity,
so a pure scale difference can be told apart from a structural one.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from vlm_lens.artifacts import load_lens_set  # noqa: E402

REL_MEDIAN_LIMIT = 0.03
REL_MAX_LIMIT = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir-a", required=True, help="first run directory (artifacts/)")
    parser.add_argument("--dir-b", required=True, help="second run directory (artifacts/)")
    parser.add_argument("--label-a", default="bf16")
    parser.add_argument("--label-b", default="fp32")
    parser.add_argument("--json", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lenses_a, _ = load_lens_set(args.dir_a)
    lenses_b, _ = load_lens_set(args.dir_b)
    if set(lenses_a) != set(lenses_b):
        raise SystemExit(f"mask sets differ: {sorted(lenses_a)} vs {sorted(lenses_b)}")

    report: dict[str, object] = {"labels": [args.label_a, args.label_b], "masks": {}}
    for mask in sorted(lenses_a):
        lens_a, lens_b = lenses_a[mask], lenses_b[mask]
        rows: dict[str, dict[str, float]] = {}
        for layer in sorted(set(lens_a.jacobians) & set(lens_b.jacobians)):
            a = lens_a.jacobians[layer].float()
            b = lens_b.jacobians[layer].float()
            norm_a = float(a.norm())
            difference = float((b - a).norm() / norm_a)
            scale = float((a * b).sum() / (a * a).sum())
            scaled_residual = float((b - scale * a).norm() / norm_a)
            cosine = float(
                (a * b).sum() / (a.norm() * b.norm() + 1e-30)
            )
            rows[str(layer)] = {
                "rel_frobenius": round(difference, 5),
                "cosine": round(cosine, 6),
                "best_scale": round(scale, 5),
                "rel_after_scale": round(scaled_residual, 5),
            }
        values = [row["rel_frobenius"] for row in rows.values()]
        median = statistics.median(values)
        report["masks"][mask] = {
            "n_prompts": [int(lens_a.n_prompts), int(lens_b.n_prompts)],
            "per_layer": rows,
            "median_rel": round(median, 5),
            "max_rel": round(max(values), 5),
        }
        print(f"\nmask={mask}  n_prompts={lens_a.n_prompts}/{lens_b.n_prompts}")
        print("layer  rel_fro   cosine   best_scale  rel_after_scale")
        for layer, row in rows.items():
            print(
                f"{layer:>5}  {row['rel_frobenius']:.5f}  {row['cosine']:.6f}  "
                f"{row['best_scale']:.5f}     {row['rel_after_scale']:.5f}"
            )
        print(f"median={median:.5f}  max={max(values):.5f}")

    medians = [entry["median_rel"] for entry in report["masks"].values()]
    maxima = [entry["max_rel"] for entry in report["masks"].values()]
    worst_median, worst_max = max(medians), max(maxima)
    report["verdict"] = {
        "worst_median_rel": round(worst_median, 5),
        "worst_max_rel": round(worst_max, 5),
        "use_fp32": worst_median > REL_MEDIAN_LIMIT or worst_max > REL_MAX_LIMIT,
        "rule": f"fp32 if median > {REL_MEDIAN_LIMIT} or max > {REL_MAX_LIMIT}",
    }
    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nverdict: {json.dumps(report['verdict'])}")
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
