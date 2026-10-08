#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Lens-zoo data-scaling eval: every fitted lens scored on ONE held-out manifest.

The campaign's lenses sit at different data scales (the 20-image x1 shards, the 50-image
s2 halves, the 100-image s2-merged) on the same COCO corpus family, so they are scoreable
on one held-out manifest without new fits. One model load scores the whole zoo plus ONE
``use_jacobian=False`` pass of the untrained logit-lens baseline (once total, not per
lens), giving the data-scaling curve D35/D39 call for (the small-``n`` caveat) under the
D19 rank-vs-KL tension, where the tail-heavy mean true-token rank needs a fairer read:

* per-layer ``rank_ratio`` = ``mean_rank_true / model_mean_rank_true`` (1.0 = the model's
  own readout at that layer);
* LQS = mean over the layers present in BOTH the lens rows and the baseline rows of
  ``log(baseline.mean_rank_true / row.mean_rank_true)`` — positive = beats the baseline.

Lenses whose ``source_layers`` are a subset of the fit score fine: ``score_table_fast``
unions the final layer, and LQS is taken only over common layers. An optional
``--bias-dir`` (the moment census) is passed through to ``score_table_fast``; the
baseline readout never transports, so it stays unbiased by construction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from vlm_lens.data.manifest import read_manifest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_eval  # noqa: E402

#: Layers the scaling table reports rank_ratio at (the per-8 grid plus the L20 bump / L30).
TABLE_LAYERS = (0, 8, 16, 20, 24, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout-manifest", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--mask", default="text")
    parser.add_argument("--tags", default="text")
    parser.add_argument("--skip-first", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--limit", type=int, default=None, help="score only the first N samples (smoke)")
    parser.add_argument(
        "--backend", choices=("hf-llava", "tiny"), default="hf-llava",
        help="model backend: 'hf-llava' (default, CUDA) or the tiny CPU smoke fixture",
    )
    parser.add_argument(
        "--bias-dir", default=None, help="moment-census dir; passed through to score_table_fast"
    )
    parser.add_argument(
        "--lens", action="append", dest="lenses", default=None,
        help="zoo entry as name=artifacts-dir; repeatable; n_prompts is read from the loaded lens",
    )
    return parser.parse_args()


def parse_lens_entries(entries: list[str] | None) -> list[tuple[str, str]]:
    """Split ``name=dir`` entries; a missing or repeated name/dir is a CLI error."""
    if not entries:
        raise SystemExit("--lens name=dir is required (repeatable)")
    spec: list[tuple[str, str]] = []
    seen: set[str] = set()
    for entry in entries:
        name, sep, lens_dir = entry.partition("=")
        name, lens_dir = name.strip(), lens_dir.strip()
        if not sep or not name or not lens_dir:
            raise SystemExit(f"--lens must be name=dir, got {entry!r}")
        if name in seen:
            raise SystemExit(f"--lens name {name!r} given twice")
        seen.add(name)
        spec.append((name, lens_dir))
    return spec


def lqs(rows: list[s2_eval.ExtendedScore], baseline_rows: list[s2_eval.ExtendedScore]) -> float:
    """Mean log rank improvement over the baseline over the common (layer, tag) cells.

    Positive = the lens beats the baseline (lower true-token rank). Only cells present in
    both row lists count, so a lens whose ``source_layers`` are a subset of the fit is fine.
    """
    baseline = {(row.layer, row.tag): row for row in baseline_rows}
    ratios = [
        math.log(baseline[(row.layer, row.tag)].mean_rank_true / row.mean_rank_true)
        for row in rows
        if (row.layer, row.tag) in baseline
    ]
    if not ratios:
        raise SystemExit("no (layer, tag) cells present in both the lens and baseline rows")
    return sum(ratios) / len(ratios)


def print_scaling_table(zoo: dict[str, dict]) -> None:
    """name / n_prompts / LQS / rank_ratio at the table layers ('-' where not fitted)."""
    header = (
        f"{'name':<12}  {'n_prompts':>9}  {'LQS':>7}   "
        + "  ".join(f"ratio_L{layer}" for layer in TABLE_LAYERS)
    )
    print(header)
    print("-" * len(header))
    for name, entry in zoo.items():
        ratios = entry["rank_ratio"]
        cells = "  ".join(
            f"{ratios[str(layer)]:>9.3f}" if str(layer) in ratios else f"{'-':>9}"
            for layer in TABLE_LAYERS
        )
        print(f"{name:<12}  {entry['n_prompts']:>9}  {entry['lqs']:>7.3f}   {cells}")


def main() -> int:
    args = parse_args()
    tags = tuple(part.strip() for part in args.tags.split(",") if part.strip())
    spec = parse_lens_entries(args.lenses)
    heldout = read_manifest(args.heldout_manifest)
    if args.limit:
        heldout = heldout[: args.limit]
    model = s2_eval.load_model(args)
    print(
        f"== lens zoo: {len(spec)} lenses, {len(heldout)} held-out samples, "
        f"mask={args.mask}, tags={','.join(tags)} =="
    )

    # The untrained logit-lens baseline: scored once total (not per lens) with
    # use_jacobian=False on the first zoo entry's lens dir (only its layer list matters;
    # the baseline readout never transports, so it is unbiased by construction).
    _, _, baseline_map = s2_eval.score_table_fast(
        model, spec[0][1], heldout, mask=args.mask, tags=tags,
        skip_first=args.skip_first, max_seq_len=args.max_seq_len,
        modes=("default",), use_jacobian=False,
    )
    baseline_rows = baseline_map["default"]

    zoo: dict[str, dict] = {}
    for name, lens_dir in spec:
        lens, _, rows_map = s2_eval.score_table_fast(
            model, lens_dir, heldout, mask=args.mask, tags=tags,
            skip_first=args.skip_first, max_seq_len=args.max_seq_len,
            modes=("default",), use_jacobian=True, bias_dir=args.bias_dir,
        )
        rows = rows_map["default"]
        zoo[name] = {
            "n_prompts": int(lens.n_prompts),
            "layers": sorted(lens.source_layers),
            "rows": [row.to_json() for row in rows],
            "lqs": lqs(rows, baseline_rows),
            "rank_ratio": {
                str(row.layer): row.mean_rank_true / row.model_mean_rank_true for row in rows
            },
        }
        print(f"[zoo] {name}: n_prompts={lens.n_prompts} LQS={zoo[name]['lqs']:.3f}")

    print("\n== data-scaling table (rank_ratio = mean_rank_true / model_mean_rank_true) ==")
    print_scaling_table(zoo)

    report = {
        "heldout_manifest": args.heldout_manifest,
        "n_heldout_samples": len(heldout),
        "skip_first": args.skip_first,
        "mask": args.mask,
        "tags": list(tags),
        "bias_dir": args.bias_dir,
        "zoo": zoo,
        "logit_baseline": {"rows": [row.to_json() for row in baseline_rows]},
    }
    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
