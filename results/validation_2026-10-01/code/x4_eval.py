#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""X4 (conditional): does the source-mask boundary change the caption lens?

The label-length structure ``<image> <question> <answer>`` only reaches back past the
question through the image block, which is why V3 rejects the ALD-only justification for
``skip_first=1``. This script scores several skip_first variants (same shard, same layer
set) on held-out captions and reports the text-tag tables side by side, so the report can
state the *empirical* sensitivity instead of asserting one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

import vlm_lens  # noqa: E402, F401  # installs the vendored jlens path: must precede jlens
from vlm_lens.artifacts import load_lens_set  # noqa: E402
from vlm_lens.data.manifest import read_manifest  # noqa: E402
from vlm_lens.evaluate import format_scores, score_lens  # noqa: E402
from vlm_lens.models.llava import LlavaLensModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", required=True, nargs="+", metavar="SF=DIR")
    parser.add_argument("--heldout-manifest", required=True)
    parser.add_argument("--mask", default="text")
    parser.add_argument("--tags", default="text,image,all")
    parser.add_argument("--json", required=True)
    parser.add_argument("--max-seq-len", type=int, default=1536)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tags = tuple(part.strip() for part in args.tags.split(",") if part.strip())
    heldout = read_manifest(args.heldout_manifest)
    model = LlavaLensModel.from_pretrained(
        dtype=torch.bfloat16, device="cuda", local_files_only=True
    )
    report: dict[str, object] = {"heldout": args.heldout_manifest, "n_heldout": len(heldout)}
    tables: dict[str, list] = {}
    for spec in args.variants:
        skip_first, lens_dir = spec.split("=", 1)
        # Scoring uses a fixed skip_first=1 mask for every variant: only the *lens's* fit
        # boundary changes, not the scoring protocol.
        lenses, provenance = load_lens_set(lens_dir)
        lens = lenses[args.mask]
        rows = score_lens(
            model, lens, heldout, tags=tags, skip_first=1, max_seq_len=args.max_seq_len
        )
        tables[skip_first] = [row.to_json() for row in rows]
        report[skip_first] = {
            "lens_dir": lens_dir,
            "n_prompts": int(lens.n_prompts),
            "provenance": {key: provenance.get(key) for key in ("skip_first", "masks", "notes")},
            "rows": tables[skip_first],
        }
        print(f"\n== X4 skip_first={skip_first} (fitting boundary) ==")
        print(format_scores(rows))

    last_layers = {key: table[-1] for key, table in tables.items()}
    print("\n== X4 layer-31 text-tag summary ==")
    for key, row in last_layers.items():
        print(
            f"fit skip_first={key:<3} rank={row['mean_rank_true']:.2f} "
            f"top1={row['top1_agreement']:.3f} kl={row['mean_kl']:.3f} "
            f"model_rank={row['model_mean_rank_true']:.2f}"
        )
    report["summary_l31_text"] = last_layers

    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
