#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""X11: untrained logit-lens baseline on the caption held-out (D34 diagnostics).

Scores the same samples/positions/targets as `s2_eval.py` but with `use_jacobian=False`:
the readout is `unembed(final_norm(h_l))` - the vanilla logit lens, J = I. Comparing its
per-layer rank/KL against the fitted S2 rows attributes the D34 findings (the L20 rank bump
and the steep L30->L31 convergence) to either the model's mid-layer states (shared by both
readouts) or the fitted average-Jacobian map (lens-specific).

Reuses the A/B-verified GPU-metric scorer; same JSON row schema as s2_eval's tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from vlm_lens.data.manifest import read_manifest  # noqa: E402
from vlm_lens.evaluate import format_scores  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_eval  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lens-dir", required=True, help="merged lens dir (layer list + masks only)")
    ap.add_argument("--heldout-manifest", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--tags", default=",".join(s2_eval.TAGS))
    ap.add_argument("--mask", default="text")
    ap.add_argument("--skip-first", type=int, default=1)
    ap.add_argument("--max-seq-len", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    tags = tuple(part.strip() for part in args.tags.split(",") if part.strip())

    model = s2_eval.load_model(argparse.Namespace(backend="hf-llava"))
    heldout = read_manifest(args.heldout_manifest)
    if args.limit:
        heldout = heldout[: args.limit]
    print(f"== X11 logit-lens baseline (use_jacobian=False), {len(heldout)} samples ==")
    lens, provenance, rows = s2_eval.score_table_fast(
        model,
        args.lens_dir,
        heldout,
        mask=args.mask,
        tags=tags,
        skip_first=args.skip_first,
        max_seq_len=args.max_seq_len,
        modes=("default",),
        use_jacobian=False,
    )
    report = {
        "experiment": "X11_logit_lens_baseline",
        "lens_dir": args.lens_dir,
        "heldout_manifest": args.heldout_manifest,
        "n_samples": len(heldout),
        "skip_first": args.skip_first,
        "tags": list(tags),
        "rows": [row.to_json() for row in rows["default"]],
    }
    print(format_scores(rows["default"]))
    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
