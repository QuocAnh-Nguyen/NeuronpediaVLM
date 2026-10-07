#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""A/B check: s2_eval.score_table_fast vs the reference CPU-metric scorer (evaluate.score_lens).

Runs both on the same real samples (main-style tags, both modes) and prints per-field max
abs differences plus exact-match counts for the integer metrics. Exit code 1 if the integer
metrics differ at all or KL drifts more than 1e-4 relative.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from vlm_lens.artifacts import load_lens_set  # noqa: E402
from vlm_lens.data.manifest import read_manifest  # noqa: E402
from vlm_lens.evaluate import score_lens  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_eval  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lens-dir", required=True)
    ap.add_argument("--heldout-manifest", required=True)
    ap.add_argument("--n-samples", type=int, default=1)
    ap.add_argument("--tags", default=",".join(s2_eval.TAGS))
    args = ap.parse_args()
    tags = tuple(t.strip() for t in args.tags.split(",") if t.strip())
    model = s2_eval.load_model(argparse.Namespace(backend="hf-llava"))
    samples = read_manifest(args.heldout_manifest)[: args.n_samples]
    lenses, _ = load_lens_set(args.lens_dir)
    lens = lenses["text"]

    print("== reference (CPU metrics) ==")
    ref_default = score_lens(model, lens, samples, tags=tags, skip_first=1, max_seq_len=1536)
    ref_ip = score_lens(
        model, lens, samples, tags=tags, skip_first=1, max_seq_len=1536,
        include_placeholders=True,
    )
    print("== fast (GPU metrics) ==")
    _, _, fast = s2_eval.score_table_fast(
        model, args.lens_dir, samples, mask="text", tags=tags, skip_first=1, max_seq_len=1536,
        modes=("default", "include_placeholders"),
    )

    for name, ref_rows, fast_rows in (
        ("default", ref_default, fast["default"]),
        ("include_placeholders", ref_ip, fast["include_placeholders"]),
    ):
        ref_map = {(r.layer, r.tag): r for r in ref_rows}
        fast_map = {(r.layer, r.tag): r for r in fast_rows}
        if set(ref_map) != set(fast_map):
            print(f"{name}: CELL MISMATCH {sorted(set(ref_map) ^ set(fast_map))[:8]}")
            return 1
        max_rank = max_agree = max_mr = 0.0
        max_kl_rel = 0.0
        for key in ref_map:
            a, b = ref_map[key], fast_map[key]
            max_rank = max(max_rank, abs(a.mean_rank_true - b.mean_rank_true))
            max_agree = max(max_agree, abs(a.top1_agreement - b.top1_agreement))
            max_mr = max(max_mr, abs(a.model_mean_rank_true - b.model_mean_rank_true))
            if a.mean_kl:
                max_kl_rel = max(max_kl_rel, abs(a.mean_kl - b.mean_kl) / abs(a.mean_kl))
        ok = max_rank == 0 and max_agree == 0 and max_mr == 0 and max_kl_rel < 1e-4
        print(
            f"{name}: cells={len(ref_map)} max|drank|={max_rank:.6f} "
            f"max|dagree|={max_agree:.6f} max|dmodel_rank|={max_mr:.6f} "
            f"max_rel_dKL={max_kl_rel:.2e} -> {'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            return 1
    print("A/B PASS (integer metrics exact; KL within 1e-4 relative)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
