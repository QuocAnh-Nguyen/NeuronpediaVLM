#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Regression test for the analysis scripts' pure check logic (no model, no GPU).

Covers ``s1_score.depth_checks`` (gate check (iv)): the depth-trend rule that decides the S1
go/no-go, plus the monotonicity diagnostics. Run anywhere with torch installed:

    python results/validation_2026-10-01/code/test_analysis_logic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s1_score import depth_checks  # noqa: E402


def _rows(rank_at) -> list[dict]:
    return [{"layer": layer, "rank_j": rank_at(layer)} for layer in range(32)]


def main() -> int:
    checks: list[tuple[str, bool]] = []

    clean = depth_checks(_rows(lambda layer: 100.0 - layer * 3), 32)
    checks += [
        ("clean improvement passes the trend", clean["iv_rank_improves_with_depth"] is True),
        ("clean improvement is monotone", clean["iv_rank_monotone_last_half"] is True),
        ("no non-monotone steps recorded", clean["iv_n_nonmonotone_steps_last_half"] == 0),
        ("mid/last ranks reported", (clean["iv_rank_mid"], clean["iv_rank_last"]) == (52.0, 7.0)),
    ]

    wobble = _rows(lambda layer: 100.0 - layer * 3)
    wobble[30]["rank_j"] = wobble[29]["rank_j"] + 0.5  # finite-shard wobble
    d = depth_checks(wobble, 32)
    checks += [
        ("finite-shard wobble keeps the trend", d["iv_rank_improves_with_depth"] is True),
        ("wobble turns off strict monotonicity", d["iv_rank_monotone_last_half"] is False),
        ("wobble counts one non-monotone step", d["iv_n_nonmonotone_steps_last_half"] == 1),
    ]

    degrade = _rows(lambda layer: 100.0 - layer * 3)
    degrade[31]["rank_j"] = 999.0  # lens degrades at the end
    checks += [
        (
            "last-layer regression fails the trend",
            depth_checks(degrade, 32)["iv_rank_improves_with_depth"] is False,
        ),
        (
            "flat profile fails the trend",
            depth_checks(_rows(lambda layer: 10.0), 32)["iv_rank_improves_with_depth"] is False,
        ),
    ]

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}: {name}")
    print(f"=== SUMMARY: PASS={len(checks) - len(failed)} FAIL={len(failed)} ===")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
