#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Regression test for the analysis scripts' check logic: gate check (iv) plus the
finite-difference check (ii) on the tiny CPU fixture (no GPU, no downloads).

Covers ``s1_score.depth_checks`` (the depth-trend rule that decides the S1 go/no-go, plus
its monotonicity diagnostics) and ``s1_score.finite_difference_check`` against the tiny
random-weight LLaVA model. Run from the repo root with the package importable:

    PYTHONPATH=src python results/validation_2026-10-01/code/test_analysis_logic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s1_score import build_gate, depth_checks, finite_difference_check  # noqa: E402


def finite_difference_case() -> list[tuple[str, bool]]:
    """Check (ii) must reproduce the estimator's *columns* on the tiny fixture.

    Three bug classes push the reported error to O(1) and are excluded here: perturbing
    several source dims in one pass (a row/column sum - the original server crash),
    comparing against the row ``J_l[i, :]`` of a transposed reading (the estimator stores
    output-dim-major), and any path that leaves the perturbation out of the graph. A
    correct check lands at numerical-noise level (measured 4.4e-04 at ``eps=1e-5``).
    """
    from vlm_lens.models.llava import LlavaLensModel
    from vlm_lens.models.tiny_llava import TinyLlavaConfig, build_tiny_llava, random_image

    config = TinyLlavaConfig(
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
    hf, processor = build_tiny_llava(config)
    model = LlavaLensModel(hf, processor)
    prompt = "USER: <image>\nDescribe this image.\nASSISTANT:"
    batch = model.encode_mm(prompt, random_image(0, image_size=config.image_size), max_length=128)
    out = finite_difference_check(
        model,
        [batch],
        layers=[0, 1],
        n_rows=3,
        max_seq_len=128,
        skip_first=1,
        eps=1e-5,  # the fixture's residual RMS is ~0.01; the real model uses 1e-2
    )
    worst = max(out["per_layer_mean"].values())
    return [
        (f"FD columns reproduce the estimator (worst {worst:.2e} <= 0.05)", worst <= 0.05),
        (f"FD worst at noise level ({worst:.2e} < 0.01)", worst < 0.01),
        (f"FD reports one entry per column ({len(out['per_col'])} == 6)", len(out["per_col"]) == 6),
    ]


def gate_case() -> list[tuple[str, bool]]:
    """The reported gate must tell *not measured* apart from *failed* (D20).

    With ``--skip-fd`` the deliverable JSON has no finite-difference rows; a gate that read
    the absent criterion as ``False`` would misreport a skipped check as a failure.
    """

    def report(*, fd: float | None, identity: float = 0.0, middle: bool = True) -> dict:
        out: dict = {
            "check_i_identity_max_abs_diff": identity,
            "checks": {
                "iii_last_layer_rank_diff": 0.0,
                "iv_rank_improves_with_depth": True,
                "j_not_worse_in_middle": middle,
            },
        }
        if fd is not None:
            out["check_ii_finite_difference"] = {"per_layer_mean": {"0": fd}}
        return out

    skipped = build_gate(report(fd=None))
    measured = build_gate(report(fd=0.01))
    noisy = build_gate(report(fd=0.5))
    middle_lost = build_gate(report(fd=0.01, middle=False))
    return [
        (
            "skipped FD is not_measured, not failed",
            skipped["not_measured"] == ["ii_finite_difference_ok"],
        ),
        ("skipped FD passes the measured criteria", skipped["PASS"] is True),
        ("skipped FD cannot read as complete", skipped["PASS_complete"] is False),
        (
            "measured FD at noise level passes and completes",
            (measured["PASS"], measured["PASS_complete"]) == (True, True),
        ),
        (
            "a real FD failure still fails the gate",
            (noisy["PASS"], noisy["PASS_complete"], noisy["not_measured"]) == (False, False, []),
        ),
        (
            "a middle-band loss fails the measured gate (S1's real verdict)",
            middle_lost["PASS"] is False and middle_lost["not_measured"] == [],
        ),
    ]


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
    checks += gate_case()
    checks += finite_difference_case()

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}: {name}")
    print(f"=== SUMMARY: PASS={len(checks) - len(failed)} FAIL={len(failed)} ===")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
