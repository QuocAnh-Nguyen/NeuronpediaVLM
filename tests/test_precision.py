# SPDX-License-Identifier: Apache-2.0
"""Guard the fp32/TF32 switch (register D20 / experiment X6).

True-fp32 cuBLAS on Hopper runs on CUDA cores (~15x slower than bf16 tensor cores at the
LLaVA fit's shapes: measured ~10 min/sample versus ~4.6 min), so the fp32 comparison fit is
run with TF32. ``torch`` defaults ``allow_tf32`` to False and has been migrating the knob to
``fp32_precision``; a flag that silently no-ops on one of those APIs would show up as a slow
fit rather than a bug, which is exactly the failure mode this test prevents.
"""

from __future__ import annotations

import torch

from vlm_lens.fitting import configure_tf32


def _read_allow_tf32() -> bool:
    return bool(torch.backends.cuda.matmul.allow_tf32)


def test_configure_tf32_toggles_and_restores() -> None:
    original = _read_allow_tf32()
    try:
        assert configure_tf32(True)["allow_tf32"] is True
        assert _read_allow_tf32() is True
        assert configure_tf32(False)["allow_tf32"] is False
        assert _read_allow_tf32() is False
    finally:
        configure_tf32(original)
    assert _read_allow_tf32() is original
