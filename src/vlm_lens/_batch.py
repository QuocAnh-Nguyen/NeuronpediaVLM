# SPDX-License-Identifier: Apache-2.0
"""Shared sample resolution: ``FitSample`` / ``MultimodalBatch`` / plain text -> batch.

Every entry point that accepts "a sample" resolves it the same way; keeping the rule in
one place means a new sample kind (or a new ``encode_mm`` argument) cannot make the fit
loop, the readout and the edit path disagree about what a sample means.
"""

from __future__ import annotations

from vlm_lens.data.manifest import FitSample
from vlm_lens.models.llava import LlavaLensModel, MultimodalBatch


def as_batch(
    model: LlavaLensModel,
    sample: FitSample | MultimodalBatch | str,
    max_seq_len: int,
) -> MultimodalBatch:
    """Encode ``sample`` for ``model`` (pass-through for an already-encoded batch)."""
    if isinstance(sample, MultimodalBatch):
        return sample
    if isinstance(sample, str):
        return model.encode_mm(sample, None, max_length=max_seq_len)
    return model.encode_mm(sample.text, list(sample.images) or None, max_length=max_seq_len)
