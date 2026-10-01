# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures: a tiny random-weight LLaVA, samples, and a fitted lens set.

Dimensions are miniature (d_model=16, 3 blocks, 16 image tokens) so a full Jacobian fit
runs in seconds on CPU; the image block is still expanded through the real placeholder
path, which is what the position masks depend on.
"""

from __future__ import annotations

import numpy as np
import pytest

from vlm_lens.models.llava import LlavaLensModel
from vlm_lens.models.tiny_llava import TinyLlavaConfig, build_tiny_llava, random_image

TINY_CONFIG = TinyLlavaConfig(
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

IMAGE_TOKENS = (TINY_CONFIG.image_size // TINY_CONFIG.patch_size) ** 2  # 16
PROMPT = "USER: <image>\nDescribe this image.\nASSISTANT:"
TEXT_ONLY = "The capital of France is"


@pytest.fixture(scope="session")
def tiny_model() -> LlavaLensModel:
    model, processor = build_tiny_llava(TINY_CONFIG)
    return LlavaLensModel(model, processor)


@pytest.fixture()
def tiny_batch(tiny_model: LlavaLensModel):
    return tiny_model.encode_mm(PROMPT, random_image(0, image_size=TINY_CONFIG.image_size))


@pytest.fixture()
def tiny_text_batch(tiny_model: LlavaLensModel):
    return tiny_model.encode_mm(TEXT_ONLY, None)


@pytest.fixture()
def image_array() -> np.ndarray:
    return random_image(1, image_size=TINY_CONFIG.image_size)


@pytest.fixture(scope="session")
def tiny_lenses(tiny_model: LlavaLensModel):
    """A fitted lens set (text/all masks) on two tiny random-image samples."""
    from vlm_lens.fitting import fit_masked

    samples = [
        tiny_model.encode_mm(PROMPT, random_image(seed, image_size=TINY_CONFIG.image_size))
        for seed in (10, 11)
    ]
    result = fit_masked(
        tiny_model,
        samples,
        source_layers=[0, 1],
        dim_batch=8,
        skip_first=1,
        masks=("text", "all"),
    )
    return result.lenses
