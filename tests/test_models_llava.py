# SPDX-License-Identifier: Apache-2.0
"""Adapter hygiene: fail-closed ``image_seq_length`` resolution and the vision fingerprint.

``config.json`` at the pinned checkpoint has no ``image_seq_length`` (F5), so the value is
either the installed ``transformers`` class default or a derivation from the vision
geometry. A zero used to make the placeholder guard inert (F6/V10); these tests pin the
fail-closed behaviour the fit's correctness depends on.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vlm_lens.fitting import model_fingerprint
from vlm_lens.models.llava import LlavaLensModel, resolve_image_seq_length
from vlm_lens.models.tiny_llava import random_image

from .conftest import IMAGE_TOKENS, PROMPT, TINY_CONFIG


def test_resolve_image_seq_length_prefers_config():
    config = SimpleNamespace(
        image_seq_length=576, vision_config=SimpleNamespace(image_size=336, patch_size=14)
    )
    assert resolve_image_seq_length(config) == (576, "config")


def test_resolve_image_seq_length_derives_from_vision_geometry():
    config = SimpleNamespace(
        image_seq_length=0, vision_config=SimpleNamespace(image_size=336, patch_size=14)
    )
    value, source = resolve_image_seq_length(config)
    assert value == 576
    assert source.startswith("derived")


def test_resolve_image_seq_length_fails_closed():
    with pytest.raises(ValueError, match="fail-open"):
        resolve_image_seq_length(SimpleNamespace(image_seq_length=None, vision_config=None))
    with pytest.raises(ValueError, match="fail-open"):
        resolve_image_seq_length(
            SimpleNamespace(
                image_seq_length=0, vision_config=SimpleNamespace(image_size=336, patch_size=0)
            )
        )


def test_tiny_model_fingerprint_records_vision_config(tiny_model):
    assert tiny_model.image_seq_length == IMAGE_TOKENS
    fingerprint = model_fingerprint(tiny_model)
    assert fingerprint["image_seq_length"] == IMAGE_TOKENS
    assert fingerprint["image_seq_length_source"]
    assert fingerprint["vision_image_size"] == TINY_CONFIG.image_size
    assert fingerprint["vision_patch_size"] == TINY_CONFIG.patch_size
    assert fingerprint["vision_feature_select_strategy"]


def test_placeholder_guard_cannot_be_skipped(tiny_model: LlavaLensModel):
    original = tiny_model.image_seq_length
    tiny_model.image_seq_length = 0
    try:
        with pytest.raises(ValueError, match="refusing to skip the placeholder guard"):
            tiny_model.encode_mm(PROMPT, random_image(0, image_size=TINY_CONFIG.image_size))
    finally:
        tiny_model.image_seq_length = original
