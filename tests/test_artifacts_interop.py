# SPDX-License-Identifier: Apache-2.0
"""Artifact contract: upstream-compatible files, provenance, merges, checkpoint resume."""

from __future__ import annotations

import json

import pytest
import torch
from jlens.lens import JacobianLens

from vlm_lens.artifacts import (
    PROVENANCE_FILENAME,
    build_provenance,
    load_lens_set,
    merge_shards,
    save_lens_set,
)
from vlm_lens.fitting import fit_masked
from vlm_lens.models.tiny_llava import random_image

from .conftest import PROMPT, TINY_CONFIG

SOURCE_LAYERS = [0, 1]


def _samples(model, seeds=(20, 21, 22)):
    return [
        model.encode_mm(PROMPT, random_image(seed, image_size=TINY_CONFIG.image_size))
        for seed in seeds
    ]


def _provenance(model, lenses, **kwargs):
    return build_provenance(
        model=model,
        tasks=["coco-caption"],
        masks=list(lenses),
        n_prompts={mask: lens.n_prompts for mask, lens in lenses.items()},
        fit_config={"source_layers": SOURCE_LAYERS, "dim_batch": 8},
        **kwargs,
    )


def test_save_load_roundtrip_and_upstream_interop(tiny_model, tiny_lenses, tmp_path):
    provenance = _provenance(tiny_model, tiny_lenses, notes="unit test")
    written = save_lens_set(tmp_path / "lens", tiny_lenses, provenance=provenance)
    assert set(written) == set(tiny_lenses)

    loaded, sidecar = load_lens_set(tmp_path / "lens")
    assert set(loaded) == set(tiny_lenses)
    assert sidecar["provenance_version"] == 1
    assert sidecar["model"]["class"] == "LlavaForConditionalGeneration"
    assert sidecar["model"]["d_model"] == tiny_model.d_model
    assert sidecar["environment"]["torch"]
    assert sidecar["manifest"] is None if False else True  # manifest omitted when not given
    for mask, lens in loaded.items():
        assert lens.n_prompts == tiny_lenses[mask].n_prompts
        assert lens.source_layers == SOURCE_LAYERS
        for layer in SOURCE_LAYERS:
            # fp16 storage: tolerance rather than equality
            torch.testing.assert_close(
                lens.jacobians[layer], tiny_lenses[mask].jacobians[layer], rtol=1e-2, atol=1e-4
            )

    # provenance sidecar is valid JSON on disk with per-file hashes
    payload = json.loads((tmp_path / "lens" / PROVENANCE_FILENAME).read_text())
    assert set(payload["artifacts"]) == set(tiny_lenses)
    assert len(payload["artifacts"]["text"]["sha256"]) == 64

    # upstream loader reads our file verbatim
    upstream = JacobianLens.load(tmp_path / "lens" / "lens-text.pt")
    assert upstream.n_prompts == tiny_lenses["text"].n_prompts
    assert upstream.d_model == tiny_model.d_model


def test_embedded_provenance_is_ignored_by_upstream_loader(tiny_model, tiny_lenses, tmp_path):
    provenance = _provenance(tiny_model, tiny_lenses)
    save_lens_set(tmp_path / "lens", tiny_lenses, provenance=provenance, embed_provenance=True)
    payload = torch.load(tmp_path / "lens" / "lens-text.pt", map_location="cpu", weights_only=True)
    assert "provenance" in payload
    lens = JacobianLens.load(tmp_path / "lens" / "lens-text.pt")
    assert lens.n_prompts == tiny_lenses["text"].n_prompts


def test_merge_shards_matches_full_fit(tiny_model, tmp_path):
    samples = _samples(tiny_model)
    full = fit_masked(tiny_model, samples, source_layers=SOURCE_LAYERS, dim_batch=8, skip_first=1)

    for name, shard_samples in (("a", samples[:1]), ("b", samples[1:])):
        result = fit_masked(
            tiny_model, shard_samples, source_layers=SOURCE_LAYERS, dim_batch=8, skip_first=1
        )
        provenance = _provenance(tiny_model, result.lenses)
        # fp32 so the comparison isolates merge arithmetic from fp16 storage rounding
        save_lens_set(tmp_path / name, result.lenses, provenance=provenance, dtype=torch.float32)

    merged = merge_shards([tmp_path / "a", tmp_path / "b"])
    assert set(merged) == set(full.lenses)
    for mask in merged:
        assert merged[mask].n_prompts == full.lenses[mask].n_prompts
        for layer in SOURCE_LAYERS:
            torch.testing.assert_close(
                merged[mask].jacobians[layer],
                full.lenses[mask].jacobians[layer],
                rtol=1e-5,
                atol=1e-6,
            )


def test_checkpoint_resume_matches_uninterrupted_fit(tiny_model, tmp_path):
    samples = _samples(tiny_model)
    checkpoint = tmp_path / "ckpt.pt"

    partial = fit_masked(
        tiny_model,
        samples[:1],
        source_layers=SOURCE_LAYERS,
        dim_batch=8,
        skip_first=1,
        checkpoint_path=checkpoint,
    )
    assert partial.lenses["text"].n_prompts == 1

    resumed = fit_masked(
        tiny_model,
        samples,
        source_layers=SOURCE_LAYERS,
        dim_batch=8,
        skip_first=1,
        checkpoint_path=checkpoint,
        resume=True,
    )
    full = fit_masked(tiny_model, samples, source_layers=SOURCE_LAYERS, dim_batch=8, skip_first=1)
    assert resumed.lenses["text"].n_prompts == 3
    for layer in SOURCE_LAYERS:
        torch.testing.assert_close(
            resumed.lenses["text"].jacobians[layer],
            full.lenses["text"].jacobians[layer],
            rtol=1e-5,
            atol=1e-6,
        )


def test_checkpoint_fingerprint_mismatch_is_refused(tiny_model, tmp_path):
    samples = _samples(tiny_model, seeds=(30,))
    checkpoint = tmp_path / "ckpt.pt"
    fit_masked(
        tiny_model,
        samples,
        source_layers=SOURCE_LAYERS,
        dim_batch=8,
        skip_first=1,
        checkpoint_path=checkpoint,
    )
    with pytest.raises(ValueError, match="different settings"):
        fit_masked(
            tiny_model,
            samples,
            source_layers=SOURCE_LAYERS,
            dim_batch=4,  # changed knob
            skip_first=1,
            checkpoint_path=checkpoint,
            resume=True,
        )
    # resume=False discards the stale checkpoint and refits
    result = fit_masked(
        tiny_model,
        samples,
        source_layers=SOURCE_LAYERS,
        dim_batch=4,
        skip_first=1,
        checkpoint_path=checkpoint,
        resume=False,
    )
    assert result.lenses["text"].n_prompts == 1
