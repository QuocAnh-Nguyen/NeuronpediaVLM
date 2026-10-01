# SPDX-License-Identifier: Apache-2.0
"""Synthetic corpus builder: deterministic manifests, real image files."""

from __future__ import annotations

import hashlib

import pytest

from vlm_lens.data.dummy import build_dummy_manifest, random_rgb_image
from vlm_lens.data.manifest import manifest_meta, read_manifest


def _file_hash(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_dummy_manifest_round_trip(tmp_path):
    manifest_path = build_dummy_manifest(tmp_path / "corpus", n_samples=3, image_size=56, seed=7)
    samples = read_manifest(manifest_path)
    assert len(samples) == 3
    assert manifest_meta(manifest_path)["corpus"] == "dummy"
    assert manifest_meta(manifest_path)["image_size"] == 56

    for index, sample in enumerate(samples):
        assert sample.sample_id == f"dummy-{index:03d}"
        assert sample.text == (
            f"USER: <image>\n{sample.meta['question']}\nASSISTANT:"
        )
        assert sample.meta["image_seed"] == 7 + index
        image_path = sample.images[0]
        assert image_path.endswith(".png")
        assert (tmp_path / "corpus" / "images" / f"dummy-{index:03d}.png").exists()

    questions = [sample.meta["question"] for sample in samples]
    assert len(set(questions)) == 3  # consecutive images cycle the question bank


def test_build_dummy_manifest_is_deterministic(tmp_path):
    first = build_dummy_manifest(tmp_path / "a", n_samples=2, image_size=56, seed=1)
    second = build_dummy_manifest(tmp_path / "b", n_samples=2, image_size=56, seed=1)
    assert [sample.text for sample in read_manifest(first)] == [
        sample.text for sample in read_manifest(second)
    ]
    assert _file_hash(tmp_path / "a" / "images" / "dummy-000.png") == _file_hash(
        tmp_path / "b" / "images" / "dummy-000.png"
    )

    other = build_dummy_manifest(tmp_path / "c", n_samples=2, image_size=56, seed=2)
    assert [sample.text for sample in read_manifest(other)] == [
        sample.text for sample in read_manifest(first)
    ]
    assert _file_hash(tmp_path / "a" / "images" / "dummy-000.png") != _file_hash(
        tmp_path / "c" / "images" / "dummy-000.png"
    )


def test_random_rgb_image_shape_and_dtype():
    image = random_rgb_image(0, image_size=32)
    assert image.shape == (32, 32, 3)
    assert image.dtype.name == "uint8"


@pytest.mark.parametrize(
    "kwargs",
    [{"n_samples": 0}, {"n_samples": 1, "questions": ()}],
)
def test_build_dummy_manifest_rejects_bad_args(tmp_path, kwargs):
    with pytest.raises(ValueError):
        build_dummy_manifest(tmp_path / "bad", image_size=56, **kwargs)
