# SPDX-License-Identifier: Apache-2.0
"""COCO caption builder: manifest round-trip and the processor's argument contract.

``generate_captions`` hands images to the model's own processor, whose accepted types are
narrower than ours: ``pathlib.Path`` raises ``TypeError`` (transformers 5.x), and paths
or bytes are decoded through ``torchvision.io.decode_image``, which fails on a
torchvision built without libjpeg (the cluster env carries exactly that, as an editable
install from another project). Only a real processor exhibits either failure, so the
strict shim below pins the contract on the tiny fixture.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from vlm_lens.data.captions import (
    build_coco_caption_manifest,
    prompt_template_hash,
    prompt_text,
)
from vlm_lens.data.manifest import manifest_meta, read_manifest
from vlm_lens.models.llava import LlavaLensModel
from vlm_lens.models.tiny_llava import random_image

from .conftest import TINY_CONFIG


class _StrictProcessor:
    """Delegates to a processor, but rejects the image types HF's processor rejects."""

    def __init__(self, inner: Any, seen: list[list[type]]) -> None:
        self.inner = inner
        self.seen = seen

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        images = kwargs.get("images")
        if images is not None:
            batch = list(images) if isinstance(images, (list, tuple)) else [images]
            self.seen.append([type(image) for image in batch])
            for image in batch:
                if not isinstance(image, (str, bytes, Image.Image)):
                    raise TypeError(f"only a single or a list of entries is supported but got {type(image)}")
        return self.inner(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


@pytest.fixture()
def coco_dir(tmp_path: Path) -> Path:
    """Four real JPEGs with COCO-style names (the builder globs ``*.jpg``)."""
    images_dir = tmp_path / "val2014"
    images_dir.mkdir()
    for index in range(4):
        image = random_image(index, image_size=TINY_CONFIG.image_size)
        Image.fromarray(image).save(images_dir / f"COCO_val2014_{index:012d}.jpg")
    return images_dir


def test_caption_manifest_round_trip_and_processor_contract(
    tiny_model: LlavaLensModel, coco_dir: Path, tmp_path: Path
) -> None:
    seen: list[list[type]] = []
    original = tiny_model.processor
    tiny_model.processor = _StrictProcessor(original, seen)
    try:
        path = build_coco_caption_manifest(
            tmp_path / "manifest-coco.jsonl",
            images_dir=coco_dir,
            model=tiny_model,
            n_images=2,
            batch_size=2,
            max_new_tokens=4,
        )
    finally:
        tiny_model.processor = original

    samples = read_manifest(path)
    assert len(samples) == 2
    assert seen, "the processor was never called"
    assert all(kind is Image.Image for batch in seen for kind in batch), seen

    for sample in samples:
        assert Path(sample.images[0]).is_file()
        caption = sample.meta["caption"]
        assert isinstance(caption, str) and caption  # tiny model decodes to token names
        assert sample.meta["variant"] == "prompt+caption"
        assert sample.text == prompt_text(sample.meta["question"], caption)

    meta = manifest_meta(path)
    assert meta["corpus"] == "coco_val2014_captions"
    assert meta["images_dir"] == str(coco_dir)
    assert meta["n_images"] == 2
    assert meta["n_dropped_over_length"] == 0


def test_template_hash_and_reproducible_sample_ids(tiny_model, coco_dir: Path, tmp_path: Path):
    """V8/V12: the template is hashed into provenance and ids do not depend on the
    process-salted ``hash()``."""
    kwargs = dict(images_dir=coco_dir, model=tiny_model, n_images=2, batch_size=2, max_new_tokens=3)
    samples = read_manifest(build_coco_caption_manifest(tmp_path / "m1.jsonl", **kwargs))
    meta = manifest_meta(tmp_path / "m1.jsonl")
    assert meta["prompt_template"] == "USER: <image>\n{question}\nASSISTANT:"
    assert meta["prompt_template_sha256"] == prompt_template_hash()
    assert len(meta["prompt_template_sha256"]) == 64
    for sample in samples:
        slug = hashlib.sha256(sample.meta["question"].encode("utf-8")).hexdigest()[:8]
        assert sample.sample_id.endswith(slug)
    again = read_manifest(build_coco_caption_manifest(tmp_path / "m2.jsonl", **kwargs))
    assert [sample.sample_id for sample in again] == [sample.sample_id for sample in samples]


def test_image_list_owns_the_selection(tiny_model, coco_dir: Path, tmp_path: Path):
    """A caller-provided image list must be used verbatim and recorded in the header (D4)."""
    images = sorted(coco_dir.glob("*.jpg"))[1:3]
    kwargs = dict(images_dir=coco_dir, model=tiny_model, batch_size=2, max_new_tokens=3)
    path = build_coco_caption_manifest(tmp_path / "m.jsonl", image_list=images, **kwargs)
    samples = read_manifest(path)
    assert [sample.meta["image"] for sample in samples] == [image.name for image in images]
    meta = manifest_meta(path)
    assert meta["image_list_given"] is True
    assert meta["image_names"] == [image.name for image in images]
    with pytest.raises(FileNotFoundError, match="do not exist"):
        build_coco_caption_manifest(
            tmp_path / "bad.jsonl", image_list=[coco_dir / "missing.jpg"], **kwargs
        )
    with pytest.raises(ValueError, match="duplicate"):
        build_coco_caption_manifest(
            tmp_path / "dup.jsonl", image_list=[images[0], images[0]], **kwargs
        )
