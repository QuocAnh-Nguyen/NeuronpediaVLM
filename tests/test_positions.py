# SPDX-License-Identifier: Apache-2.0
"""Position-mask semantics: modality split, sink skipping, last-position exclusion."""

from __future__ import annotations

import pytest
import torch

from vlm_lens.positions import (
    ALL_MASKS,
    DEFAULT_MASKS,
    IMAGE_QUARTER_MASKS,
    build_position_masks,
    mask_summary,
)

from .conftest import IMAGE_TOKENS


def test_masks_split_modalities(tiny_batch):
    masks = build_position_masks(tiny_batch.input_ids, 50, skip_first=1, masks=DEFAULT_MASKS)
    n_image = int(masks["image"].sum())
    assert n_image == IMAGE_TOKENS
    seq_len = tiny_batch.seq_len
    assert int(masks["all"].sum()) == seq_len - 2  # minus BOS and the final token
    assert int(masks["text"].sum()) == seq_len - 2 - IMAGE_TOKENS
    # text and image partition all; image positions are contiguous (the placeholder block)
    assert torch.equal(masks["text"] | masks["image"], masks["all"])
    assert not bool((masks["text"] & masks["image"]).any())
    image_positions = masks["image"].nonzero(as_tuple=True)[0]
    assert torch.equal(image_positions, torch.arange(image_positions[0], image_positions[0] + IMAGE_TOKENS))


def test_last_position_excluded_and_skip_first(tiny_text_batch):
    seq_len = tiny_text_batch.seq_len
    masks = build_position_masks(tiny_text_batch.input_ids, 50, skip_first=2)
    assert not bool(masks["all"][-1])
    assert not bool(masks["all"][0])
    assert not bool(masks["all"][1])
    assert bool(masks["all"][2])
    assert int(masks["all"].sum()) == seq_len - 3
    # text-only sample: the image mask exists but is empty
    assert int(masks["image"].sum()) == 0
    assert torch.equal(masks["text"], masks["all"])


def test_exclude_last_switch(tiny_text_batch):
    masks = build_position_masks(tiny_text_batch.input_ids, 50, skip_first=1, exclude_last=False)
    assert bool(masks["all"][-1])
    assert int(masks["all"].sum()) == tiny_text_batch.seq_len - 1


def test_invalid_inputs(tiny_text_batch):
    with pytest.raises(ValueError, match="unknown masks"):
        build_position_masks(tiny_text_batch.input_ids, 50, masks=("vision",))
    with pytest.raises(ValueError, match="skip_first"):
        build_position_masks(tiny_text_batch.input_ids, 50, skip_first=-1)
    with pytest.raises(ValueError, match="single sequence"):
        build_position_masks(torch.zeros(2, 5, dtype=torch.long), 50)


def test_mask_summary_fractions(tiny_batch):
    masks = build_position_masks(tiny_batch.input_ids, 50, skip_first=1)
    summary = mask_summary(masks, input_ids=tiny_batch.input_ids, image_token_id=50)
    assert summary["image"]["n_positions"] == IMAGE_TOKENS
    assert 0 < summary["text"]["frac_of_seq"] < 1
    assert summary["all"]["n_positions"] == summary["text"]["n_positions"] + summary["image"]["n_positions"]


def test_image_quarters_partition_the_block(tiny_batch):
    """X2: quarters are contiguous, disjoint, equally populated, and union to the block."""
    assert set(IMAGE_QUARTER_MASKS) <= set(ALL_MASKS)
    masks = build_position_masks(
        tiny_batch.input_ids, 50, skip_first=1, masks=IMAGE_QUARTER_MASKS
    )
    quarters = [masks[f"image-q{index}"] for index in range(4)]
    combined = quarters[0].clone()
    for quarter in quarters[1:]:
        assert not bool((combined & quarter).any())
        combined |= quarter
    image = build_position_masks(tiny_batch.input_ids, 50, skip_first=1, masks=("image",))[
        "image"
    ]
    assert torch.equal(combined, image)
    assert all(int(quarter.sum()) == IMAGE_TOKENS // 4 for quarter in quarters)
    # block order: quarters partition the block left to right
    firsts = [int(quarter.nonzero(as_tuple=True)[0][0]) for quarter in quarters]
    assert firsts == sorted(firsts)


def test_quarters_split_the_valid_positions_only(tiny_batch):
    """Skipping leading patches must not skew the quarters against each other."""
    masks = build_position_masks(
        tiny_batch.input_ids, 50, skip_first=4, masks=IMAGE_QUARTER_MASKS + ("image",)
    )
    counts = sorted(int(masks[f"image-q{index}"].sum()) for index in range(4))
    assert sum(counts) == int(masks["image"].sum()) < IMAGE_TOKENS
    assert max(counts) - min(counts) <= 1
