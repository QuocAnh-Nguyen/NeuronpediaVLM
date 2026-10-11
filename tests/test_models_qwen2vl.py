# SPDX-License-Identifier: Apache-2.0
"""Qwen2VLLensModel adapter: placeholder guards, residual equivalence, position masks.

The equivalence invariant is the one everything else rests on: the pre-norm block
outputs the fit hooks record, and the logits read out of them, must equal HuggingFace's
own forward on identical inputs (bit-exact at tiny scale, as ``scripts/check_equivalence
--backend tiny`` demands). The image-token guard is the variable-resolution analog of
the LLaVA ``image_seq_length`` check: the tokenized image-token count must equal what
the processor's own ``image_grid_thw`` implies (``prod(grid) // merge**2`` per image),
never truncated.
"""

from __future__ import annotations

import pytest
import torch

from vlm_lens.fitting import model_fingerprint
from vlm_lens.models.qwen2vl import MultimodalBatch, Qwen2VLLensModel
from vlm_lens.models.tiny_qwen2vl import (
    IM_START_TOKEN,
    TinyQwen2VLConfig,
    build_tiny_qwen2vl,
    random_image,
)
from vlm_lens.positions import build_position_masks, mask_summary

TINY_CONFIG = TinyQwen2VLConfig()  # d_model 64, 4 layers, vocab 128; 56x56 -> 4 tokens
IMAGE_TOKENS = TINY_CONFIG.image_tokens_per_image  # 4
PROMPT = "USER: <image>\nDescribe this image.\nASSISTANT:"
TEXT_ONLY = "The capital of France is"


@pytest.fixture(scope="session")
def qwen_model() -> Qwen2VLLensModel:
    hf_model, processor = build_tiny_qwen2vl(TINY_CONFIG)
    return Qwen2VLLensModel(hf_model, processor)


@pytest.fixture()
def qwen_batch(qwen_model: Qwen2VLLensModel) -> MultimodalBatch:
    return qwen_model.encode_mm(PROMPT, random_image(0, image_size=TINY_CONFIG.image_size))


@pytest.fixture()
def qwen_text_batch(qwen_model: Qwen2VLLensModel) -> MultimodalBatch:
    return qwen_model.encode_mm(TEXT_ONLY, None)


# --------------------------------------------------------------------- encoding
def test_encode_mm_expands_image_tokens_to_grid(qwen_model, qwen_batch):
    batch = qwen_batch
    assert batch.image_grid_thw.tolist() == [[1, 4, 4]]
    assert batch.n_image_tokens == IMAGE_TOKENS
    assert batch.n_image_tokens == qwen_model.expected_image_tokens(batch.image_grid_thw)
    assert torch.equal(batch.image_token_mask, batch.input_ids == qwen_model.image_token_id)
    # production pixel layout: one row per patch, flattened as [C, temporal, ph, pw]
    patches = (TINY_CONFIG.image_size // TINY_CONFIG.patch_size) ** 2
    row_dim = 3 * 2 * TINY_CONFIG.patch_size**2
    assert tuple(batch.pixel_values.shape) == (patches, row_dim)
    assert batch.mm_token_type_ids.dtype == torch.long
    assert torch.equal(batch.mm_token_type_ids.bool(), batch.image_token_mask)


def test_encode_mm_is_deterministic(qwen_model, qwen_batch):
    again = qwen_model.encode_mm(PROMPT, random_image(0, image_size=TINY_CONFIG.image_size))
    assert torch.equal(again.input_ids, qwen_batch.input_ids)
    assert torch.equal(again.pixel_values, qwen_batch.pixel_values)


def test_merged_token_count_is_fixed_per_image_size(qwen_model):
    """56x56 -> 4 tokens, 336x336 -> 144: the LLaVA-style prompt works at both scales."""
    for image_size, expected in ((56, 4), (336, 144)):
        batch = qwen_model.encode_mm(PROMPT, random_image(1, image_size=image_size))
        assert batch.n_image_tokens == expected
        assert batch.image_grid_thw.tolist() == [[1, image_size // 14, image_size // 14]]
        assert batch.n_image_tokens == qwen_model.expected_image_tokens(batch.image_grid_thw)


def test_encode_mm_rejects_ungroupable_image_size(qwen_model):
    with pytest.raises(ValueError, match="multiples of"):
        qwen_model.encode_mm(PROMPT, random_image(2, image_size=100))


def test_placeholder_count_mismatch_raises(qwen_model):
    image = random_image(0, image_size=TINY_CONFIG.image_size)
    with pytest.raises(ValueError, match=r"1 <image> placeholders but 0 images"):
        qwen_model.encode_mm(PROMPT, None)
    with pytest.raises(ValueError, match=r"0 <image> placeholders but 1 images"):
        qwen_model.encode_mm(TEXT_ONLY, image)


def test_image_token_count_mismatch_raises(qwen_model):
    """A processor that expands to the wrong count is caught against the grid (fail-closed)."""
    processor = qwen_model.processor
    original = processor.spatial_merge_size
    processor.spatial_merge_size = 1  # expansion now emits 16 pads, the model expects 4
    try:
        with pytest.raises(ValueError, match="image tokens but .* were expected"):
            qwen_model.encode_mm(PROMPT, random_image(0, image_size=TINY_CONFIG.image_size))
    finally:
        processor.spatial_merge_size = original


def test_over_length_raises_instead_of_truncating(qwen_model, qwen_batch):
    with pytest.raises(ValueError, match="over the .*-token limit"):
        qwen_model.encode_mm(PROMPT, random_image(0, image_size=TINY_CONFIG.image_size),
                             max_length=qwen_batch.seq_len - 1)


def test_encode_decode_round_trip(qwen_model, qwen_batch):
    ids = qwen_batch.input_ids[0]
    tokens = qwen_model.processor.convert_ids_to_tokens(ids)
    # every image token survives decoding as the pad marker
    assert tokens.count("<|image_pad|>") == IMAGE_TOKENS
    # the generation decode path (skip_special_tokens=True) drops every special string
    decoded = qwen_model._decode(ids)
    assert IM_START_TOKEN not in decoded
    assert "<|image_pad|>" not in decoded
    non_special = [t for t in tokens if not t.startswith("<|") and not t.endswith("|>")]
    assert decoded.split(" ") == non_special


def test_load_image_resolves_paths_to_pil(qwen_model, tmp_path):
    from PIL import Image

    path = tmp_path / "img.png"
    Image.fromarray(random_image(0, image_size=TINY_CONFIG.image_size)).save(path)
    loaded = qwen_model.load_image(path)
    assert loaded.mode == "RGB"
    # the path never reaches the processor: encoding via path == encoding via PIL image
    via_path = qwen_model.encode_mm(PROMPT, path)
    via_pil = qwen_model.encode_mm(PROMPT, loaded)
    assert torch.equal(via_path.input_ids, via_pil.input_ids)
    assert torch.equal(via_path.pixel_values, via_pil.pixel_values)


# --------------------------------------------------------------------- equivalence
def test_forward_residual_matches_hf_block_outputs(qwen_model, qwen_batch):
    """Pre-norm block outputs == HF's own hidden states; unembed(last) == HF logits.

    transformers 5.x exposes ``hidden_states`` as ``[embeddings, block0, .., block_{n-2},
    post-final-norm]`` — the LAST block's pre-norm output is not in the tuple (its slot
    holds the post-final-norm output), which is exactly why the fit records block
    outputs with hooks instead of trusting ``output_hidden_states``.
    """
    batch = qwen_batch
    with torch.no_grad():
        hf_out = qwen_model.hf_model(**batch.hf_kwargs(), use_cache=False, output_hidden_states=True)
        residual = qwen_model.forward_residual(batch)
    assert sorted(residual) == list(range(qwen_model.n_layers))
    for layer in range(qwen_model.n_layers - 1):
        # hidden_states[0] is the embedding; entry l+1 is block l's pre-norm output
        assert torch.equal(residual[layer], hf_out.hidden_states[layer + 1])
    last = qwen_model.n_layers - 1
    with torch.no_grad():
        ours_logits = qwen_model.unembed(residual[last])
        stack = qwen_model.forward_mm(batch)
    assert torch.equal(ours_logits, hf_out.logits)
    assert torch.equal(stack, hf_out.hidden_states[-1])
    # forward_mm is the post-final-norm stack output, NOT the lens tensor
    assert not torch.equal(stack, residual[last])

def test_forward_residual_text_only_matches_hf(qwen_model, qwen_text_batch):
    batch = qwen_text_batch
    with torch.no_grad():
        hf_out = qwen_model.hf_model(**batch.hf_kwargs(), use_cache=False)
        residual = qwen_model.forward_residual(batch, layers=[qwen_model.n_layers - 1])
    assert torch.equal(qwen_model.unembed(residual[qwen_model.n_layers - 1]), hf_out.logits)


def test_forward_residual_rejects_out_of_range_layers(qwen_model, qwen_batch):
    with pytest.raises(ValueError, match="out of range"):
        qwen_model.forward_residual(qwen_batch, layers=[qwen_model.n_layers])


def test_mrope_needs_mm_token_type_ids(qwen_model, qwen_batch):
    """hf_kwargs carries the modality labels because the model refuses M-RoPE without them."""
    kwargs = qwen_batch.hf_kwargs()
    del kwargs["mm_token_type_ids"]
    with pytest.raises(ValueError, match="mm_token_type_ids"):
        qwen_model.hf_model(**kwargs, use_cache=False)


# --------------------------------------------------------------------- batch plumbing
def test_multimodal_batch_expand_replicates_one_image(qwen_model, qwen_batch):
    expanded = qwen_batch.expand(4)
    n, seq = 4, qwen_batch.seq_len
    assert tuple(expanded.input_ids.shape) == (n, seq)
    assert tuple(expanded.image_token_mask.shape) == (n, seq)
    assert expanded.image_grid_thw.shape == (n, 3)
    assert expanded.image_grid_thw.unique(dim=0).shape[0] == 1
    patches = qwen_batch.pixel_values.shape[0]
    assert tuple(expanded.pixel_values.shape) == (n * patches, qwen_batch.pixel_values.shape[1])
    # rows are block-repeated (one image per sample), not interleaved
    assert torch.equal(expanded.pixel_values[patches : 2 * patches], qwen_batch.pixel_values)
    assert qwen_batch.expand(1) is qwen_batch
    # hf_kwargs carries everything the HF forward needs, replicated consistently
    kwargs = expanded.hf_kwargs()
    assert set(kwargs) == {
        "input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids"
    }
    with torch.no_grad():
        single = qwen_model.hf_model(**qwen_batch.hf_kwargs(), use_cache=False).logits
        batched = qwen_model.hf_model(**kwargs, use_cache=False).logits
    for row in range(n):
        assert torch.equal(batched[row], single[0])


def test_multimodal_batch_expand_rejects_multi_image(qwen_model):
    image_a = random_image(3, image_size=TINY_CONFIG.image_size)
    image_b = random_image(4, image_size=TINY_CONFIG.image_size)
    batch = qwen_model.encode_mm("<image> and <image>", [image_a, image_b])
    assert batch.n_image_tokens == 2 * IMAGE_TOKENS
    assert batch.image_grid_thw.shape == (2, 3)
    with pytest.raises(ValueError, match="one image per sample"):
        batch.expand(2)


# --------------------------------------------------------------------- masks
def test_position_masks_with_image_token_id(qwen_model, qwen_batch):
    batch = qwen_batch
    masks = build_position_masks(
        batch.input_ids,
        qwen_model.image_token_id,
        masks=("text", "image", "all", "image-q0", "image-q1", "image-q2", "image-q3"),
    )
    assert int(masks["image"].sum()) == IMAGE_TOKENS
    # skip_first=1 drops BOS, exclude_last drops the final position (no next-token target)
    assert int(masks["all"].sum()) == batch.seq_len - 2
    assert torch.equal(masks["text"], masks["all"] & ~masks["image"])
    quarters = torch.stack([masks[f"image-q{i}"] for i in range(4)])
    assert torch.equal(quarters.any(dim=0), masks["image"])  # quarters partition the image
    assert (quarters.sum(dim=1) == 1).all()  # 4 positions -> one per contiguous quarter
    summary = mask_summary(masks, input_ids=batch.input_ids, image_token_id=qwen_model.image_token_id)
    assert summary["image"]["n_positions"] == IMAGE_TOKENS
    assert summary["text"]["n_positions"] == int(masks["text"].sum())


def test_position_masks_text_only_have_no_image_positions(qwen_model, qwen_text_batch):
    masks = build_position_masks(qwen_text_batch.input_ids, qwen_model.image_token_id)
    assert int(masks["image"].sum()) == 0
    assert int(masks["text"].sum()) == qwen_text_batch.seq_len - 2


# --------------------------------------------------------------------- surfaces
def test_predict_caption_returns_decoded_new_tokens(qwen_model, qwen_batch):
    batch = qwen_batch
    caption = qwen_model.predict_caption(batch, max_new_tokens=4)
    assert caption != ""
    with torch.no_grad():
        generated = qwen_model.hf_model.generate(
            **batch.hf_kwargs(), max_new_tokens=4, do_sample=False, use_cache=True
        )
    assert generated.shape[1] == batch.seq_len + 4
    assert torch.equal(generated[0, : batch.seq_len], batch.input_ids[0])
    assert caption == qwen_model._decode(generated[0, batch.seq_len :])


def test_vision_fingerprint_is_plain_and_recorded(qwen_model):
    fingerprint = qwen_model.vision_fingerprint()
    assert fingerprint["spatial_merge_size"] == TINY_CONFIG.spatial_merge_size
    assert fingerprint["vision_patch_size"] == TINY_CONFIG.patch_size
    assert fingerprint["image_token_id"] == TINY_CONFIG.image_token_id
    assert all(
        value is None or isinstance(value, (int, str)) for value in fingerprint.values()
    )
    full = model_fingerprint(qwen_model)
    assert full["model_type"] == "qwen2_vl"
    assert full["d_model"] == TINY_CONFIG.d_model
    assert full["n_layers"] == TINY_CONFIG.n_layers
    assert full["spatial_merge_size"] == TINY_CONFIG.spatial_merge_size
