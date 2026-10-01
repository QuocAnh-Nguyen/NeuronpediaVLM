# SPDX-License-Identifier: Apache-2.0
"""Estimator correctness: the vectorised, masked, batched fit vs a brute-force estimator.

The brute force is an independent implementation: batch size 1, one backward per output
dimension, cotangent at every valid target position, and the per-mask reduction written
out explicitly. Any disagreement means the fast path (``dim_batch`` replication, shared
cotangent, joint mask reduction) is wrong.
"""

from __future__ import annotations

import pytest
import torch
from jlens.hooks import ActivationRecorder

from vlm_lens.fitting import drop_stats, fit_masked, jacobian_for_sample
from vlm_lens.models.tiny_llava import random_image
from vlm_lens.positions import build_position_masks

SOURCE_LAYERS = [0, 1]


def brute_force_jacobians(
    model,
    batch,
    source_layers,
    *,
    target_layer: int,
    skip_first: int,
    masks=("all",),
) -> dict[str, dict[int, torch.Tensor]]:
    d_model = model.d_model
    position_masks = build_position_masks(
        batch.input_ids, model.image_token_id, skip_first=skip_first, masks=masks
    )
    active = {name: mask for name, mask in position_masks.items() if bool(mask.any())}
    targets = build_position_masks(
        batch.input_ids, model.image_token_id, skip_first=skip_first, masks=("all",)
    )["all"].nonzero(as_tuple=True)[0]

    out = {
        name: {layer: torch.zeros(d_model, d_model) for layer in source_layers}
        for name in active
    }
    with (
        ActivationRecorder(
            model.layers, at=[*source_layers, target_layer], start_graph_at=min(source_layers)
        ) as recorder,
        torch.enable_grad(),
    ):
        model.forward_mm(batch)
        target = recorder.activations[target_layer]
        sources = [recorder.activations[layer] for layer in source_layers]
        for dim in range(d_model):
            cotangent = torch.zeros_like(target)
            cotangent[0, targets, dim] = 1.0
            grads = torch.autograd.grad(
                target, sources, grad_outputs=cotangent, retain_graph=(dim < d_model - 1)
            )
            for layer, grad in zip(source_layers, grads, strict=True):
                per_position = grad[0].float()  # [seq_len, d_model]
                for name, mask in active.items():
                    positions = mask.nonzero(as_tuple=True)[0]
                    out[name][layer][dim, :] = per_position[positions].mean(dim=0)
    return out


def test_matches_brute_force_multimodal(tiny_model, tiny_batch):
    estimated, info = jacobian_for_sample(
        tiny_model, tiny_batch, SOURCE_LAYERS, dim_batch=tiny_model.d_model, skip_first=1
    )
    expected = brute_force_jacobians(
        tiny_model,
        tiny_batch,
        SOURCE_LAYERS,
        target_layer=tiny_model.n_layers - 1,
        skip_first=1,
        masks=("text", "image", "all"),
    )
    assert set(estimated) == set(expected) == {"text", "image", "all"}
    assert info.mask_positions["image"] == int(tiny_batch.image_token_mask.sum())
    for name in expected:
        for layer in SOURCE_LAYERS:
            torch.testing.assert_close(
                estimated[name][layer], expected[name][layer], rtol=1e-5, atol=1e-6
            )


def test_dim_batch_sharding_is_invariant(tiny_model, tiny_batch):
    """Rows computed in several backward passes must equal the single-pass result."""
    one_pass, _ = jacobian_for_sample(
        tiny_model, tiny_batch, SOURCE_LAYERS, dim_batch=tiny_model.d_model, skip_first=1
    )
    many_passes, _ = jacobian_for_sample(tiny_model, tiny_batch, SOURCE_LAYERS, dim_batch=5, skip_first=1)
    for name in one_pass:
        for layer in SOURCE_LAYERS:
            torch.testing.assert_close(
                one_pass[name][layer], many_passes[name][layer], rtol=1e-5, atol=1e-6
            )


def test_subset_of_masks_is_consistent(tiny_model, tiny_batch):
    """Fitting only ``text`` must give the same matrix as the joint three-mask fit."""
    joint, _ = jacobian_for_sample(tiny_model, tiny_batch, SOURCE_LAYERS, dim_batch=8, skip_first=1)
    text_only, _ = jacobian_for_sample(
        tiny_model, tiny_batch, SOURCE_LAYERS, dim_batch=8, skip_first=1, masks=("text",)
    )
    assert set(text_only) == {"text"}
    torch.testing.assert_close(joint["text"][0], text_only["text"][0], rtol=0, atol=0)


def test_text_only_sample_has_no_image_contributions(tiny_model, tiny_text_batch):
    estimated, info = jacobian_for_sample(
        tiny_model, tiny_text_batch, SOURCE_LAYERS, dim_batch=8, skip_first=1
    )
    assert "image" not in estimated
    assert set(estimated) == {"text", "all"}
    torch.testing.assert_close(estimated["text"][0], estimated["all"][0], rtol=0, atol=0)
    assert info.n_image_tokens == 0
    assert "image" not in info.mask_positions


def test_skip_first_changes_positions(tiny_model, tiny_batch):
    first, _ = jacobian_for_sample(tiny_model, tiny_batch, SOURCE_LAYERS, dim_batch=8, skip_first=1)
    later, _ = jacobian_for_sample(tiny_model, tiny_batch, SOURCE_LAYERS, dim_batch=8, skip_first=4)
    assert not torch.allclose(first["all"][0], later["all"][0])
    expected = brute_force_jacobians(
        tiny_model,
        tiny_batch,
        SOURCE_LAYERS,
        target_layer=tiny_model.n_layers - 1,
        skip_first=4,
        masks=("all",),
    )
    torch.testing.assert_close(later["all"][0], expected["all"][0], rtol=1e-5, atol=1e-6)


def test_text_only_sample_matches_brute_force(tiny_model, tiny_text_batch):
    estimated, _ = jacobian_for_sample(
        tiny_model, tiny_text_batch, SOURCE_LAYERS, dim_batch=8, skip_first=1
    )
    expected = brute_force_jacobians(
        tiny_model,
        tiny_text_batch,
        SOURCE_LAYERS,
        target_layer=tiny_model.n_layers - 1,
        skip_first=1,
        masks=("text", "all"),
    )
    for name in expected:
        torch.testing.assert_close(estimated[name][1], expected[name][1], rtol=1e-5, atol=1e-6)


def test_too_short_sample_raises(tiny_model):
    with pytest.raises(ValueError, match="no valid (source|target) positions"):
        jacobian_for_sample(tiny_model, "hi", SOURCE_LAYERS, skip_first=16)


def test_multiple_images_supported(tiny_model):
    prompt = "USER: <image> and <image>\nCompare.\nASSISTANT:"
    batch = tiny_model.encode_mm(
        prompt, [random_image(3, image_size=56), random_image(4, image_size=56)]
    )
    assert batch.n_image_tokens == 2 * 16
    # Batch replication is not expressible for multi-image samples; dim_batch=1 is.
    with pytest.raises(ValueError, match="one image per sample"):
        jacobian_for_sample(tiny_model, batch, SOURCE_LAYERS, dim_batch=8, skip_first=1)
    estimated, info = jacobian_for_sample(
        tiny_model, batch, SOURCE_LAYERS, dim_batch=1, skip_first=1
    )
    assert info.mask_positions["image"] == 32
    assert set(estimated) == {"text", "image", "all"}


def test_target_mask_text_keeps_causally_disjoint_rows_identical(tiny_model):
    """X1: with the image block first, later text sources cannot causally reach any
    placeholder target, so their rows must be bit-identical under either target set."""
    batch = tiny_model.encode_mm("<image> A red car beside a blue bike.", random_image(3, image_size=56))
    common = dict(source_layers=SOURCE_LAYERS, dim_batch=8, skip_first=1, masks=("text",))
    all_targets, info_all = jacobian_for_sample(tiny_model, batch, target_mask="all", **common)
    text_targets, info_text = jacobian_for_sample(tiny_model, batch, target_mask="text", **common)
    assert info_all.target_mask == "all" and info_text.target_mask == "text"
    assert info_all.target_positions > info_text.target_positions > 0
    for layer in SOURCE_LAYERS:
        torch.testing.assert_close(
            all_targets["text"][layer], text_targets["text"][layer], rtol=0, atol=0
        )


def test_target_mask_changes_image_rows(tiny_model, tiny_batch):
    """Dropping patch-continuation targets is a structural change for image sources."""
    all_rows, _ = jacobian_for_sample(tiny_model, tiny_batch, SOURCE_LAYERS, dim_batch=8)
    text_rows, info = jacobian_for_sample(
        tiny_model, tiny_batch, SOURCE_LAYERS, dim_batch=8, target_mask="text"
    )
    assert info.target_positions < 592  # the 576 placeholder targets are gone
    image = all_rows["image"][1]
    relative = float((image - text_rows["image"][1]).norm() / image.norm())
    assert relative > 0.05


def test_checkpoint_fingerprint_covers_target_mask(tiny_model, tmp_path):
    samples = [
        tiny_model.encode_mm("<image> a dog", random_image(seed, image_size=56)) for seed in (20, 21)
    ]
    kwargs = dict(source_layers=[0], dim_batch=8, masks=("text",), checkpoint_path=tmp_path / "ckpt.pt")
    fit_masked(tiny_model, samples, target_mask="all", **kwargs)
    with pytest.raises(ValueError, match="different settings"):
        fit_masked(tiny_model, samples, target_mask="text", **kwargs)


def test_drop_stats_reports_rate_and_examples():
    skipped = [f"sample-{index}: over length" for index in range(11)]
    stats = drop_stats(8, skipped)
    assert stats["n_samples_seen"] == 8
    assert stats["n_skipped"] == 11
    assert stats["drop_rate"] == pytest.approx(11 / 8)
    assert stats["skipped_examples"] == skipped[:10]
    assert drop_stats(0, [])["drop_rate"] == 0.0
