# SPDX-License-Identifier: Apache-2.0
"""Hold-out scoring: metrics must equal an independent brute-force reference."""

from __future__ import annotations

import pytest
import torch

from vlm_lens.evaluate import (
    format_scores,
    frequency_control,
    score_lens,
    unigram_counts,
)
from vlm_lens.models.tiny_llava import random_image
from vlm_lens.positions import build_position_masks
from vlm_lens.readout import lens_readout

from .conftest import IMAGE_TOKENS, PROMPT, TEXT_ONLY, TINY_CONFIG


def _brute_force_text_scores(model, lens, batch, *, layer: int, skip_first: int = 1):
    """Independent rank/agreement/KL computation for the text tag at one layer."""
    masks = build_position_masks(
        batch.input_ids, model.image_token_id, skip_first=skip_first, masks=("text",)
    )
    positions = [
        int(p)
        for p in masks["text"].nonzero(as_tuple=True)[0]
        if int(p) + 1 < batch.seq_len
        and int(batch.input_ids[0, p + 1]) != model.image_token_id
    ]
    readout = lens_readout(model, lens, batch, layers=[layer], positions=positions)
    lens_logits = readout.lens_logits[layer].double()
    model_logits = readout.model_logits.double()
    targets = readout.input_ids[[p + 1 for p in readout.positions]]

    ranks, model_ranks, agreements, kls = [], [], [], []
    for row_index, target in enumerate(targets.tolist()):
        lens_row, model_row = lens_logits[row_index], model_logits[row_index]
        lens_sorted = torch.sort(lens_row, descending=True).values
        ranks.append(int((lens_sorted > lens_row[target]).sum()) + 1)
        model_sorted = torch.sort(model_row, descending=True).values
        model_ranks.append(int((model_sorted > model_row[target]).sum()) + 1)
        agreements.append(
            int(torch.argmax(lens_row).item() == torch.argmax(model_row).item())
        )
        p_model = torch.softmax(model_row, dim=-1)
        p_lens = torch.softmax(lens_row, dim=-1)
        kls.append(float((p_model * (p_model.log() - p_lens.log())).sum()))

    n = len(ranks)
    return {
        "n": n,
        "mean_rank_true": sum(ranks) / n,
        "model_mean_rank_true": sum(model_ranks) / n,
        "top1_agreement": sum(agreements) / n,
        "mean_kl": sum(kls) / n,
    }


def test_score_lens_matches_brute_force(tiny_model, tiny_lenses, tiny_batch):
    scores = score_lens(tiny_model, tiny_lenses["text"], [tiny_batch], tags=("text",))
    by_layer = {score.layer: score for score in scores}
    assert set(by_layer) == {0, 1, tiny_model.n_layers - 1}

    for layer in (0, 1):
        expected = _brute_force_text_scores(
            tiny_model, tiny_lenses["text"], tiny_batch, layer=layer
        )
        score = by_layer[layer]
        assert score.n == expected["n"] > 0
        torch.testing.assert_close(
            score.mean_rank_true, expected["mean_rank_true"], rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            score.model_mean_rank_true, expected["model_mean_rank_true"], rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            score.top1_agreement, expected["top1_agreement"], rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(score.mean_kl, expected["mean_kl"], rtol=1e-4, atol=1e-6)


def test_final_layer_row_is_the_model_itself(tiny_model, tiny_lenses, tiny_batch):
    """No Jacobian exists at the final layer: its row must be the model's own logits."""
    final_layer = tiny_model.n_layers - 1
    scores = score_lens(tiny_model, tiny_lenses["text"], [tiny_batch], tags=("text",))
    row = next(score for score in scores if score.layer == final_layer)
    assert row.n > 0
    assert row.top1_agreement == 1.0
    assert row.mean_kl < 1e-9
    torch.testing.assert_close(
        row.mean_rank_true, row.model_mean_rank_true, rtol=0.0, atol=0.0
    )


def test_chunk_size_does_not_change_scores(tiny_model, tiny_lenses, tiny_batch):
    scores = score_lens(tiny_model, tiny_lenses["text"], [tiny_batch], tags=("text", "image"))
    chunked = score_lens(
        tiny_model, tiny_lenses["text"], [tiny_batch], tags=("text", "image"), chunk_size=1
    )
    assert [(score.layer, score.tag, score.n) for score in scores] == [
        (score.layer, score.tag, score.n) for score in chunked
    ]
    for coarse, fine in zip(scores, chunked, strict=True):
        torch.testing.assert_close(coarse.mean_rank_true, fine.mean_rank_true, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            coarse.model_mean_rank_true, fine.model_mean_rank_true, rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(coarse.top1_agreement, fine.top1_agreement, rtol=0.0, atol=1e-9)
        torch.testing.assert_close(coarse.mean_kl, fine.mean_kl, rtol=1e-4, atol=1e-9)


def test_text_only_sample_scores_text_only(tiny_model, tiny_lenses):
    scores = score_lens(tiny_model, tiny_lenses["text"], [TEXT_ONLY], tags=("text", "image"))
    assert scores
    assert {score.tag for score in scores} == {"text"}
    assert all(score.n > 0 for score in scores)


def test_image_placeholder_targets_are_excluded(tiny_model, tiny_lenses):
    """A text position whose next token is the image placeholder is not scorable."""
    batch = tiny_model.encode_mm(PROMPT, random_image(3, image_size=TINY_CONFIG.image_size))
    mask = build_position_masks(batch.input_ids, tiny_model.image_token_id, masks=("text",))["text"]
    candidates = [int(p) for p in mask.nonzero(as_tuple=True)[0] if int(p) + 1 < batch.seq_len]
    scorable = [
        p for p in candidates if int(batch.input_ids[0, p + 1]) != tiny_model.image_token_id
    ]
    assert len(candidates) == len(scorable) + 1  # the token before the image block

    scores = score_lens(tiny_model, tiny_lenses["text"], [batch], tags=("text",))
    assert all(score.n == len(scorable) for score in scores)


def test_format_scores_renders_every_row(tiny_model, tiny_lenses, tiny_batch):
    scores = score_lens(tiny_model, tiny_lenses["text"], [tiny_batch], tags=("text",))
    table = format_scores(scores)
    assert len(table.splitlines()) == len(scores) + 2  # header + separator
    assert table.splitlines()[0].split() == [
        "layer",
        "tag",
        "n",
        "rank_true",
        "rank_model",
        "top1_agree",
        "KL(model||lens)",
    ]


def test_include_placeholders_widens_the_image_tag(tiny_model, tiny_lenses, tiny_batch):
    """V5: the default exclusion scores one position per sample; including placeholders
    scores the whole block, and at the final layer that is a perfect model match."""
    default = score_lens(tiny_model, tiny_lenses["text"], [tiny_batch], tags=("image",), layers=[0])
    widened = score_lens(
        tiny_model,
        tiny_lenses["text"],
        [tiny_batch],
        tags=("image",),
        layers=[0],
        include_placeholders=True,
    )
    assert all(score.n == 1 for score in default)
    assert all(score.n == IMAGE_TOKENS for score in widened)
    final = [score for score in widened if score.layer == tiny_model.n_layers - 1][0]
    assert final.mean_kl == 0.0
    assert final.top1_agreement == 1.0


def test_unigram_counts_match_python_counting(tiny_model):
    texts = [TEXT_ONLY, PROMPT]
    tokenizer = tiny_model.tokenizer
    expected: dict[int, int] = {}
    for text in texts:
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        for token in encoded:
            expected[int(token)] = expected.get(int(token), 0) + 1
    vocab_size = int(tiny_model.unembed_weight().shape[0])
    counts = unigram_counts(tiny_model, texts, vocab_size=vocab_size)
    assert float(counts.sum()) == sum(expected.values())
    for token, count in expected.items():
        assert float(counts[token]) == count


def test_frequency_control_matches_explicit_arithmetic(tiny_model, tiny_lenses, tiny_text_batch):
    texts = [TEXT_ONLY, "The capital of France is Paris."]
    top_k = 3
    rows = frequency_control(
        tiny_model,
        tiny_lenses["text"],
        [tiny_text_batch],
        texts,
        layers=[0],
        tags=("text",),
        top_k=top_k,
    )
    assert rows and all(row.top_k == top_k and row.n > 0 for row in rows)

    vocab_size = int(tiny_model.unembed_weight().shape[0])
    counts = unigram_counts(tiny_model, texts, vocab_size=vocab_size)
    frequent = set(int(token) for token in torch.topk(counts, k=top_k).indices)
    masks = build_position_masks(
        tiny_text_batch.input_ids, tiny_model.image_token_id, skip_first=1, masks=("text",)
    )
    positions = [
        int(p)
        for p in masks["text"].nonzero(as_tuple=True)[0]
        if int(p) + 1 < tiny_text_batch.seq_len
    ]
    readout = lens_readout(tiny_model, tiny_lenses["text"], tiny_text_batch, layers=[0], positions=positions)
    targets = readout.input_ids[[p + 1 for p in readout.positions]]
    lens_top1 = readout.lens_logits[0].argmax(dim=1)
    expected_lens = sum(int(token) in frequent for token in lens_top1.tolist()) / len(positions)
    expected_true = sum(int(token) in frequent for token in targets.tolist()) / len(positions)
    row = next(row for row in rows if row.layer == 0)
    assert row.lens_top1_in_top_k == pytest.approx(expected_lens)
    assert row.true_in_top_k == pytest.approx(expected_true)
