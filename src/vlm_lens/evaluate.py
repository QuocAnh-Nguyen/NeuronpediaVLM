# SPDX-License-Identifier: Apache-2.0
"""Hold-out fidelity scoring for a fitted lens.

A fitted lens ``lens_l(h) = unembed(J_l @ h)`` is only useful if it preserves the model's
own next-token prediction: the transported residual must decode to the distribution the
LLM would have produced at that position. This module measures that, per layer and per
modality tag, against the model's logits on **held-out** samples (never the fitting
manifest — the estimator's average is fit to those positions):

``mean_rank_true``
    Mean 1-based rank of the true next token (``input_ids[p + 1]``) under the lens.
``model_mean_rank_true``
    The same under the model's own logits: the ceiling a perfect lens would match.
``top1_agreement``
    Fraction of positions where the lens argmax equals the model argmax.
``mean_kl``
    Mean ``KL(model || lens)`` in nats; 0 iff the lens reproduces the model exactly.

Tag semantics follow :mod:`vlm_lens.positions`: ``text`` positions are the paper-matched
source reduction, ``image`` positions test the fused patch states (their next token is
text, so the rank is still well defined), ``all`` mixes both. Positions whose next token
is an image placeholder are always excluded: predicting a placeholder is not a
verbalization target and would otherwise dilute the text scores at the image boundary.

The default layer set includes the final layer, where no Jacobian exists and the readout
degenerates to the model's own logits — a free sanity row (agreement 1.0, KL 0).

``include_placeholders`` reverses the placeholder exclusion (V5): patch positions are then
scored too, which turns the ``image`` row into a whole-block model-matching number
(descriptive only, V4). :func:`frequency_control` adds the unigram-prior controls used by
the S1 gate, so rank metrics cannot be carried by a handful of frequent tokens.

Example::

    lenses, _ = load_lens_set("runs/coco-100/artifacts")
    scores = score_lens(model, lenses["text"], held_out_samples)
    print(format_scores(scores))
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from jlens.lens import JacobianLens

from vlm_lens._batch import as_batch
from vlm_lens.data.manifest import FitSample
from vlm_lens.models.llava import LlavaLensModel, MultimodalBatch
from vlm_lens.positions import build_position_masks
from vlm_lens.readout import lens_readout


@dataclass(frozen=True)
class LensScore:
    """Fidelity metrics of one lens layer at one modality tag."""

    layer: int
    tag: str
    n: int
    mean_rank_true: float
    model_mean_rank_true: float
    top1_agreement: float
    mean_kl: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _rank_of_row(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """1-based rank of each target id in its logit row (ties count as better ranks)."""
    target_logit = logits.gather(1, targets[:, None]).squeeze(1)
    return (logits > target_logit[:, None]).sum(dim=1) + 1


def _empty_stats() -> dict[str, float]:
    return {"n": 0.0, "rank": 0.0, "model_rank": 0.0, "agree": 0.0, "kl": 0.0}


def _iter_scored_batches(
    model: LlavaLensModel,
    lens: JacobianLens,
    samples: Sequence[FitSample | MultimodalBatch | str],
    *,
    layers: Sequence[int],
    tags: Sequence[str],
    skip_first: int,
    max_seq_len: int,
    use_jacobian: bool,
    include_placeholders: bool,
    chunk_size: int,
) -> Iterator[tuple[str, torch.Tensor, dict[int, torch.Tensor], torch.Tensor]]:
    """Yield ``(tag, model_logits, {layer: lens_logits}, targets)`` per scored chunk.

    Shared by :func:`score_lens` and :func:`frequency_control` so both agree on exactly
    which positions are scored and how the lens logits are produced. At the final layer
    the model's own logits stand in for the (non-existent) Jacobian readout.
    """
    final_layer = model.n_layers - 1
    for sample in samples:
        batch = as_batch(model, sample, max_seq_len)
        masks = build_position_masks(
            batch.input_ids,
            model.image_token_id,
            skip_first=skip_first,
            masks=tags,
        )
        active = {tag: mask for tag, mask in masks.items() if bool(mask.any())}
        if not active:
            continue
        index_list = sorted(
            {int(p) for mask in active.values() for p in mask.nonzero(as_tuple=True)[0]}
        )
        readout = lens_readout(
            model,
            lens,
            batch,
            layers=[layer for layer in layers if not (use_jacobian and layer == final_layer)],
            positions=index_list,
            use_jacobian=use_jacobian,
            max_seq_len=max_seq_len,
        )
        row_of = {position: row for row, position in enumerate(readout.positions)}
        input_ids = readout.input_ids
        seq_len = input_ids.numel()

        for tag, mask in active.items():
            # Positions, not readout rows: the target of position p is input_ids[p + 1].
            positions = [
                int(p) for p in mask.nonzero(as_tuple=True)[0] if int(p) + 1 < seq_len
            ]
            if not include_placeholders:
                positions = [
                    p for p in positions if int(input_ids[p + 1]) != model.image_token_id
                ]
            if not positions:
                continue
            rows = torch.tensor([row_of[p] for p in positions], dtype=torch.long)
            targets = input_ids[torch.tensor([p + 1 for p in positions], dtype=torch.long)]
            for start in range(0, rows.numel(), chunk_size):
                chunk = rows[start : start + chunk_size]
                model_logits = readout.model_logits[chunk]
                lens_logits = {
                    layer: (
                        model_logits
                        if use_jacobian and layer == final_layer
                        else readout.lens_logits[layer][chunk]
                    )
                    for layer in layers
                }
                yield tag, model_logits, lens_logits, targets[start : start + chunk_size]


def score_lens(
    model: LlavaLensModel,
    lens: JacobianLens,
    samples: Sequence[FitSample | MultimodalBatch | str],
    *,
    layers: Sequence[int] | None = None,
    tags: Sequence[str] = ("text", "image"),
    skip_first: int = 1,
    max_seq_len: int = 1536,
    use_jacobian: bool = True,
    chunk_size: int = 256,
    include_placeholders: bool = False,
) -> list[LensScore]:
    """Score ``lens`` on ``samples``; returns one :class:`LensScore` per (layer, tag).

    Args:
        layers: Layers to score; defaults to the fitted layers plus the final layer.
        tags: Position groups to score (``text`` / ``image`` / ``all`` / ``image-q0``...).
        skip_first: Leading positions excluded, as in the fit (attention sinks).
        use_jacobian: ``False`` scores the vanilla logit lens instead (baseline).
        chunk_size: Positions scored per tensor chunk (memory knob; results identical).
        include_placeholders: Score positions whose next token is an image placeholder too
            (V5). The default exclusion reduces the ``image`` tag to the last patch of each
            sample; including them measures whole-block model matching (descriptive only).

    Samples whose requested tags have no valid positions (e.g. a text-only sample with
    ``tags=("image",)``) contribute nothing.
    """
    final_layer = model.n_layers - 1
    score_layers = sorted(set(lens.source_layers if layers is None else layers) | {final_layer})
    stats: dict[int, dict[str, dict[str, float]]] = {
        layer: {tag: _empty_stats() for tag in tags} for layer in score_layers
    }

    for tag, model_chunk, lens_chunks, chunk_targets in _iter_scored_batches(
        model,
        lens,
        samples,
        layers=score_layers,
        tags=tags,
        skip_first=skip_first,
        max_seq_len=max_seq_len,
        use_jacobian=use_jacobian,
        include_placeholders=include_placeholders,
        chunk_size=chunk_size,
    ):
        model_rank = _rank_of_row(model_chunk, chunk_targets).float()
        model_top1 = model_chunk.argmax(dim=1)
        model_log_probs = F.log_softmax(model_chunk, dim=-1)
        for layer in score_layers:
            lens_chunk = lens_chunks[layer]
            accumulator = stats[layer][tag]
            accumulator["n"] += float(chunk_targets.numel())
            accumulator["rank"] += float(
                _rank_of_row(lens_chunk, chunk_targets).float().sum()
            )
            accumulator["model_rank"] += float(model_rank.sum())
            accumulator["agree"] += float(
                (lens_chunk.argmax(dim=1) == model_top1).float().sum()
            )
            accumulator["kl"] += float(
                F.kl_div(
                    F.log_softmax(lens_chunk, dim=-1),
                    model_log_probs,
                    reduction="none",
                    log_target=True,
                )
                .sum(dim=-1)
                .sum()
            )

    out: list[LensScore] = []
    for layer in score_layers:
        for tag in tags:
            accumulator = stats[layer][tag]
            n = int(accumulator["n"])
            if n == 0:
                continue
            out.append(
                LensScore(
                    layer=layer,
                    tag=tag,
                    n=n,
                    mean_rank_true=accumulator["rank"] / n,
                    model_mean_rank_true=accumulator["model_rank"] / n,
                    top1_agreement=accumulator["agree"] / n,
                    mean_kl=accumulator["kl"] / n,
                )
            )
    return out


@dataclass(frozen=True)
class FrequencyControl:
    """Token-frequency controls for one (layer, tag) cell of a score table (S1 gate).

    ``true_in_top_k`` is the fraction of scored positions whose true next token is among
    the ``top_k`` most frequent tokens of ``freq_texts``: the hit rate a "always guess a
    frequent token" strategy gets. ``lens_top1_in_top_k`` / ``model_top1_in_top_k`` say
    how often the lens / model actually predict such a token, and the ``unigram_*`` rows
    are the prior's own scores — the baseline the lens has to beat.
    """

    layer: int
    tag: str
    n: int
    top_k: int
    true_in_top_k: float
    lens_top1_in_top_k: float
    model_top1_in_top_k: float
    unigram_mean_rank_true: float
    lens_mean_rank_true: float
    model_mean_rank_true: float
    unigram_top1_agreement: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def unigram_counts(model: LlavaLensModel, texts: Sequence[str], *, vocab_size: int) -> torch.Tensor:
    """Token counts over ``texts``: the unigram baseline's sufficient statistic."""
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("model has no tokenizer; cannot build a unigram baseline")
    ids: list[int] = []
    for text in texts:
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        if hasattr(encoded, "tolist"):  # tokenizers may return tensors
            encoded = encoded.tolist()
        while encoded and isinstance(encoded[0], list):  # [1, n] batch shape
            encoded = encoded[0]
        ids.extend(int(token) for token in encoded)
    if not ids:
        raise ValueError("texts produced no tokens")
    counts = torch.bincount(torch.tensor(ids, dtype=torch.long), minlength=vocab_size)
    if counts.numel() > vocab_size:
        raise ValueError(
            f"tokenizer produced ids >= vocab_size={vocab_size}; wrong tokenizer pairing"
        )
    return counts.to(torch.float32)


def frequency_control(
    model: LlavaLensModel,
    lens: JacobianLens,
    samples: Sequence[FitSample | MultimodalBatch | str],
    freq_texts: Sequence[str],
    *,
    layers: Sequence[int] | None = None,
    tags: Sequence[str] = ("text",),
    top_k: int = 50,
    skip_first: int = 1,
    max_seq_len: int = 1536,
    include_placeholders: bool = False,
    chunk_size: int = 256,
) -> list[FrequencyControl]:
    """Rank metrics against a unigram prior, one row per (layer, tag) (S1 gate)."""
    final_layer = model.n_layers - 1
    score_layers = sorted(set(lens.source_layers if layers is None else layers) | {final_layer})
    vocab_size = int(model.unembed_weight().shape[0])
    counts = unigram_counts(model, freq_texts, vocab_size=vocab_size)
    log_unigram = (counts + 1.0).log_softmax(dim=-1)
    k = max(1, min(int(top_k), vocab_size))
    frequent = torch.zeros(vocab_size, dtype=torch.bool)
    frequent[torch.topk(counts, k=k).indices] = True
    unigram_argmax = int(counts.argmax())

    stats: dict[int, dict[str, dict[str, float]]] = {
        layer: {
            tag: {
                "n": 0.0,
                "true_freq": 0.0,
                "lens_freq": 0.0,
                "model_freq": 0.0,
                "unigram_rank": 0.0,
                "lens_rank": 0.0,
                "model_rank": 0.0,
                "unigram_agree": 0.0,
            }
            for tag in tags
        }
        for layer in score_layers
    }
    for tag, model_chunk, lens_chunks, chunk_targets in _iter_scored_batches(
        model,
        lens,
        samples,
        layers=score_layers,
        tags=tags,
        skip_first=skip_first,
        max_seq_len=max_seq_len,
        use_jacobian=True,
        include_placeholders=include_placeholders,
        chunk_size=chunk_size,
    ):
        n = int(chunk_targets.numel())
        if n == 0:
            continue
        model_top1 = model_chunk.argmax(dim=1)
        unigram_rank = _rank_of_row(
            log_unigram.unsqueeze(0).expand(n, -1), chunk_targets
        ).float()
        model_rank = _rank_of_row(model_chunk, chunk_targets).float()
        for layer in score_layers:
            lens_chunk = lens_chunks[layer]
            accumulator = stats[layer][tag]
            accumulator["n"] += float(n)
            accumulator["true_freq"] += float(frequent[chunk_targets].sum())
            accumulator["lens_freq"] += float(frequent[lens_chunk.argmax(dim=1)].sum())
            accumulator["model_freq"] += float(frequent[model_top1].sum())
            accumulator["unigram_rank"] += float(unigram_rank.sum())
            accumulator["lens_rank"] += float(
                _rank_of_row(lens_chunk, chunk_targets).float().sum()
            )
            accumulator["model_rank"] += float(model_rank.sum())
            accumulator["unigram_agree"] += float((model_top1 == unigram_argmax).sum())

    out: list[FrequencyControl] = []
    for layer in score_layers:
        for tag in tags:
            accumulator = stats[layer][tag]
            n = int(accumulator["n"])
            if n == 0:
                continue
            out.append(
                FrequencyControl(
                    layer=layer,
                    tag=tag,
                    n=n,
                    top_k=k,
                    true_in_top_k=accumulator["true_freq"] / n,
                    lens_top1_in_top_k=accumulator["lens_freq"] / n,
                    model_top1_in_top_k=accumulator["model_freq"] / n,
                    unigram_mean_rank_true=accumulator["unigram_rank"] / n,
                    lens_mean_rank_true=accumulator["lens_rank"] / n,
                    model_mean_rank_true=accumulator["model_rank"] / n,
                    unigram_top1_agreement=accumulator["unigram_agree"] / n,
                )
            )
    return out


def format_scores(scores: Sequence[LensScore], *, digits: int = 3) -> str:
    """Fixed-width table of :func:`score_lens` output (for CLIs and notebooks)."""
    header = (
        f"{'layer':>5}  {'tag':<5}  {'n':>6}  {'rank_true':>9}  "
        f"{'rank_model':>10}  {'top1_agree':>10}  {'KL(model||lens)':>15}"
    )
    lines = [header, "-" * len(header)]
    for score in scores:
        lines.append(
            f"{score.layer:>5}  {score.tag:<5}  {score.n:>6}  "
            f"{score.mean_rank_true:>9.{digits}f}  "
            f"{score.model_mean_rank_true:>10.{digits}f}  "
            f"{score.top1_agreement:>10.{digits}f}  {score.mean_kl:>15.{digits}f}"
        )
    return "\n".join(lines)


__all__ = [
    "FrequencyControl",
    "LensScore",
    "format_scores",
    "frequency_control",
    "score_lens",
    "unigram_counts",
]
