#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""X15: decoding-time hesitation — per-step model uncertainty vs hallucination outcomes.

[D44/H7] hypothesis: the model's own decoding-time uncertainty at the step that
COMPLETES a generated word predicts whether that word is hallucinated (matched COCO
category not annotated on the image) or grounded (category annotated). Training-free
detector; matches the LURE/HTDC "hesitation" findings.

For N COCO val2014 images (--images-dir/--annotations, x13's loading) the script
greedy-generates the caption ONCE per image while recording, at every generated step,
two readings of the same decoding position:

  - the MODEL's own next-token distribution: the raw lm_head logits ``generate``
    returns per step under ``output_scores`` (greedy: element i predicts generated
    token i, the same step convention the residual hooks enforce). From it: the
    full-vocab entropy, the top1-top2 margin (log space), and the log-prob + 1-based
    rank of the token actually chosen;
  - the CALIBRATED logit-lens distribution at the same position, per layer: x14's
    readout ``unembed(s_l * (h_l + b_l)) / temp + logit_bias`` over the identity
    lens, applied to the raw block-output residual the same hooks cache, with the
    entropy reduced on device (no full-vocab host copy).

The caption text, per-token char spans and the COCO matcher (whole-sequence decode,
longest-first + plural tolerance + span masking) are x14's FIXED machinery. Every
caption word gets a label: GROUNDED (matched category annotated on the image),
HALLUCINATED (matched category not annotated) or OTHER (no category matched), and is
mapped to its COMPLETION STEP — the decoding step whose token emits the word's final
character. The aggregates compare the two readings at grounded vs hallucinated
completion steps: bucket means for entropy/margin/logprob plus a tie-aware
Mann-Whitney AUROC (hallucinated = positive class, so >0.5 means the score runs
higher on hallucinated steps; the hesitation hypothesis predicts >0.5 for the
entropies and <0.5 for the margin) for model entropy, margin, and lens entropy per
layer.

Cost: one generate per image (prefill + max_new_tokens decode steps) plus
|layers| x max_new_tokens cheap matvec+unembeds for the lens entropies — never a
full forward per layer.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402, I001  # the vlm_lens import must precede jlens: it installs the path
import vlm_lens  # noqa: E402, F401

from jlens.lens import JacobianLens  # noqa: E402

from vlm_lens.artifacts import load_bias, load_lens_set  # noqa: E402
from vlm_lens.data.captions import (  # noqa: E402
    DEFAULT_QUESTIONS,
    list_coco_images,
    prompt_text,
    select_images,
)
from vlm_lens.models.llava import LlavaLensModel  # noqa: E402

DEFAULT_LENS_DIR = "/data/anhnq/vlm-lens-out/validation/s2-merged/artifacts"
DEFAULT_BIAS_DIR = "/data/anhnq/vlm-lens-out/validation/step5e"
DEFAULT_IMAGES_DIR = "/data/baodq/coco2014/val2014"
DEFAULT_ANNOTATIONS = "/data/baodq/coco2014/annotations/instances_val2014.json"

MATCHER_NOTE = (
    "x14's fixed machinery: the whole generated sequence is decoded ONCE with special "
    "tokens dropped (decoding per token and joining would strip every sentencepiece "
    "'▁' lead space and glue the words together, killing every word-boundary match), "
    "the text is lowercased, and the 80 COCO category names are matched longest-first "
    "with word boundaries, an optional (s|es) plural suffix tolerated, and span "
    "masking (each match is blanked with same-length spaces so offsets stay valid and "
    "'hot dog' does not also match 'dog'); each match is attributed to the DECODING "
    "STEP whose generated token emits the word's final character (prefix-decode "
    "offsets give each token's character span in the caption); caption words no "
    "category matched are kept with label 'other' (excluded from every aggregate); "
    "irregular plurals (person/people, mouse/mice) are a documented gap"
)
RANK_RULE = "1-based rank, ties count as better ranks (s2_eval._rank_of_row semantics)"
ENTROPY_NOTE = (
    "next-token Shannon entropy of the model's own distribution: softmax over the RAW "
    "lm_head logits HF generate returns per step (output_scores; argmax mismatches vs "
    "the chosen token are counted in meta.n_argmax_mismatch)"
)
MARGIN_NOTE = (
    "top1-top2 margin in log space: the log_softmax top-2 gap (equals the logit gap; "
    "higher = more decided)"
)
CHOSEN_NOTE = (
    "log-prob and 1-based rank of the ACTUALLY GENERATED token under the model's "
    "per-step distribution at its decoding step"
)
LENS_ENTROPY_NOTE = (
    "Shannon entropy of the CALIBRATED logit-lens softmax at the same decoding "
    "position, per layer: x14's _calibrated_lens_logits math (identity transport, + "
    "bias inside the fitted branch, * scale, unembed, / temp, + logit_bias; unfitted "
    "final layer raw) with the reduction computed on device instead of copying the "
    "full logits to the host"
)
AUROC_NOTE = (
    "tie-aware Mann-Whitney AUROC with HALLUCINATED as the positive class: >0.5 means "
    "the score runs HIGHER on hallucinated word-completion steps (hypothesis predicts "
    ">0.5 for the entropies, <0.5 for the margin)"
)


@dataclass(frozen=True)
class WordRecord:
    """One caption word: a COCO category mention (grounded/hallucinated) or other."""

    text: str  # the matched category phrase, or the word itself for "other"
    label: str  # "grounded" | "hallucinated" | "other"
    category: str | None
    category_id: int | None
    step: int  # raw generated position whose token emits the word's final character
    token_id: int  # the completing token's id
    piece: str  # the completing token's decoded piece (stripped)
    char_span: tuple[int, int]  # inclusive-exclusive span in the caption


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lens-dir", default=DEFAULT_LENS_DIR,
        help="the J source lens dir: only its fitted layer list and d_model are used "
        "(the transport itself is the in-script identity)",
    )
    parser.add_argument("--mask", default="text", help="which lens-<mask>.pt and bias-<mask>.pt to use")
    parser.add_argument(
        "--bias-dir", default=DEFAULT_BIAS_DIR,
        help="directory holding bias-<mask>.pt (the step5e moment payload: bias/scale, "
        "optional temp/logit_bias) applied inside every lens readout",
    )
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS, help="COCO instances_val2014.json")
    parser.add_argument("--n-images", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=0, help="image-selection seed")
    parser.add_argument("--question", default=DEFAULT_QUESTIONS[0], help="the captioning question")
    parser.add_argument(
        "--backend", choices=("hf-llava", "tiny"), default="hf-llava",
        help="model backend: 'hf-llava' (default, CUDA) or the tiny CPU smoke fixture",
    )
    parser.add_argument("--json", required=True)
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> LlavaLensModel:
    """The probe model: the HF checkpoint (default, CUDA) or the tiny CPU fixture."""
    if args.backend == "tiny":
        from vlm_lens.models.tiny_llava import TinyLlavaConfig, build_tiny_llava

        hf_model, processor = build_tiny_llava(TinyLlavaConfig())
        return LlavaLensModel(hf_model, processor)
    return LlavaLensModel.from_pretrained(
        dtype=torch.bfloat16, device="cuda", local_files_only=True
    )


def identity_lens(source: JacobianLens, device: torch.device) -> JacobianLens:
    """``torch.eye`` JacobianLens over the source lens's fitted layers (zero fitting).

    Built directly on ``device`` so ``transport``'s per-call ``.to`` is a no-op.
    """
    return JacobianLens(
        jacobians={
            layer: torch.eye(source.d_model, dtype=torch.float32, device=device)
            for layer in source.source_layers
        },
        n_prompts=source.n_prompts,
        d_model=source.d_model,
    )


def load_calibrated_params(
    bias_dir: str, mask: str
) -> tuple[dict[int, torch.Tensor], dict[int, float], dict[int, float], dict[int, torch.Tensor], dict[str, Any]]:
    """The bias-<mask>.pt payload as lens_readout's four dicts, plus the file's meta."""
    payload, meta = load_bias(Path(bias_dir) / f"bias-{mask}.pt")
    bias = {int(layer): entry["bias"] for layer, entry in payload.items()}
    scale = {int(layer): float(entry["scale"]) for layer, entry in payload.items()}
    temp = {
        int(layer): float(entry["temp"])
        for layer, entry in payload.items()
        if "temp" in entry
    }
    logit_bias = {
        int(layer): entry["logit_bias"]
        for layer, entry in payload.items()
        if "logit_bias" in entry
    }
    return bias, scale, temp, logit_bias, meta


def _calibrated_lens_entropy(
    model: LlavaLensModel,
    lens: JacobianLens,
    hidden: torch.Tensor,
    layer: int,
    bias: dict[int, torch.Tensor] | None,
    scale: dict[int, float] | None,
    temp: dict[int, float] | None,
    logit_bias: dict[int, torch.Tensor] | None,
) -> float:
    """x14 ``_calibrated_lens_logits``' per-layer math, reduced to the lens softmax's entropy.

    transport (+ bias inside the fitted branch) -> scale -> unembed -> temp ->
    logit_bias; an unfitted final layer is read raw with no corrections, exactly as in
    ``lens_readout``. With the identity lens this is the entropy of
    ``softmax(unembed(s_l * (h_l + b_l)))``. The reduction runs on the model device —
    the per-step/per-layer scan is |layers| x T readouts per image, so the full-vocab
    host copy x14's rank probe needs is skipped.
    """
    residual = hidden
    if layer in lens.jacobians:
        residual = lens.transport(residual, layer)
        if bias is not None and layer in bias:
            residual = residual + bias[layer].to(residual.device)
    if scale is not None and layer in scale:
        residual = residual * scale[layer]
    logits = model.unembed(residual).float()
    if temp is not None and layer in temp:
        logits = logits / temp[layer]
    if logit_bias is not None and layer in logit_bias:
        logits = logits + logit_bias[layer].to(logits.device)
    logprobs = logits.log_softmax(dim=-1)
    return float(-(logprobs.exp() * logprobs).sum())


@torch.no_grad()
def generate_with_residual_traces(
    model: LlavaLensModel,
    batch: Any,
    layers: list[int],
    *,
    max_new_tokens: int,
) -> tuple[torch.Tensor, list[dict[int, torch.Tensor]], list[torch.Tensor]]:
    """Greedy generation caching the RAW block-output residual per decoding position.

    Mirrors x14 (and ``trace_generation``)'s hook machinery and step convention (with
    KV caching, the prefill forward predicts token 0 at the last prompt position and
    each decode forward predicts one more token, so hook call i predicts generated
    token i) and additionally requests ``output_scores``: ``scores[i]`` is the model's
    raw lm-head logits predicting generated token i — the model-side distribution the
    uncertainty metrics read.
    """
    final_layer = model.n_layers - 1
    record_at = sorted(set(layers) | {final_layer})
    first_layer = record_at[0]

    step_records: list[dict[int, torch.Tensor]] = []

    def make_hook(layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = output if torch.is_tensor(output) else output[0]
            # Blocks fire in call order within one forward pass, so the first recorded
            # layer opening a new record is exactly where a new step starts.
            if layer == first_layer:
                step_records.append({})
            step_records[-1][layer] = tensor[0, -1].detach().float()

        return hook

    handles = [
        model.layers[layer].register_forward_hook(make_hook(layer)) for layer in record_at
    ]
    try:
        generated = model.hf_model.generate(
            **batch.hf_kwargs(),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            output_scores=True,
            return_dict_in_generate=True,
        )
    finally:
        for handle in handles:
            handle.remove()

    new_tokens = generated.sequences[0, batch.seq_len :].detach().cpu()
    scores = [step[0].detach().float() for step in generated.scores]
    if len(step_records) != len(new_tokens) or len(scores) != len(new_tokens):
        raise RuntimeError(
            f"hook recorded {len(step_records)} forwards / {len(scores)} score rows but "
            f"{len(new_tokens)} tokens were generated; the step alignment convention "
            "does not hold for this generate() configuration"
        )
    return new_tokens, step_records, scores


def step_metrics(scores: list[torch.Tensor], new_tokens: torch.Tensor) -> dict[str, list[Any]]:
    """Model-side per-step uncertainty from the raw per-step lm-head logits.

    One row per generated step: the full-vocab entropy of the model's next-token
    distribution (ENTROPY_NOTE), the top1-top2 margin in log space (MARGIN_NOTE), and
    the log-prob + 1-based rank of the actually generated token (CHOSEN_NOTE, rank per
    RANK_RULE). ``n_argmax_mismatch`` counts steps where the distribution's argmax is
    not the generated token — 0 for plain greedy; nonzero means generation-config
    processors shifted the choice, so the "chosen" readings describe the generated
    token's standing under the raw distribution.
    """
    logits = torch.stack([step for step in scores])  # [T, vocab] float32
    logprobs = logits.log_softmax(dim=-1)
    entropy = -(logprobs.exp() * logprobs).sum(dim=-1)
    top2 = logprobs.topk(2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    chosen = new_tokens.to(logits.device)
    chosen_logit = logits.gather(1, chosen[:, None]).squeeze(1)
    chosen_logprob = logprobs.gather(1, chosen[:, None]).squeeze(1)
    chosen_rank = (logits > chosen_logit[:, None]).sum(dim=1) + 1
    return {
        "entropy": [float(v) for v in entropy.tolist()],
        "margin": [float(v) for v in margin.tolist()],
        "logprob": [float(v) for v in chosen_logprob.tolist()],
        "rank": [int(v) for v in chosen_rank.tolist()],
        "n_argmax_mismatch": int((logits.argmax(dim=-1) != chosen).sum().item()),
    }


def lens_step_entropies(
    model: LlavaLensModel,
    lens: JacobianLens,
    step_records: list[dict[int, torch.Tensor]],
    layers: list[int],
    bias: dict[int, torch.Tensor] | None,
    scale: dict[int, float] | None,
    temp: dict[int, float] | None,
    logit_bias: dict[int, torch.Tensor] | None,
) -> dict[int, list[float]]:
    """Calibrated-lens softmax entropy at every decoding position, per layer."""
    return {
        layer: [
            _calibrated_lens_entropy(
                model, lens, record[layer], layer, bias, scale, temp, logit_bias
            )
            for record in step_records
        ]
        for layer in layers
    }


def load_coco_grounding(
    annotations_path: str,
) -> tuple[dict[str, int], dict[str, set[int]]]:
    """``name -> category_id`` and ``file_name -> set(category_id)`` from a COCO instances file."""
    with open(annotations_path, encoding="utf-8") as handle:
        data = json.load(handle)
    category_ids = {str(cat["name"]): int(cat["id"]) for cat in data["categories"]}
    file_to_id = {str(img["file_name"]): int(img["id"]) for img in data["images"]}
    per_image_id: dict[int, set[int]] = {}
    for ann in data["annotations"]:
        per_image_id.setdefault(int(ann["image_id"]), set()).add(int(ann["category_id"]))
    per_file = {name: per_image_id.get(img_id, set()) for name, img_id in file_to_id.items()}
    return category_ids, per_file


def _category_patterns(category_ids: dict[str, int]) -> list[tuple[str, int, re.Pattern[str]]]:
    """x13 ``detect_mentions`` matchers for lowercased text, longest name first."""
    patterns = []
    for name in sorted(category_ids, key=len, reverse=True):
        cleaned = name.strip().lower()
        pattern = re.compile(r"\b" + re.escape(cleaned) + r"(?:s|es)?\b")
        patterns.append((cleaned, category_ids[name], pattern))
    return patterns


def classify_generated_words(
    tokenizer: Any,
    new_token_ids: torch.Tensor,
    patterns: list[tuple[str, int, re.Pattern[str]]],
    present_category_ids: set[int],
) -> tuple[str, list[WordRecord]]:
    """Match generated tokens to COCO categories (see MATCHER_NOTE).

    x14's ``classify_generated_words`` machinery verbatim — ONE whole-sequence decode
    of the generated tokens with special tokens dropped, prefix-decode offsets giving
    each surviving token its character span, and the x13-style longest-first match
    with span masking — extended so the caption words NO category matched are also
    returned, labeled "other" (excluded from every aggregate).
    """
    special_ids = set(getattr(tokenizer, "all_special_ids", None) or [])
    token_ids: list[int] = []
    raw_index_of: list[int] = []  # filtered position -> raw generated position
    for raw_position, raw_id in enumerate(new_token_ids.tolist()):
        if int(raw_id) in special_ids:
            continue
        token_ids.append(int(raw_id))
        raw_index_of.append(raw_position)

    caption = tokenizer.decode(token_ids, skip_special_tokens=True)
    starts = [
        len(tokenizer.decode(token_ids[:end], skip_special_tokens=True))
        for end in range(len(token_ids))
    ]
    token_ends = starts[1:] + [len(caption)]  # exclusive char end of each token's span

    # x13 detect_mentions: longest first, plural tolerance, each match masks its span
    # (blanked with same-length spaces so later matches keep valid caption offsets).
    matches: list[tuple[int, int, str, int]] = []  # (char_start, char_end, name, id)
    masked = caption.lower()
    for name, category_id, pattern in patterns:
        found = list(pattern.finditer(masked))
        if not found:
            continue
        matches.extend((match.start(), match.end(), name, category_id) for match in found)
        masked = pattern.sub(lambda m: " " * (m.end() - m.start()), masked)

    def completing_step(char_end: int) -> int:
        # The token whose span contains the match's/word's LAST character completes
        # it; bisect_right lands on the LAST token with that start, so a zero-span
        # token (which owns no characters) is never chosen.
        index = bisect_right(starts, char_end - 1) - 1
        return raw_index_of[index]

    records: list[WordRecord] = []
    for char_start, char_end, name, category_id in sorted(matches):
        index = bisect_right(starts, char_end - 1) - 1
        records.append(
            WordRecord(
                text=name,
                label="grounded" if category_id in present_category_ids else "hallucinated",
                category=name,
                category_id=category_id,
                step=completing_step(char_end),
                token_id=token_ids[index],
                piece=caption[starts[index] : token_ends[index]].strip(),
                char_span=(char_start, char_end),
            )
        )
    for word in re.finditer(r"\w+", caption):
        word_start, word_end = word.start(), word.end()
        if any(word_start < match_end and match_start < word_end for match_start, match_end, _n, _cid in matches):
            continue  # inside a category match span (the match owns its label)
        index = bisect_right(starts, word_end - 1) - 1
        records.append(
            WordRecord(
                text=word.group(0),
                label="other",
                category=None,
                category_id=None,
                step=completing_step(word_end),
                token_id=token_ids[index],
                piece=caption[starts[index] : token_ends[index]].strip(),
                char_span=(word_start, word_end),
            )
        )
    records.sort(key=lambda record: record.char_span)
    return caption, records


def auroc(grounded_scores: Sequence[float], halluc_scores: Sequence[float]) -> float | None:
    """Tie-aware Mann-Whitney AUROC, HALLUCINATED as the positive class (AUROC_NOTE).

    ``P(halluc > grounded) + 0.5 * P(tie)`` computed from ascending 1-based average
    ranks; ``None`` when either bucket is empty.
    """
    n_pos, n_neg = len(halluc_scores), len(grounded_scores)
    if not n_pos or not n_neg:
        return None
    paired = sorted(
        zip(grounded_scores + list(halluc_scores), [0] * n_neg + [1] * n_pos, strict=True)
    )
    ranks = [0.0] * len(paired)
    i = 0
    while i < len(paired):
        j = i
        while j < len(paired) and paired[j][0] == paired[i][0]:
            j += 1
        average = (i + j + 1) / 2  # 1-based average rank of the tie group
        for k in range(i, j):
            ranks[k] = average
        i = j
    rank_sum_pos = sum(rank for rank, (_score, label) in zip(ranks, paired, strict=True) if label == 1)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def _mean_or_none(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _cell(value: float | None) -> str:
    """Digest cell: three decimals, or a dash when the bucket is empty."""
    return f"{value:>10.3f}" if value is not None else f"{'-':>10}"


def _round5(values: Sequence[float]) -> list[float]:
    return [round(value, 5) for value in values]


def _word_json(record: WordRecord) -> dict[str, Any]:
    return {
        "text": record.text,
        "label": record.label,
        "category": record.category,
        "category_id": record.category_id,
        "step": record.step,
        "token_id": record.token_id,
        "piece": record.piece,
        "char_span": list(record.char_span),
    }


def _bucket_summary(values: dict[str, list[float]]) -> dict[str, Any]:
    """Bucket means at word-completion steps (entropy/margin/logprob per the notes)."""
    return {
        "n_steps": len(values["entropy"]),
        "entropy_mean": _mean_or_none(values["entropy"]),
        "margin_mean": _mean_or_none(values["margin"]),
        "logprob_mean": _mean_or_none(values["logprob"]),
    }


def print_digest(
    layers: list[int], collected: dict[str, dict[str, Any]], n_other: int
) -> None:
    """The digest table: bucket means, model-side AUROC, per-layer lens AUROC."""
    n_grounded = len(collected["grounded"]["entropy"])
    n_halluc = len(collected["hallucinated"]["entropy"])
    print("\n== X15 hesitation: decoding-time uncertainty vs hallucination outcomes ==")
    print(
        f"matched words: {n_grounded} grounded, {n_halluc} hallucinated "
        f"(other caption words: {n_other}, informational only)"
    )
    if not n_grounded or not n_halluc:
        print("insufficient matched words for a separation (need both buckets non-empty)")
    print("\nbucket means at word-completion steps (model-side):")
    print(f"{'bucket':<13}{'n':>8}{'entropy':>11}{'margin':>11}{'logprob':>11}")
    for label in ("grounded", "hallucinated"):
        summary = _bucket_summary(collected[label])
        print(
            f"{label:<13}{summary['n_steps']:>8}"
            f"{_cell(summary['entropy_mean'])}{_cell(summary['margin_mean'])}"
            f"{_cell(summary['logprob_mean'])}"
        )
    print("\nAUROC (hallucinated = positive class; >0.5 = score higher on hallucinated):")
    print(
        f"  model_entropy {_cell(auroc(collected['grounded']['entropy'], collected['hallucinated']['entropy']))}"
    )
    print(
        f"  top1_margin   {_cell(auroc(collected['grounded']['margin'], collected['hallucinated']['margin']))}"
    )
    lens_aucs = {
        layer: auroc(
            collected["grounded"]["lens"][layer], collected["hallucinated"]["lens"][layer]
        )
        for layer in layers
    }
    print("  lens_entropy (per layer):")
    for layer in layers:
        print(f"    L{layer:<4}{_cell(lens_aucs[layer])}")
    scored = [(value, layer) for layer, value in lens_aucs.items() if value is not None]
    if scored:
        best = max(scored)
        print(f"  best lens separation: L{best[1]} (AUROC {best[0]:.3f})")


def main() -> int:
    args = parse_args()
    category_ids, per_file_categories = load_coco_grounding(args.annotations)
    patterns = _category_patterns(category_ids)

    source_lenses, _ = load_lens_set(args.lens_dir)
    if args.mask not in source_lenses:
        raise ValueError(f"mask {args.mask!r} not in {args.lens_dir} (found {sorted(source_lenses)})")
    source_lens = source_lenses[args.mask]
    if not source_lens.source_layers:
        raise ValueError(f"lens {args.mask!r} in {args.lens_dir} has no fitted layers")
    model = load_model(args)
    device = model.unembed_weight().device
    lens = identity_lens(source_lens, device)
    bias, scale, temp, logit_bias, bias_meta = load_calibrated_params(args.bias_dir, args.mask)
    bias = {layer: tensor.to(device) for layer, tensor in bias.items()} if bias else None
    logit_bias = (
        {layer: tensor.to(device) for layer, tensor in logit_bias.items()} if logit_bias else None
    )

    final_layer = model.n_layers - 1
    layers = sorted(set(lens.source_layers) | {final_layer})
    images = select_images(list_coco_images(args.images_dir), args.n_images, seed=args.seed)
    prompt = prompt_text(args.question)
    tokenizer = model.tokenizer

    print(
        f"model ready: layers={model.n_layers} d_model={model.d_model} backend={args.backend}; "
        f"identity lens over fitted layers {lens.source_layers[0]}..{lens.source_layers[-1]} "
        f"with bias-{args.mask}.pt from {args.bias_dir}"
    )
    #: matched word-completion-step readings per label (grounded=0 / hallucinated=1 in AUROC)
    collected: dict[str, dict[str, Any]] = {
        label: {"entropy": [], "margin": [], "logprob": [], "lens": {layer: [] for layer in layers}}
        for label in ("grounded", "hallucinated")
    }
    image_records: list[dict[str, Any]] = []
    n_generated = 0
    n_other_words = 0
    n_argmax_mismatch = 0
    n_skipped_no_grounding = 0

    for index, image_path in enumerate(images):
        present = per_file_categories.get(image_path.name)
        if not present:
            print(f"  [{index + 1}/{len(images)}] {image_path.name}: no annotations entry - skipped", flush=True)
            n_skipped_no_grounding += 1
            continue

        batch = model.encode_mm(prompt, image_path, max_length=args.max_seq_len)
        new_tokens, step_records, scores = generate_with_residual_traces(
            model, batch, layers, max_new_tokens=args.max_new_tokens
        )
        metrics = step_metrics(scores, new_tokens)
        n_argmax_mismatch += metrics.pop("n_argmax_mismatch")
        lens_entropies = lens_step_entropies(
            model, lens, step_records, layers, bias, scale, temp, logit_bias
        )
        caption, words = classify_generated_words(tokenizer, new_tokens, patterns, present)
        matched = [word for word in words if word.label != "other"]

        for word in matched:
            bucket = collected[word.label]
            bucket["entropy"].append(metrics["entropy"][word.step])
            bucket["margin"].append(metrics["margin"][word.step])
            bucket["logprob"].append(metrics["logprob"][word.step])
            for layer in layers:
                bucket["lens"][layer].append(lens_entropies[layer][word.step])

        n_grounded = sum(1 for word in matched if word.label == "grounded")
        n_generated += int(new_tokens.numel())
        n_other_words += len(words) - len(matched)
        image_records.append(
            {
                "image": image_path.name,
                "caption": caption,
                "n_generated": int(new_tokens.numel()),
                "n_words_matched": len(matched),
                "n_words_grounded": n_grounded,
                "n_words_hallucinated": len(matched) - n_grounded,
                "n_words_other": len(words) - len(matched),
                "steps": {
                    "token_ids": [int(token) for token in new_tokens.tolist()],
                    "model_entropy": _round5(metrics["entropy"]),
                    "top1_margin": _round5(metrics["margin"]),
                    "chosen_logprob": _round5(metrics["logprob"]),
                    "chosen_rank": metrics["rank"],
                    "lens_entropy": {
                        str(layer): _round5(values) for layer, values in lens_entropies.items()
                    },
                },
                "words": [_word_json(word) for word in words],
            }
        )
        print(
            f"  [{index + 1}/{len(images)}] {image_path.name}: {len(matched)} matched words "
            f"({n_grounded}g/{len(matched) - n_grounded}h, {len(words) - len(matched)} other) "
            f"{caption[:70]!r}",
            flush=True,
        )

    if n_argmax_mismatch:
        print(
            f"WARNING: {n_argmax_mismatch} steps where argmax(model logits) != generated "
            "token (generation-config processors?); chosen-token readings describe the "
            "generated token's standing under the raw distribution",
            flush=True,
        )

    print_digest(layers, collected, n_other_words)

    aggregate = {
        "n_words_grounded": len(collected["grounded"]["entropy"]),
        "n_words_hallucinated": len(collected["hallucinated"]["entropy"]),
        "n_words_other": n_other_words,
        "grounded": _bucket_summary(collected["grounded"]),
        "hallucinated": _bucket_summary(collected["hallucinated"]),
        "auroc": {
            "note": AUROC_NOTE,
            "model_entropy": auroc(
                collected["grounded"]["entropy"], collected["hallucinated"]["entropy"]
            ),
            "top1_margin": auroc(
                collected["grounded"]["margin"], collected["hallucinated"]["margin"]
            ),
            "lens_entropy": {
                str(layer): auroc(
                    collected["grounded"]["lens"][layer], collected["hallucinated"]["lens"][layer]
                )
                for layer in layers
            },
        },
    }
    per_layer = {
        str(layer): {
            "grounded_entropy_mean": _mean_or_none(collected["grounded"]["lens"][layer]),
            "halluc_entropy_mean": _mean_or_none(collected["hallucinated"]["lens"][layer]),
            "n_grounded": len(collected["grounded"]["lens"][layer]),
            "n_hallucinated": len(collected["hallucinated"]["lens"][layer]),
        }
        for layer in layers
    }

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "lens_dir": str(args.lens_dir),
        "mask": args.mask,
        "bias_dir": str(args.bias_dir),
        "identity_lens": {
            "note": "torch.eye per fitted layer; readout = unembed(s_l*(h_l+b_l)) + logit_bias_l",
            "d_model": lens.d_model,
            "layers": lens.source_layers,
            "n_prompts": int(lens.n_prompts),
        },
        "calibrated": {
            "bias_layers": sorted(bias) if bias else [],
            "scale_layers": sorted(scale) if scale else [],
            "temp_layers": sorted(temp) if temp else [],
            "logit_bias_layers": sorted(logit_bias) if logit_bias else [],
        },
        "bias_meta": bias_meta,
        "question": args.question,
        "prompt": prompt,
        "images_dir": str(args.images_dir),
        "annotations": str(args.annotations),
        "n_images": len(images),
        "n_images_skipped_no_annotations": n_skipped_no_grounding,
        "max_new_tokens": args.max_new_tokens,
        "max_seq_len": args.max_seq_len,
        "seed": args.seed,
        "n_tokens_generated": n_generated,
        "n_argmax_mismatch": n_argmax_mismatch,
        "matcher": MATCHER_NOTE,
        "metric_notes": {
            "entropy": ENTROPY_NOTE,
            "margin": MARGIN_NOTE,
            "chosen": CHOSEN_NOTE,
            "lens_entropy": LENS_ENTROPY_NOTE,
            "rank_rule": RANK_RULE,
        },
        # steps are indexed by RAW generated position (special tokens included), the
        # same convention as x14's token_index; word.step indexes these arrays.
        "step_indexing": "steps arrays are indexed by raw generated position; words' "
        "'step' field is the completion step (the token emitting the word's final char)",
    }
    report = {"meta": meta, "aggregate": aggregate, "per_layer": per_layer, "images": image_records}
    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
