#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""X14: the training-free workspace probe — how the calibrated lens forms the caption, layer by layer.

For N COCO images the script greedy-generates the caption once while recording, at every
generated step, the raw residual at the decoding position for every lens layer (one hook
pass per forward, mirroring ``trace_generation``'s step convention: hook call i predicts
generated token i). The calibrated readout then runs on the cached residuals — never a
full forward per layer — so the cost is N x ~max_new_tokens x |layers| cheap
matvec+unembeds.

Readout: the calibrated logit lens ``z_l = unembed(s_l * (h_l + b_l))`` (plus the
payload's optional ``logit_bias``/``temp``). The JacobianLens is built IN-SCRIPT as
``torch.eye`` per layer over --lens-dir's fitted layer list (zero fitting, identity
transport), and the affine payload comes from --bias-dir's ``bias-<mask>.pt`` (the
step5e moment census). The per-layer math mirrors ``lens_readout``'s
``bias``/``scale``/``temp``/``logit_bias`` handling exactly; an unfitted final layer
gets no corrections, as in ``lens_readout``.

Grounding comes from instances_val2014.json: every generated content word matched to one
of the 80 COCO categories (the x13 ``detect_mentions``-style matcher below) is GROUNDED
when the category is annotated on the image, HALLUCINATED otherwise. Per layer the probe
reports:

  - ``grounded_rank`` / ``halluc_rank``: mean rank of the generated word's true token id
    in the lens distribution at its decoding position (the word's lens trajectory);
  - ``true_object_rank_at_halluc_positions``: at hallucinated words' decoding positions,
    the best (lowest) rank among the image's ACTUAL categories — the "does the model
    know" test: values far below ``halluc_rank`` mean the true object was already
    readable in the same distribution the hallucinated word was drawn from;
  - ``last_patch_object_rank``: the workspace-formation curve — at the LAST image-token
    position, the mean best rank of the image's true categories, per layer;
  - ``handoff_true_object_rank_mean`` / ``_best`` (``--handoff-rank``): at the LAST
    PROMPT position — the prefill forward's decoding position, whose distribution
    predicts generated token 0 (the final ``ASSISTANT:`` prompt token, right before the
    first generated token) — the mean and best rank of the image's TRUE categories under
    the same calibrated lens. The prefill record IS ``step_records[0]``, so the
    measurement reuses the generation pass; the literature predicts objects ARE
    decodable at this text position even though they are not at image positions.
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
from typing import Any

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402, I001  # the vlm_lens import must precede jlens: it installs the path
import vlm_lens  # noqa: E402, F401

from jlens.hooks import ActivationRecorder  # noqa: E402
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
    "x13 detect_mentions-style matcher: the whole generated sequence is decoded ONCE "
    "with special tokens dropped (decoding per token and joining would strip every "
    "sentencepiece '▁' lead space and glue the words together, killing every "
    "word-boundary match), the text is lowercased, and the 80 COCO category names are "
    "matched longest-first with word boundaries, an optional (s|es) plural suffix "
    "tolerated, and span masking (each match is blanked with same-length spaces so "
    "offsets stay valid and 'hot dog' does not also match 'dog'); each occurrence is "
    "attributed to the DECODING STEP whose generated token emits the word's final "
    "character (prefix-decode offsets give each token's character span in the "
    "caption); irregular plurals (person/people, mouse/mice) are a documented gap"
)
HANDOFF_NOTE = (
    "at the LAST PROMPT position — the prefill forward's decoding position, whose "
    "distribution predicts generated token 0 (the final 'ASSISTANT:' prompt token, "
    "right before the first generated token) — the calibrated lens distribution is "
    "probed with the image's true category token ids (the last subword of ' ' + name); "
    "per image the best (lowest) rank among the image's categories is kept, then per "
    "layer the mean over images is handoff_true_object_rank_mean and the best over "
    "images is handoff_true_object_rank_best; step_records[0] IS this position's "
    "cached residual, so the measurement reuses the generation pass"
)
CATEGORY_TOKEN_RULE = (
    "an image category is probed through the LAST subword id of ' ' + name (the "
    "leading-space emission form, mirroring interventions._token_id)"
)
RANK_RULE = "1-based rank, ties count as better ranks (s2_eval._rank_of_row semantics)"


@dataclass(frozen=True)
class ClassifiedWord:
    """One generated token attributed to a COCO category by the simple matcher."""

    token_index: int  # index into the generated sequence == step_records step
    token_id: int
    piece: str
    category: str
    category_id: int
    grounded: bool


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
    parser.add_argument("--n-images", type=int, default=24)
    parser.add_argument("--question", default=DEFAULT_QUESTIONS[0], help="the captioning question")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument(
        "--backend", choices=("hf-llava", "tiny"), default="hf-llava",
        help="model backend: 'hf-llava' (default, CUDA) or the tiny CPU smoke fixture",
    )
    parser.add_argument(
        "--handoff-rank",
        action="store_true",
        help="also probe the image's true categories at the LAST PROMPT position (the "
        "prefill decoding position whose distribution predicts generated token 0, the "
        "final 'ASSISTANT:' token): per layer, handoff_true_object_rank_mean/_best",
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


def _calibrated_lens_logits(
    model: LlavaLensModel,
    lens: JacobianLens,
    hidden: torch.Tensor,
    layer: int,
    bias: dict[int, torch.Tensor] | None,
    scale: dict[int, float] | None,
    temp: dict[int, float] | None,
    logit_bias: dict[int, torch.Tensor] | None,
) -> torch.Tensor:
    """``lens_readout``'s per-layer math for one cached residual vector.

    transport (+ bias inside the fitted branch) -> scale -> unembed -> temp ->
    logit_bias; an unfitted final layer is read raw with no corrections, exactly as in
    ``lens_readout``. With the identity lens this is ``unembed(s_l * (h_l + b_l))``.
    """
    residual = hidden
    if layer in lens.jacobians:
        residual = lens.transport(residual, layer)
        if bias is not None and layer in bias:
            residual = residual + bias[layer].to(residual.device)
    if scale is not None and layer in scale:
        residual = residual * scale[layer]
    logits = model.unembed(residual).float().cpu()
    if temp is not None and layer in temp:
        logits = logits / temp[layer]
    if logit_bias is not None and layer in logit_bias:
        logits = logits + logit_bias[layer].to(logits.device)
    return logits


@torch.no_grad()
def generate_with_residual_traces(
    model: LlavaLensModel,
    batch: Any,
    layers: list[int],
    *,
    max_new_tokens: int,
) -> tuple[torch.Tensor, list[dict[int, torch.Tensor]]]:
    """Greedy generation caching the RAW block-output residual at each decoding position.

    Mirrors ``trace_generation``'s hook machinery and step convention (with KV caching,
    the prefill forward predicts token 0 at the last prompt position and each decode
    forward predicts one more token, so hook call i predicts generated token i) but
    stores the raw residual only — the calibrated per-layer readout runs afterwards on
    these cached vectors.
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
        )
    finally:
        for handle in handles:
            handle.remove()

    new_tokens = generated[0, batch.seq_len :].detach().cpu()
    if len(step_records) != len(new_tokens):
        raise RuntimeError(
            f"hook recorded {len(step_records)} forwards but {len(new_tokens)} tokens "
            "were generated; the step alignment convention does not hold for this "
            "generate() configuration"
        )
    return new_tokens, step_records


@torch.no_grad()
def last_patch_residuals(
    model: LlavaLensModel, batch: Any, layers: list[int]
) -> dict[int, torch.Tensor] | None:
    """Raw block-output residuals at the LAST image-token position, one forward pass."""
    mask = batch.image_token_mask[0]
    if not bool(mask.any()):
        return None
    last_patch = int(mask.nonzero(as_tuple=True)[0][-1])
    with ActivationRecorder(model.layers, at=layers) as recorder:
        model.forward_mm(batch)
        return {
            layer: recorder.activations[layer][0, last_patch].detach().float()
            for layer in layers
        }


def _rank_of_id(logits: torch.Tensor, token_id: int) -> int:
    """1-based rank of ``token_id`` in the logit row (ties count as better ranks)."""
    return int((logits > logits[token_id]).sum().item()) + 1


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
) -> tuple[str, list[ClassifiedWord]]:
    """Match generated tokens to COCO categories (see MATCHER_NOTE).

    Mirrors x13's ``detect_mentions``: the caption is ONE whole-sequence decode of the
    generated tokens with special tokens dropped (per-token decodes strip each
    sentencepiece '▁' lead and glue the words together, so ``\\b``-anchored patterns
    never match), matched longest-first on the lowercased text with span masking.
    Prefix-decode offsets give each surviving token its character span; a match is
    attributed to the token that emits its final character — the decoding step that
    completes the word.
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

    instances = []
    for _char_start, char_end, name, category_id in sorted(matches):
        # The token whose span contains the match's LAST character completes the word;
        # bisect_right lands on the LAST token with that start, so a zero-span token
        # (which owns no characters) is never chosen.
        index = bisect_right(starts, char_end - 1) - 1
        instances.append(
            ClassifiedWord(
                token_index=raw_index_of[index],  # steps are indexed by RAW generated position
                token_id=token_ids[index],
                piece=caption[starts[index] : token_ends[index]].strip(),
                category=name,
                category_id=category_id,
                grounded=category_id in present_category_ids,
            )
        )
    return caption, instances


def _category_token_ids(tokenizer: Any, names: list[str]) -> list[int]:
    """Last subword id of ' ' + name per category (see CATEGORY_TOKEN_RULE)."""
    ids = []
    for name in sorted(names):
        encoded = tokenizer(" " + name, add_special_tokens=False)["input_ids"]
        if torch.is_tensor(encoded):  # the tiny stand-in returns [1, n] tensors
            encoded = encoded[0].tolist()
        ids.append(int(encoded[-1]))
    return ids


def _mean_or_none(values: list[int]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _cell(value: float | None) -> str:
    """Digest cell: one decimal, or a dash when the bucket is empty."""
    return f"{value:>10.1f}" if value is not None else f"{'-':>10}"

def main() -> int:
    args = parse_args()
    category_ids, per_file_categories = load_coco_grounding(args.annotations)
    patterns = _category_patterns(category_ids)
    name_by_id = {category_id: name for name, category_id in category_ids.items()}

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
    images = select_images(list_coco_images(args.images_dir), args.n_images)
    prompt = prompt_text(args.question)
    tokenizer = model.tokenizer

    print(
        f"model ready: layers={model.n_layers} d_model={model.d_model} backend={args.backend}; "
        f"identity lens over fitted layers {lens.source_layers[0]}..{lens.source_layers[-1]} "
        f"with bias-{args.mask}.pt from {args.bias_dir}"
    )
    per_layer: dict[int, dict[str, list[int]]] = {
        layer: {
            "grounded": [],
            "halluc": [],
            "true_at_halluc": [],
            "last_patch": [],
            "handoff": [],
        }
        for layer in layers
    }
    image_summaries: list[dict[str, Any]] = []
    n_generated = 0
    n_grounded_words = 0
    n_hallucinated_words = 0
    n_skipped_no_grounding = 0

    for index, image_path in enumerate(images):
        present = per_file_categories.get(image_path.name)
        if not present:
            print(f"  [{index + 1}/{len(images)}] {image_path.name}: no annotations entry - skipped", flush=True)
            n_skipped_no_grounding += 1
            continue
        present_token_ids = _category_token_ids(tokenizer, [name_by_id[cid] for cid in present])

        batch = model.encode_mm(prompt, image_path, max_length=args.max_seq_len)
        patch_residuals = last_patch_residuals(model, batch, layers)
        patch_logits = (
            {
                layer: _calibrated_lens_logits(
                    model, lens, patch_residuals[layer], layer, bias, scale, temp, logit_bias
                )
                for layer in layers
            }
            if patch_residuals is not None
            else None
        )
        new_tokens, step_records = generate_with_residual_traces(
            model, batch, layers, max_new_tokens=args.max_new_tokens
        )
        caption, instances = classify_generated_words(tokenizer, new_tokens, patterns, present)
        n_generated += int(new_tokens.numel())

        for instance in instances:
            bucket = "grounded" if instance.grounded else "halluc"
            for layer in layers:
                logits = _calibrated_lens_logits(
                    model, lens, step_records[instance.token_index][layer], layer,
                    bias, scale, temp, logit_bias,
                )
                per_layer[layer][bucket].append(_rank_of_id(logits, instance.token_id))
                if not instance.grounded and present_token_ids:
                    per_layer[layer]["true_at_halluc"].append(
                        min(_rank_of_id(logits, cid) for cid in present_token_ids)
                    )
        if patch_logits is not None and present_token_ids:
            for layer in layers:
                per_layer[layer]["last_patch"].append(
                    min(_rank_of_id(patch_logits[layer], cid) for cid in present_token_ids)
                )

        if args.handoff_rank and step_records and present_token_ids:
            # step_records[0] is the prefill forward's record: with KV caching its
            # distribution at the last prompt position (the final 'ASSISTANT:' token,
            # right before generated token 0) is the handoff position's lens input.
            handoff_logits = {
                layer: _calibrated_lens_logits(
                    model, lens, step_records[0][layer], layer, bias, scale, temp, logit_bias
                )
                for layer in layers
            }
            for layer in layers:
                per_layer[layer]["handoff"].append(
                    min(_rank_of_id(handoff_logits[layer], cid) for cid in present_token_ids)
                )

        n_grounded = sum(1 for word in instances if word.grounded)
        n_grounded_words += n_grounded
        n_hallucinated_words += len(instances) - n_grounded
        image_summaries.append(
            {
                "image": image_path.name,
                "text": caption,
                "n_generated": int(new_tokens.numel()),
                "n_words_matched": len(instances),
                "grounded": sorted({word.category for word in instances if word.grounded}),
                "hallucinated": sorted({word.category for word in instances if not word.grounded}),
            }
        )
        print(
            f"  [{index + 1}/{len(images)}] {image_path.name}: {len(instances)} matched words "
            f"({n_grounded}g/{len(instances) - n_grounded}h) {caption[:70]!r}",
            flush=True,
        )

    per_layer_out = {
        str(layer): {
            "grounded_rank": _mean_or_none(per_layer[layer]["grounded"]),
            "halluc_rank": _mean_or_none(per_layer[layer]["halluc"]),
            "true_object_rank_at_halluc_positions": _mean_or_none(per_layer[layer]["true_at_halluc"]),
            "last_patch_object_rank": _mean_or_none(per_layer[layer]["last_patch"]),
            "handoff_true_object_rank_mean": _mean_or_none(per_layer[layer]["handoff"]),
            "handoff_true_object_rank_best": (
                min(per_layer[layer]["handoff"]) if per_layer[layer]["handoff"] else None
            ),
            "n_grounded": len(per_layer[layer]["grounded"]),
            "n_hallucinated": len(per_layer[layer]["halluc"]),
        }
        for layer in layers
    }

    print("\n== X14 workspace probe: calibrated lens ranks per layer ==")
    print(
        f"{'layer':<7}{'n_gnd':>6}{'n_hal':>6}{'grounded':>10}{'halluc':>10}"
        f"{'true@hall':>10}{'lastpatch':>10}{'handoff':>10}{'h.best':>10}"
    )
    for layer in layers:
        row = per_layer_out[str(layer)]

        print(
            f"L{layer:<6}{row['n_grounded']:>6}{row['n_hallucinated']:>6}"
            f"{_cell(row['grounded_rank'])}{_cell(row['halluc_rank'])}"
            f"{_cell(row['true_object_rank_at_halluc_positions'])}"
            f"{_cell(row['last_patch_object_rank'])}"
            f"{_cell(row['handoff_true_object_rank_mean'])}"
            f"{_cell(row['handoff_true_object_rank_best'])}"
        )

    separable = [
        layer
        for layer in layers
        if per_layer_out[str(layer)]["grounded_rank"] is not None
        and per_layer_out[str(layer)]["halluc_rank"] is not None
    ]
    if separable:
        best = max(
            separable,
            key=lambda layer: per_layer_out[str(layer)]["halluc_rank"]
            - per_layer_out[str(layer)]["grounded_rank"],
        )
        row = per_layer_out[str(best)]
        print(
            f"\nbest separation: L{best} (halluc {row['halluc_rank']} vs grounded "
            f"{row['grounded_rank']}, gap +{round(row['halluc_rank'] - row['grounded_rank'], 1)})"
        )
        if row["true_object_rank_at_halluc_positions"] is not None:
            print(
                "does-the-model-know test: at hallucinated words' decoding positions the "
                f"image's actual categories rank {row['true_object_rank_at_halluc_positions']} "
                f"vs the hallucinated word's {row['halluc_rank']} at L{best}"
            )
    formed = [
        (per_layer_out[str(layer)]["last_patch_object_rank"], layer)
        for layer in layers
        if per_layer_out[str(layer)]["last_patch_object_rank"] is not None
    ]
    if formed:
        best_patch = min(formed)
        print(
            f"workspace formation (last-patch true-object rank) bottoms out at "
            f"L{best_patch[1]} (rank {best_patch[0]})"
        )

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
        "n_tokens_generated": n_generated,
        "n_words_grounded": n_grounded_words,
        "n_words_hallucinated": n_hallucinated_words,
        "matcher": MATCHER_NOTE,
        "handoff_rank": bool(args.handoff_rank),
        "handoff_rule": HANDOFF_NOTE,
        "category_token_rule": CATEGORY_TOKEN_RULE,
        "rank_rule": RANK_RULE,
        "images": image_summaries,
    }
    report = {"per_layer": per_layer_out, "meta": meta}
    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
