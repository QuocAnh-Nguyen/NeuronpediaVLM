#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""X19: single-image hallucination risk — decoding-time lens entropy + the
pre-generation true-category gap, fused (D45/S3).

Goal: an IMAGE-level hallucination-risk detector with no training. For N COCO
val2014 images (--images-dir/--annotations, x13's loading; --n-images default 200)
the script greedy-generates the caption ONCE per image with x15's machinery
(forward hooks caching the raw block-output residual at every decoding position,
plus ``output_scores``), recording the calibrated logit-lens entropy at every step
and layer, and labels the image HALLUCINATING iff the caption mentions at least
one COCO category that is NOT annotated on the image (x15's
``classify_generated_words``: x13's longest-first matcher + span masking + plural
tolerance; a mention of an annotated category is GROUNDED). Images with no
annotations entry are skipped.

Per image and per requested lens layer (--layers, x18's contract: every requested
layer must be fitted in --lens-dir when the dir exists) four signals are computed:

  (i)  mean_entropy — mean over the image's generated steps of the calibrated-lens
       softmax entropy (x15's ``lens_step_entropies``: x14's readout
       ``unembed(s_l * (h_l + b_l)) / temp + logit_bias`` over the identity lens).
       Steps are the raw generated positions 0..T-1; position 0 is the LAST PROMPT
       position (the x14/x18 handoff convention).
  (ii) max_entropy — max over the same steps.
  (iii) category_gap — the x18-style PRE-GENERATION gap at the last prompt position
       for the image's TRUE categories: per annotated category c,
       ``category_gap(c) = z_L[first_token_id(c)] - z_L[no_id]`` on the calibrated
       lens logits ``z_L`` at the last prompt position — x18's GAP_RULE
       ``z[yes] - z[no]`` with the category's first token id in place of the yes id
       and x18's selected 'no' token id as the contrast (exact token construction
       in CATEGORY_GAP_RULE; handoff position in HANDOFF_NOTE). The image's signal
       is the MEAN of category_gap(c) over its annotated categories (sorted by
       name, so the mean is deterministic).
  (iv) fused — ``z(mean_entropy) + z(category_gap)``, both standardized WITHIN THE
       RUN over the scored images (population moments only — no weights, no layer
       selection inside the score, no supervised fit; FUSION_NOTE). Computed per
       layer (both components at the same layer); the PRIMARY fusion is reported
       at the highest requested layer, the fully calibrated readout.

Every signal gets an image-level tie-aware Mann-Whitney AUROC (positive class =
hallucinating image; AUROC_IMAGE_NOTE) plus hallucinating-vs-clean group means and
counts, per layer and for the best layer of each signal. The digest and the JSON
carry the small-positive-class caveat (CAVEAT_NOTE).

Cost: one generate per image (prefill + max_new_tokens decode steps) plus
|layers| x max_new_tokens cheap matvec+unembeds for the step entropies and |layers|
readouts for the pre-generation gap — never a full forward per layer.

Run (real, CUDA):
  python x19_halluc_risk.py --n-images 200 --json out/x19.json
Smoke (tiny, CPU, the /tmp/x15_smoke fixtures; with the default paths absent the
dataset/lens/affine payload are synthesized exactly as x18 does):
  python x19_halluc_risk.py --backend tiny --layers 0,1,2 --n-images 2 \
      --images-dir /tmp/x15_smoke/images \
      --annotations /tmp/x15_smoke/ann/instances_smoke.json \
      --lens-dir /tmp/x15_smoke/lens --bias-dir /tmp/x15_smoke/bias \
      --max-new-tokens 8 --json /tmp/x19_smoke.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
for _path in (str(REPO / "src"), str(CODE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import vlm_lens  # noqa: E402, F401  # installs the vendored jlens path: must precede jlens
from vlm_lens.artifacts import load_lens_set  # noqa: E402
from vlm_lens.data.captions import (  # noqa: E402
    DEFAULT_QUESTIONS,
    PROMPT_TEMPLATE,
    list_coco_images,
    prompt_template_hash,
    prompt_text,
    select_images,
)
from vlm_lens.models.llava import LlavaLensModel  # noqa: E402

# Sibling reuse (the x16 pattern): x15's generation/matcher/entropy machinery and
# x18's pre-generation gap machinery are imported, not re-derived.
from x15_hesitation import (  # noqa: E402
    MATCHER_NOTE,
    _category_patterns,
    _cell,
    _mean_or_none,
    _round5,
    _word_json,
    auroc,
    classify_generated_words,
    generate_with_residual_traces,
    identity_lens,
    lens_step_entropies,
    load_calibrated_params,
    load_coco_grounding,
    load_model,
)
from x18_pope_handoff import (  # noqa: E402
    GAP_RULE,
    HANDOFF_NOTE,
    MASK,
    SYNTHETIC_NOTE,
    YESNO_RULE,
    _calibrated_lens_logits,
    _encoded_ids,
    identity_lens as identity_lens_over_layers,
    select_yes_no_ids,
    synthetic_grounding,
)

DEFAULT_LENS_DIR = "/data/anhnq/vlm-lens-out/validation/s2-merged/artifacts"
DEFAULT_BIAS_DIR = "/data/anhnq/vlm-lens-out/validation/step5e"
DEFAULT_IMAGES_DIR = "/data/baodq/coco2014/val2014"
DEFAULT_ANNOTATIONS = "/data/baodq/coco2014/annotations/instances_val2014.json"
#: x18's default readout subset (mid/late layers); the tiny fixture has 4 layers.
DEFAULT_LAYERS = "20,24,28,30"

LABEL_RULE = (
    "an image is HALLUCINATING iff its caption mentions >= 1 COCO category that is "
    "not annotated on the image (x15's classify_generated_words: whole-sequence "
    "decode, longest-first word-boundary matcher with plural tolerance and span "
    "masking, x13's detect_mentions semantics); a mention of an annotated category "
    "is GROUNDED; 'clean' image = no hallucinated mention; caption words no category "
    "matched are 'other' (excluded from every aggregate)"
)
SIGNAL_MEAN_ENTROPY_NOTE = (
    "signal (i): mean over the image's generated steps (raw positions 0..T-1; "
    "position 0 is the LAST PROMPT/handoff position predicting token 0) of the "
    "calibrated-lens softmax entropy, per layer — x15's lens_step_entropies "
    "(identity transport, + bias inside the fitted branch, * scale, unembed, "
    "/ temp, + logit_bias; unfitted layers read raw)"
)
SIGNAL_MAX_ENTROPY_NOTE = (
    "signal (ii): max over the same steps as (i), per layer — the single most "
    "hesitant decoding position of the caption under the calibrated lens"
)
CATEGORY_GAP_RULE = (
    "signal (iii), exact construction: per annotated category c, first_id(c) is the "
    "FIRST subtoken id of the encoding of ' ' + c after stripping any leading bos "
    "token and then dropping any leading subtoken equal to the head of the encoding "
    "of ' ' alone (real BPE merges the leading space into the first subtoken and "
    "adds no bos with add_special_tokens=False, so nothing is stripped; the tiny "
    "hash tokenizer prepends its bos and emits the space as its own token shared by "
    "every name, so stripping both keeps the id name-specific); then "
    "category_gap(c) = z_L[first_id(c)] - z_L[no_id] on the calibrated lens logits "
    "z_L at the LAST PROMPT position — x18's GAP_RULE (z[yes]-z[no]) with the "
    "category's first token id in place of the yes id and x18's selected 'no' id as "
    "the contrast; the image's signal is the MEAN of category_gap(c) over the "
    "image's annotated categories sorted by name. x14's _category_token_ids uses "
    "the LAST subword of ' ' + name instead; the first subtoken is used here "
    "because at the captioning handoff a mention of c BEGINS with c's first "
    "subtoken"
)
FUSION_NOTE = (
    "signal (iv): fused = z(mean_entropy) + z(category_gap), both standardized "
    "WITHIN THE RUN over all scored images (population moments, ddof=0; a "
    "zero-std signal contributes exactly 0 and is flagged degenerate in the "
    "moments payload); both components are taken at the SAME layer and the "
    "combination is a plain sum — no fitting beyond the moments payload (no "
    "weights, no layer selection inside the score, no supervised calibration); "
    "computed per layer, with the PRIMARY fusion at the highest requested layer "
    "(the fully calibrated readout); the fused score's orientation is read off "
    "its AUROC (positive class = hallucinating image), no sign is imposed a priori"
)
AUROC_IMAGE_NOTE = (
    "tie-aware Mann-Whitney AUROC with the HALLUCINATING IMAGE as the positive "
    "class: >0.5 means the score runs HIGHER on hallucinating images (x15's rank "
    "math at image level)"
)
CAVEAT_NOTE = (
    "small-positive-class caveat: with few hallucinating images the image-level "
    "AUROC and group-mean estimates are high-variance (x17's rule of thumb: "
    "unstable below ~30 images per group); read every number here together with "
    "the counts and treat single-digit positive counts as anecdotal"
)
STEP_INDEXING_NOTE = (
    "steps arrays are indexed by RAW generated position (special tokens included; "
    "position 0 = the last-prompt/handoff position predicting token 0), x15's "
    "convention; signals (i)/(ii) average/maximize over ALL of an image's steps"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lens-dir", default=DEFAULT_LENS_DIR,
        help="the J source lens dir: only its fitted layer list and d_model are used "
        "(the transport itself is the in-script identity)",
    )
    parser.add_argument(
        "--bias-dir", default=DEFAULT_BIAS_DIR,
        help="directory holding bias-text.pt (the step5e moment payload: bias/scale, "
        "optional temp/logit_bias) applied inside every lens readout",
    )
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR)
    parser.add_argument(
        "--annotations", default=DEFAULT_ANNOTATIONS,
        help="instances_val2014.json: per-image ground-truth categories + the 80 names",
    )
    parser.add_argument("--n-images", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument(
        "--layers", default=DEFAULT_LAYERS,
        help="lens layers for every lens readout (per-step entropies, the "
        "pre-generation gap, the fusion); each must be fitted in --lens-dir when "
        "the dir exists; the tiny fixture has 4 layers, pass e.g. --layers 0,1,2",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="seed for image selection and the synthetic tiny ground truth "
        "(absent objects are seeded per image, so the fixture is stable under "
        "--n-images changes)",
    )
    parser.add_argument(
        "--backend", choices=("hf-llava", "tiny"), default="hf-llava",
        help="model backend: 'hf-llava' (default, CUDA) or the tiny CPU smoke fixture",
    )
    parser.add_argument("--json", required=True)
    return parser.parse_args()


def first_category_token_ids(
    tokenizer: Any, names: Sequence[str]
) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    """The gap token id per category name: the first subtoken of ``' ' + name``.

    See CATEGORY_GAP_RULE for the exact rule (leading bos strip, then the
    leading-space drop that keeps the tiny hash tokenizer's ids name-specific).
    The full per-name encoding is returned so the JSON records exactly which id
    each category contributed.
    """
    bos_id = getattr(tokenizer, "bos_token_id", None)

    def _strip_bos(ids: list[int]) -> list[int]:
        return ids[1:] if bos_id is not None and ids and ids[0] == bos_id else ids

    space_head = (_strip_bos(_encoded_ids(tokenizer, " ")) or [None])[0]
    ids_by_name: dict[str, int] = {}
    encodings: dict[str, dict[str, Any]] = {}
    for name in sorted(set(names)):
        raw = _encoded_ids(tokenizer, " " + name)
        ids = _strip_bos(raw)
        while len(ids) > 1 and space_head is not None and ids[0] == space_head:
            ids = ids[1:]
        if not ids:
            raise ValueError(f"category {name!r} encodes to no tokens")
        ids_by_name[name] = ids[0]
        encodings[name] = {
            "form": " " + name,
            "subtoken_ids": raw,
            "first_token_id": ids[0],
        }
    return ids_by_name, encodings


def zstandardize(values: Sequence[float]) -> tuple[list[float], dict[str, Any]]:
    """Within-run z-scores plus the moments payload (population std, ddof=0).

    A zero-std signal (constant across the run) contributes exact zeros — the
    fused score then reduces to the other component — and the payload flags it.
    """
    if not values:
        return [], {"n": 0, "mean": None, "std": None, "degenerate_zero_std": None}
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    scores = [(value - mean) / std for value in values] if std else [0.0] * len(values)
    return scores, {
        "n": len(values),
        "mean": mean,
        "std": std,
        "degenerate_zero_std": std == 0.0,
    }


def image_auroc(clean_scores: Sequence[float], halluc_scores: Sequence[float]) -> float | None:
    """x15's tie-aware Mann-Whitney AUROC at image level: positive = hallucinating."""
    return auroc(clean_scores, halluc_scores)


def best_layer(aurocs: dict[str, float | None]) -> dict[str, Any] | None:
    """The best layer by AUROC (ties break to the LOWER layer, x18's convention)."""
    ranked = [(value, int(key)) for key, value in aurocs.items() if value is not None]
    if not ranked:
        return None
    value, layer = max(ranked, key=lambda pair: (pair[0], -pair[1]))
    return {"layer": layer, "auroc": value}


def main() -> int:
    args = parse_args()
    layers = sorted({int(part) for part in args.layers.split(",") if part.strip()})
    if not layers:
        raise SystemExit("--layers parsed to an empty list")

    grounding_ready = Path(args.annotations).exists() and Path(args.images_dir).exists()
    if args.backend == "tiny" and not grounding_ready:
        categories, per_file, sources = synthetic_grounding(args)
        synthetic_data = True
    else:
        category_ids, per_file = load_coco_grounding(args.annotations)
        images = select_images(
            list_coco_images(args.images_dir), args.n_images, seed=args.seed
        )
        categories, sources, synthetic_data = (
            category_ids,
            [(p.name, p) for p in images],
            False,
        )
    name_by_id = {category_id: name for name, category_id in categories.items()}
    patterns = _category_patterns(categories)

    model = load_model(args)
    if max(layers) >= model.n_layers:
        raise SystemExit(
            f"requested layers {layers} exceed the model's {model.n_layers} layers "
            f"(valid: 0..{model.n_layers - 1}); on --backend tiny pass e.g. --layers 0,1,2"
        )
    device = model.unembed_weight().device

    if Path(args.lens_dir).exists():
        source_lenses, _ = load_lens_set(args.lens_dir)
        if MASK not in source_lenses:
            raise SystemExit(f"mask {MASK!r} not in {args.lens_dir} (found {sorted(source_lenses)})")
        source_lens = source_lenses[MASK]
        if not source_lens.source_layers:
            raise SystemExit(f"lens {MASK!r} in {args.lens_dir} has no fitted layers")
        missing = [layer for layer in layers if layer not in set(source_lens.source_layers)]
        if missing:
            raise SystemExit(
                f"requested layers {missing} are not fitted in {args.lens_dir} "
                f"(fitted {source_lens.source_layers[0]}..{source_lens.source_layers[-1]})"
            )
        lens = identity_lens(source_lens, device)
        lens_from_dir = True
    else:
        if args.backend != "tiny":
            raise SystemExit(
                f"--lens-dir {args.lens_dir} does not exist (required for the hf-llava backend)"
            )
        lens = identity_lens_over_layers(layers, model.d_model, 0, device)
        lens_from_dir = False

    bias_path = Path(args.bias_dir) / f"bias-{MASK}.pt"
    if bias_path.exists():
        bias, scale, temp, logit_bias, bias_meta = load_calibrated_params(args.bias_dir, MASK)
        bias = {layer: tensor.to(device) for layer, tensor in bias.items()} if bias else None
        logit_bias = (
            {layer: tensor.to(device) for layer, tensor in logit_bias.items()}
            if logit_bias
            else None
        )
        calibrated_from_dir = True
    else:
        if args.backend != "tiny":
            raise SystemExit(
                f"--bias-dir {args.bias_dir} holds no bias-{MASK}.pt (required for the hf-llava backend)"
            )
        bias = scale = temp = logit_bias = None
        bias_meta = {}
        calibrated_from_dir = False

    yes_id, no_id, yesno_note = select_yes_no_ids(model.tokenizer)
    gap_token_ids, token_encodings = first_category_token_ids(model.tokenizer, categories)

    prompt = prompt_text(DEFAULT_QUESTIONS[0])
    tokenizer = model.tokenizer

    print(
        f"model ready: layers={model.n_layers} d_model={model.d_model} backend={args.backend}; "
        f"identity lens over {lens.source_layers[0]}..{lens.source_layers[-1]} "
        f"({'from ' + str(args.lens_dir) if lens_from_dir else 'synthesized over --layers'}) "
        f"with {'bias-' + MASK + '.pt from ' + str(args.bias_dir) if calibrated_from_dir else 'no affine payload'}; "
        f"gap contrast token 'no' id={no_id} ({yesno_note['no']['form']!r})"
    )
    print(f"== X19 hallucination risk: {len(sources)} images, layers={layers}, seed={args.seed} ==")

    image_records: list[dict[str, Any]] = []
    n_words_grounded = 0
    n_words_hallucinated = 0
    n_words_other = 0
    n_images_with_mention = 0
    n_skipped = 0
    for index, (file_name, source) in enumerate(sources):
        present_ids = per_file.get(file_name) or set()
        present_names = sorted(name_by_id[cid] for cid in present_ids if cid in name_by_id)
        if not present_names:
            print(
                f"  [{index + 1}/{len(sources)}] {file_name}: no annotations entry - skipped",
                flush=True,
            )
            n_skipped += 1
            continue

        batch = model.encode_mm(prompt, source, max_length=args.max_seq_len)
        new_tokens, step_records, _scores = generate_with_residual_traces(
            model, batch, layers, max_new_tokens=args.max_new_tokens
        )
        lens_entropies = lens_step_entropies(
            model, lens, step_records, layers, bias, scale, temp, logit_bias
        )
        caption, words = classify_generated_words(tokenizer, new_tokens, patterns, present_ids)
        matched = [word for word in words if word.label != "other"]
        n_grounded = sum(1 for word in matched if word.label == "grounded")
        n_halluc_words = len(matched) - n_grounded
        hallucinating = n_halluc_words > 0
        n_images_with_mention += int(bool(matched))

        # Signal (iii): the pre-generation true-category gap at the LAST PROMPT
        # position (step_records[0], the x14/x18 handoff convention).
        prefill = step_records[0]
        category_detail = []
        category_gap: dict[int, float] = {}
        mean_entropy: dict[int, float] = {}
        max_entropy: dict[int, float] = {}
        for layer in layers:
            logits = _calibrated_lens_logits(
                model, lens, prefill[layer], layer, bias, scale, temp, logit_bias
            )
            per_category = [
                float(logits[gap_token_ids[name]]) - float(logits[no_id])
                for name in present_names
            ]
            category_gap[layer] = sum(per_category) / len(per_category)
            mean_entropy[layer] = sum(lens_entropies[layer]) / len(lens_entropies[layer])
            max_entropy[layer] = max(lens_entropies[layer])
            category_detail.extend(
                {
                    "name": name,
                    "first_token_id": gap_token_ids[name],
                    "gap": round(value, 5),
                    "layer": layer,
                }
                for name, value in zip(present_names, per_category, strict=True)
            )

        image_records.append(
            {
                "image": file_name,
                "caption": caption,
                "hallucinating": hallucinating,
                "n_generated": int(new_tokens.numel()),
                "n_words_matched": len(matched),
                "n_words_grounded": n_grounded,
                "n_words_hallucinated": n_halluc_words,
                "n_words_other": len(words) - len(matched),
                "mean_entropy": {str(layer): round(value, 5) for layer, value in mean_entropy.items()},
                "max_entropy": {str(layer): round(value, 5) for layer, value in max_entropy.items()},
                "category_gap": {str(layer): round(value, 5) for layer, value in category_gap.items()},
                "categories": category_detail,
                "words": [_word_json(word) for word in matched],
                "steps": {
                    "token_ids": [int(token) for token in new_tokens.tolist()],
                    "lens_entropy": {
                        str(layer): _round5(values) for layer, values in lens_entropies.items()
                    },
                },
            }
        )
        n_words_grounded += n_grounded
        n_words_hallucinated += n_halluc_words
        n_words_other += len(words) - len(matched)
        print(
            f"  [{index + 1}/{len(sources)}] {file_name}: "
            f"{'HALLUCINATING' if hallucinating else 'clean'} ({n_grounded}g/{n_halluc_words}h "
            f"of {len(matched)} matched) {caption[:60]!r}",
            flush=True,
        )

    scored = image_records
    halluc_mask = [record["hallucinating"] for record in scored]
    n_halluc = sum(halluc_mask)
    n_clean = len(scored) - n_halluc

    # Signals (i)/(ii)/(iii) per layer -> AUROCs and group means; signal (iv) per
    # layer (z(mean entropy) + z(gap), both components at the same layer).
    per_layer: dict[str, dict[str, Any]] = {}
    moments_by_layer: dict[str, dict[str, Any]] = {}

    def groups(values: list[float]) -> tuple[list[float], list[float]]:
        """The hallucinating / clean split of one per-image signal."""
        return (
            [value for value, flag in zip(values, halluc_mask, strict=True) if flag],
            [value for value, flag in zip(values, halluc_mask, strict=True) if not flag],
        )

    for layer in layers:
        key = str(layer)
        ent_mean = [record["mean_entropy"][key] for record in scored]
        ent_max = [record["max_entropy"][key] for record in scored]
        gaps = [record["category_gap"][key] for record in scored]
        z_ent, moments_ent = zstandardize(ent_mean)
        z_gap, moments_gap = zstandardize(gaps)
        fused = [a + b for a, b in zip(z_ent, z_gap, strict=True)]
        moments_by_layer[key] = {"mean_entropy": moments_ent, "category_gap": moments_gap}

        ent_mean_h, ent_mean_c = groups(ent_mean)
        ent_max_h, ent_max_c = groups(ent_max)
        gap_h, gap_c = groups(gaps)
        fused_h, fused_c = groups(fused)
        per_layer[key] = {
            "n_images": len(scored),
            "n_hallucinating": n_halluc,
            "n_clean": n_clean,
            "auroc_mean_entropy": image_auroc(ent_mean_c, ent_mean_h),
            "auroc_max_entropy": image_auroc(ent_max_c, ent_max_h),
            "auroc_category_gap": image_auroc(gap_c, gap_h),
            "auroc_fused": image_auroc(fused_c, fused_h),
            "mean_entropy_hallucinating": _mean_or_none(ent_mean_h),
            "mean_entropy_clean": _mean_or_none(ent_mean_c),
            "max_entropy_hallucinating": _mean_or_none(ent_max_h),
            "max_entropy_clean": _mean_or_none(ent_max_c),
            "category_gap_hallucinating": _mean_or_none(gap_h),
            "category_gap_clean": _mean_or_none(gap_c),
        }
        for record, value in zip(scored, fused, strict=True):
            record.setdefault("fused_zscore", {})[key] = round(value, 5)

    best = {
        "mean_entropy": best_layer({k: per_layer[k]["auroc_mean_entropy"] for k in per_layer}),
        "max_entropy": best_layer({k: per_layer[k]["auroc_max_entropy"] for k in per_layer}),
        "category_gap": best_layer({k: per_layer[k]["auroc_category_gap"] for k in per_layer}),
        "fused": best_layer({k: per_layer[k]["auroc_fused"] for k in per_layer}),
    }
    top_key = str(layers[-1])
    fusion = {
        "layer": layers[-1],
        "note": FUSION_NOTE,
        "auroc": per_layer[top_key]["auroc_fused"] if scored else None,
        "moments": moments_by_layer[top_key],
    }
    summary = {
        "n_images_scored": len(scored),
        "n_hallucinating": n_halluc,
        "n_clean": n_clean,
        "hallucination_rate": round(n_halluc / len(scored), 4) if scored else None,
        "n_words_grounded": n_words_grounded,
        "n_words_hallucinated": n_words_hallucinated,
        "n_words_other": n_words_other,
        "n_images_with_any_category_mention": n_images_with_mention,
        "best_layers": best,
        "fusion": fusion,
    }

    # ---- digest -----------------------------------------------------------
    print("\n== X19 hallucination risk: image-level AUROC (positive = hallucinating image) ==")
    rate = f"{n_halluc / len(scored):.3f}" if scored else "-"
    print(
        f"images: {len(scored)} scored ({n_halluc} hallucinating / {n_clean} clean; "
        f"rate {rate}); skipped {n_skipped} without annotations"
    )
    print(
        f"caption words: {n_words_grounded + n_words_hallucinated} matched "
        f"({n_words_grounded} grounded / {n_words_hallucinated} hallucinated), "
        f"{n_words_other} other (informational only)"
    )
    print(f"caveat: {n_halluc} hallucinating images — {CAVEAT_NOTE}")
    print(
        "\nper layer (mean/max = per-step calibrated-lens entropy over the caption "
        "steps; gap = x18-style pre-generation true-category gap at the last prompt "
        "position; fused = z(mean) + z(gap) at that layer):"
    )
    print(
        f"{'layer':<7}{'n_img':>6}{'au_mean':>10}{'au_max':>10}{'au_gap':>10}"
        f"{'au_fused':>10}{'ent_hall':>10}{'ent_clean':>10}{'gap_hall':>10}{'gap_clean':>10}"
    )
    for layer in layers:
        row = per_layer[str(layer)]
        print(
            f"{f'L{layer}':<7}{row['n_images']:>6}"
            f"{_cell(row['auroc_mean_entropy'])}{_cell(row['auroc_max_entropy'])}"
            f"{_cell(row['auroc_category_gap'])}{_cell(row['auroc_fused'])}"
            f"{_cell(row['mean_entropy_hallucinating'])}{_cell(row['mean_entropy_clean'])}"
            f"{_cell(row['category_gap_hallucinating'])}{_cell(row['category_gap_clean'])}"
        )
    for name, entry in (
        ("mean-entropy", best["mean_entropy"]),
        ("max-entropy", best["max_entropy"]),
        ("category-gap", best["category_gap"]),
        ("fused", best["fused"]),
    ):
        if entry is not None:
            print(f"best {name} layer: L{entry['layer']} (AUROC {entry['auroc']:.3f})")
        else:
            print(f"best {name} layer: none (no scoreable images)")
    if scored:
        moments = fusion["moments"]
        ent_m, gap_m = moments["mean_entropy"], moments["category_gap"]
        print(
            f"primary fusion at L{fusion['layer']} (highest requested layer): "
            f"AUROC {fusion['auroc']} (z moments: entropy mean={ent_m['mean']:.5f} "
            f"std={ent_m['std']:.5f}, gap mean={gap_m['mean']:.5f} std={gap_m['std']:.5f})"
        )
    else:
        print("primary fusion: none (no scoreable images)")

    meta: dict[str, Any] = {
        "experiment": "x19_halluc_risk",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "lens_dir": str(args.lens_dir),
        "lens_from_dir": lens_from_dir,
        "bias_dir": str(args.bias_dir),
        "calibrated_from_dir": calibrated_from_dir,
        "mask": MASK,
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
            "bias_meta": bias_meta,
        },
        "question": DEFAULT_QUESTIONS[0],
        "prompt": prompt,
        "prompt_template": PROMPT_TEMPLATE,
        "prompt_template_hash": prompt_template_hash(),
        "images_dir": str(args.images_dir),
        "annotations": str(args.annotations),
        "synthetic_data": synthetic_data,
        "synthetic_note": SYNTHETIC_NOTE if synthetic_data or not lens_from_dir else None,
        "n_images": len(sources),
        "n_images_skipped_no_annotations": n_skipped,
        "n_images_scored": len(scored),
        "seed": args.seed,
        "layers": layers,
        "max_new_tokens": args.max_new_tokens,
        "max_seq_len": args.max_seq_len,
        "handoff_position": HANDOFF_NOTE,
        "gap_rule": GAP_RULE,
        "yesno": yesno_note,
        "gap_token_no_id": no_id,
        "gap_token_yes_id": yes_id,
        "category_token_rule": CATEGORY_GAP_RULE,
        "category_token_encodings": token_encodings,
        "notes": {
            "label": LABEL_RULE,
            "matcher": MATCHER_NOTE,
            "mean_entropy": SIGNAL_MEAN_ENTROPY_NOTE,
            "max_entropy": SIGNAL_MAX_ENTROPY_NOTE,
            "fusion": FUSION_NOTE,
            "auroc": AUROC_IMAGE_NOTE,
            "caveat": CAVEAT_NOTE,
            "step_indexing": STEP_INDEXING_NOTE,
        },
    }

    report = {"meta": meta, "images": scored, "per_layer": per_layer, "summary": summary}
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.json} ({len(scored)} scored images over {len(sources)} selected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
