#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""X18: the pre-generation POPE detector — the calibrated lens's yes/no gap at the
last prompt position (D44 H4, HALP-style).

Hypothesis (D44 H4): one pre-generation forward pass per question suffices for a
training-free POPE detector. At the LAST PROMPT position (the prefill decoding
position whose distribution predicts generated token 0, right after the final
"ASSISTANT:" token) the calibrated logit lens yields a distribution whose
``P(yes) - P(no)`` gap separates questions about a truly PRESENT object from ABSENT
ones, and flags the model's own hallucinated "yes" answers (a yes emitted about an
absent object should carry a lower internal-belief gap than a grounded yes).

Protocol. For N COCO val2014 images the questions come from instances_val2014.json
itself: 2 PRESENT-object questions (the image's annotated categories) + 2 ABSENT-object
questions (seeded random other categories), phrased "Is there a {object} in the
image?" in x13's LLaVA prompt template — 4 question pairs per image. Every question
runs ONE greedy generate (4 tokens) under forward hooks on the requested layers; the
prefill record (``step_records[0]``, the x14 handoff convention) caches the raw
residual at the last prompt position for every layer. The calibrated lens distribution
is computed from that cached residual exactly as in x14 — identity transport
(torch.eye per fitted layer of the --lens-dir source lens) plus the --bias-dir
``bias-<mask>.pt`` affine payload, ``z = unembed(s_l * (h_l + b_l))`` (temp /
logit_bias when the payload carries them) — and ``lens_gap = z[yes] - z[no]``. The
model's own answer is parsed from the 4 generated tokens (yes / no / unparsed;
no-answers and unparsed answers are counted) and only feeds the accuracy and
hallucination splits — the lens gap itself is fully pre-generation.

Scores (per layer, plus the best layer by present-vs-absent AUROC):
  - model accuracy on present (says yes) / absent (says no) / overall;
  - AUROC of lens_gap, ground-truth present (positive) vs absent;
  - AUROC of lens_gap, model-correct (positive) vs model-wrong;
  - AUROC of lens_gap, HALLUCINATED model-yes on absent objects (positive) vs
    correct-yes on present objects — the detector question: AUROC < 0.5 means the
    calibrated gap runs lower when the model's yes is hallucinated, so 1 - AUROC is
    the discrimination of a "low gap => hallucinated yes" flag.

``--backend tiny`` runs the whole pipeline on the tiny CPU fixture. When the default
COCO/lens/bias paths do not exist (any non-deployment box), the dataset, the identity
lens (over --layers) and an empty affine payload are synthesized so the smoke still
exercises every code path end-to-end on synthetic annotation objects; the report meta
records exactly what was synthesized.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402, I001  # the vlm_lens import must precede jlens: it installs the path
import vlm_lens  # noqa: E402, F401

from jlens.lens import JacobianLens  # noqa: E402

from vlm_lens.artifacts import load_bias, load_lens_set  # noqa: E402
from vlm_lens.data.captions import (  # noqa: E402
    PROMPT_TEMPLATE,
    list_coco_images,
    prompt_template_hash,
    prompt_text,
    select_images,
)
from vlm_lens.models.llava import LlavaLensModel, MultimodalBatch  # noqa: E402

MASK = "text"
DEFAULT_LENS_DIR = "/data/anhnq/vlm-lens-out/validation/s2-merged/artifacts"
DEFAULT_BIAS_DIR = "/data/anhnq/vlm-lens-out/validation/step5e"
DEFAULT_IMAGES_DIR = "/data/baodq/coco2014/val2014"
DEFAULT_ANNOTATIONS = "/data/baodq/coco2014/annotations/instances_val2014.json"

#: The simple, explicit POPE question form (asked through x13's LLaVA template).
QUESTION_TEMPLATE = "Is there a {object} in the image?"
DEFAULT_LAYERS = "20,24,28,30"
MAX_SEQ_LEN = 1536
#: Greedy answer length: enough for "Yes"/"No" plus a short continuation, still bounded.
ANSWER_TOKENS = 4
#: POPE's balanced pairing: 2 present + 2 absent questions per image.
PRESENT_QUESTIONS_PER_IMAGE = 2
ABSENT_QUESTIONS_PER_IMAGE = 2
#: Categories for the synthetic tiny-smoke ground truth (real COCO names).
SYNTHETIC_CATEGORIES = (
    "dog", "cat", "bird", "horse", "sheep", "cow", "car",
    "bus", "truck", "chair", "bottle", "cup", "pizza", "boat",
)

HANDOFF_NOTE = (
    "the lens reads the LAST PROMPT position: with KV caching, step_records[0] of the "
    "generate call is the prefill forward's record at the position whose distribution "
    "predicts generated token 0 (right after the final 'ASSISTANT:' token) — the x14 "
    "handoff convention, so the measurement is a single pre-generation forward"
)
GAP_RULE = (
    "lens_gap = z[yes_id] - z[no_id] on the calibrated lens distribution at the last "
    "prompt position; positive = the lens leans yes"
)
YESNO_RULE = (
    "the gap uses the tokenizer ids of the capitalized leading-space forms ' Yes' / "
    "' No' when those are single tokens — the repo's teacher-forced POPE answer form "
    "(data/pope.py appends ' ' + label.capitalize() to this same template, and "
    "generated content follows the leading-space emission convention); both "
    "' yes'/'no' and capitalized forms are evaluated for every question word and ALL "
    "candidates are recorded; if no form is a single token (the tiny hash tokenizer), "
    "the first form of (' Yes', ' yes', 'Yes', 'yes') whose yes/no ids exist and "
    "differ supplies the FINAL subtoken id (x14's _category_token_ids convention) "
    "with single_token=false"
)
AUROC_RULE = (
    "rank-based AUROC (Mann-Whitney U, average ranks for ties) = P(score_positive > "
    "score_negative) + 0.5*P(tie); groups: present_vs_absent uses the ground-truth "
    "label (positive = present); model_correct_vs_wrong (positive = correct) pools "
    "both labels; halluc_yes_vs_grounded_yes (positive = model-yes on an ABSENT "
    "object, negative = model-yes on a PRESENT object) is the detector question"
)
SYNTHETIC_NOTE = (
    "with --backend tiny, when the annotations/images/lens/bias paths are absent the "
    "whole fixture is synthesized: SYNTHETIC_CATEGORIES objects with seeded per-image "
    "ground truth, tiny random images, an identity lens over --layers (n_prompts=0) "
    "and an empty affine payload (pure unembed readout); every downstream code path "
    "is unchanged"
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
    parser.add_argument("--n-images", type=int, default=50)
    parser.add_argument(
        "--layers", default=DEFAULT_LAYERS,
        help="lens layers to read the gap at (default: a mid/late subset; the tiny "
        "fixture has 4 layers, pass e.g. --layers 1,2,3)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="seed for image selection, absent-object sampling and the synthetic "
        "tiny ground truth (absent objects are seeded per image, so the plan is "
        "stable under --n-images changes)",
    )
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


def identity_lens(
    fitted_layers: Sequence[int], d_model: int, n_prompts: int, device: torch.device
) -> JacobianLens:
    """``torch.eye`` JacobianLens over ``fitted_layers`` (zero fitting, identity transport).

    Built directly on ``device`` so ``transport``'s per-call ``.to`` is a no-op (x14).
    """
    return JacobianLens(
        jacobians={
            layer: torch.eye(d_model, dtype=torch.float32, device=device)
            for layer in fitted_layers
        },
        n_prompts=n_prompts,
        d_model=d_model,
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
    batch: MultimodalBatch,
    layers: list[int],
    *,
    max_new_tokens: int,
) -> tuple[torch.Tensor, list[dict[int, torch.Tensor]]]:
    """Greedy generation caching the RAW block-output residual at each decoding position.

    Mirrors ``trace_generation``'s hook machinery and step convention (with KV caching,
    the prefill forward predicts token 0 at the last prompt position and each decode
    forward predicts one more, so hook call i predicts generated token i);
    ``step_records[0]`` IS the last-prompt-position record this script measures.
    """
    first_layer = layers[0]
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
        model.layers[layer].register_forward_hook(make_hook(layer)) for layer in layers
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


def decode_tokens(model: LlavaLensModel, token_ids: torch.Tensor) -> str:
    """Caption-style decode of the generated tokens (mirrors ``LlavaLensModel._decode``)."""
    tokenizer = model.tokenizer
    if tokenizer is not None and hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)
    return " ".join(str(int(t)) for t in token_ids)


def _surface_forms(word: str) -> tuple[str, str, str, str]:
    """The checked surface forms for a yes/no word: ' yes'/'no' and capitalized forms."""
    return (f" {word.capitalize()}", f" {word}", word.capitalize(), word)


def _encoded_ids(tokenizer: Any, form: str) -> list[int]:
    encoded = tokenizer(form, add_special_tokens=False)["input_ids"]
    if torch.is_tensor(encoded):  # the tiny stand-in returns [1, n] tensors
        encoded = encoded[0].tolist()
    return [int(token_id) for token_id in encoded]


def select_yes_no_ids(tokenizer: Any) -> tuple[int, int, dict[str, Any]]:
    """Pick the yes/no token ids for the gap (see YESNO_RULE); record every candidate.

    The capitalized leading-space forms are preferred because that is the repo's
    teacher-forced POPE answer form; the bare and lowercase forms are always evaluated
    so the choice is documented either way. The id of a chosen form is its final
    subtoken id (x14's ``_category_token_ids`` convention).
    """
    yes_forms = _surface_forms("yes")
    no_forms = _surface_forms("no")
    candidates: dict[str, dict[str, dict[str, Any]]] = {}
    for word, forms in (("yes", yes_forms), ("no", no_forms)):
        candidates[word] = {
            form: {"ids": _encoded_ids(tokenizer, form), "single_token": None}
            for form in forms
        }
    for word in candidates:
        for entry in candidates[word].values():
            entry["single_token"] = len(entry["ids"]) == 1

    def form_id(word: str, form: str) -> int | None:
        ids = candidates[word][form]["ids"]
        return ids[-1] if ids else None

    chosen: tuple[int, int, int] | None = None
    fallback: tuple[int, int, int] | None = None
    for index in range(len(yes_forms)):
        yes_id = form_id("yes", yes_forms[index])
        no_id = form_id("no", no_forms[index])
        if yes_id is None or no_id is None or yes_id == no_id:
            continue
        both_single = (
            candidates["yes"][yes_forms[index]]["single_token"]
            and candidates["no"][no_forms[index]]["single_token"]
        )
        if both_single:
            chosen = (yes_id, no_id, index)
            break
        if fallback is None:
            fallback = (yes_id, no_id, index)
    if chosen is None:
        chosen = fallback
    if chosen is None:
        raise ValueError(
            "no usable yes/no token pair: every evaluated surface form "
            f"({yes_forms}) is empty or collides with its 'no' counterpart"
        )
    yes_id, no_id, index = chosen
    note = {
        "rule": YESNO_RULE,
        "yes": {
            "form": yes_forms[index],
            "id": yes_id,
            "single_token": candidates["yes"][yes_forms[index]]["single_token"],
        },
        "no": {
            "form": no_forms[index],
            "id": no_id,
            "single_token": candidates["no"][no_forms[index]]["single_token"],
        },
        "candidates": candidates,
    }
    return yes_id, no_id, note


def parse_answer(text: str) -> str:
    """The generated answer's verdict: ``yes`` / ``no`` / ``unparsed``.

    The first word decides when it is yes/no (the usual "Yes, there is ..." form);
    otherwise a unique standalone yes/no anywhere in the answer is used; ambiguous or
    word-free answers are ``unparsed``.
    """
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return "unparsed"
    if words[0] in ("yes", "no"):
        return words[0]
    hits = {word for word in words if word in ("yes", "no")}
    return hits.pop() if len(hits) == 1 else "unparsed"


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


def question_plan(
    present_names: list[str], all_names: list[str], file_name: str, seed: int
) -> list[tuple[str, bool]]:
    """Up to 2 present + 2 absent ``(object, present)`` questions for one image.

    Present objects: the alphabetically first annotated categories (deterministic;
    the hypothesis test does not depend on which present object is asked). Absent
    objects: seeded random non-present categories, seeded per image (``seed:file``)
    so the plan is stable under ``--n-images`` changes.
    """
    plan = [(name, True) for name in sorted(present_names)[:PRESENT_QUESTIONS_PER_IMAGE]]
    rng = random.Random(f"x18-pope:{seed}:{file_name}")
    present_set = set(present_names)
    candidates = [name for name in all_names if name not in present_set]
    n_absent = min(ABSENT_QUESTIONS_PER_IMAGE, len(candidates))
    plan.extend((name, False) for name in rng.sample(candidates, n_absent))
    return plan


def synthetic_grounding(
    args: argparse.Namespace,
) -> tuple[dict[str, int], dict[str, set[int]], list[tuple[str, Any]]]:
    """Tiny-smoke dataset: seeded per-image ground truth + tiny random images."""
    categories = {name: index + 1 for index, name in enumerate(SYNTHETIC_CATEGORIES)}
    from vlm_lens.models.tiny_llava import random_image

    per_file: dict[str, set[int]] = {}
    sources: list[tuple[str, Any]] = []
    for index in range(args.n_images):
        name = f"tiny_synthetic_{index:06d}.jpg"
        rng = random.Random(f"x18-synth-gt:{args.seed}:{name}")
        present = rng.sample(sorted(categories), rng.randint(1, 3))
        per_file[name] = {categories[n] for n in present}
        sources.append((name, random_image(args.seed + index)))
    return categories, per_file, sources


def auroc(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    """Rank-based AUROC (Mann-Whitney U; average ranks for ties), or ``None`` if a group is empty."""
    if not positive or not negative:
        return None
    combined = [(float(score), 1) for score in positive]
    combined.extend((float(score), 0) for score in negative)
    combined.sort(key=lambda pair: pair[0])
    n_pos = sum(label for _, label in combined)
    n_neg = len(combined) - n_pos
    rank_sum_pos = 0.0
    index = 0
    while index < len(combined):
        stop = index
        while stop < len(combined) and combined[stop][0] == combined[index][0]:
            stop += 1
        average_rank = (index + stop + 1) / 2.0  # 1-based ranks index+1 .. stop, averaged
        for within in range(index, stop):
            if combined[within][1]:
                rank_sum_pos += average_rank
        index = stop
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def _mean_or_none(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _cell(value: float | None) -> str:
    return f"{value:>9.3f}" if value is not None else f"{'-':>9}"


def main() -> int:
    args = parse_args()
    layers = sorted({int(part) for part in args.layers.split(",") if part.strip()})
    if not layers:
        raise SystemExit("--layers parsed to an empty list")

    model = load_model(args)
    if max(layers) >= model.n_layers:
        raise SystemExit(
            f"requested layers {layers} exceed the model's {model.n_layers} layers "
            f"(valid: 0..{model.n_layers - 1}); on --backend tiny pass e.g. --layers 1,2"
        )
    device = model.unembed_weight().device

    grounding_ready = Path(args.annotations).exists() and Path(args.images_dir).exists()
    if args.backend == "tiny" and not grounding_ready:
        categories, per_file, sources = synthetic_grounding(args)
        synthetic_data = True
    else:
        category_ids, per_file = load_coco_grounding(args.annotations)
        images = select_images(
            list_coco_images(args.images_dir), args.n_images, seed=args.seed
        )
        categories, sources, synthetic_data = category_ids, [(p.name, p) for p in images], False
    name_by_id = {category_id: name for name, category_id in categories.items()}

    if Path(args.lens_dir).exists():
        source_lenses, _ = load_lens_set(args.lens_dir)
        if MASK not in source_lenses:
            raise SystemExit(f"mask {MASK!r} not in {args.lens_dir} (found {sorted(source_lenses)})")
        source_lens = source_lenses.get(MASK)
        if not source_lens.source_layers:
            raise SystemExit(f"lens {MASK!r} in {args.lens_dir} has no fitted layers")
        missing = [layer for layer in layers if layer not in set(source_lens.source_layers)]
        if missing:
            raise SystemExit(
                f"requested layers {missing} are not fitted in {args.lens_dir} "
                f"(fitted {source_lens.source_layers[0]}..{source_lens.source_layers[-1]})"
            )
        lens = identity_lens(
            source_lens.source_layers, source_lens.d_model, source_lens.n_prompts, device
        )
        lens_from_dir = True
    else:
        if args.backend != "tiny":
            raise SystemExit(
                f"--lens-dir {args.lens_dir} does not exist (required for the hf-llava backend)"
            )
        lens = identity_lens(layers, model.d_model, 0, device)
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

    print(
        f"model ready: layers={model.n_layers} d_model={model.d_model} backend={args.backend}; "
        f"identity lens over {lens.source_layers[0]}..{lens.source_layers[-1]} "
        f"({'from ' + str(args.lens_dir) if lens_from_dir else 'synthesized over --layers'}) "
        f"with {'bias-' + MASK + '.pt from ' + str(args.bias_dir) if calibrated_from_dir else 'no affine payload'}; "
        f"yes/no ids {yes_id}/{no_id} ({yesno_note['yes']['form']!r}/{yesno_note['no']['form']!r})"
    )
    print(f"== X18 POPE handoff: {len(sources)} images, layers={layers}, seed={args.seed} ==")

    all_names = sorted(categories)
    items: list[dict[str, Any]] = []
    n_skipped = 0
    for index, (file_name, source) in enumerate(sources):
        present_ids = per_file.get(file_name) or set()
        present_names = sorted(name_by_id[cid] for cid in present_ids if cid in name_by_id)
        if not present_names:
            print(f"  [{index + 1}/{len(sources)}] {file_name}: no annotations entry - skipped", flush=True)
            n_skipped += 1
            continue
        plan = question_plan(present_names, all_names, file_name, args.seed)
        for obj, present in plan:
            prompt = prompt_text(QUESTION_TEMPLATE.format(object=obj))
            batch = model.encode_mm(prompt, source, max_length=MAX_SEQ_LEN)
            new_tokens, step_records = generate_with_residual_traces(
                model, batch, layers, max_new_tokens=ANSWER_TOKENS
            )
            answer_text = decode_tokens(model, new_tokens)
            model_answer = parse_answer(answer_text)
            prefill = step_records[0]  # the LAST PROMPT position (see HANDOFF_NOTE)
            lens_gaps: dict[str, float] = {}
            for layer in layers:
                logits = _calibrated_lens_logits(
                    model, lens, prefill[layer], layer, bias, scale, temp, logit_bias
                )
                lens_gaps[str(layer)] = float(logits[yes_id] - logits[no_id])
            items.append(
                {
                    "image": file_name,
                    "object": obj,
                    "present": bool(present),
                    "label": "yes" if present else "no",
                    "question": QUESTION_TEMPLATE.format(object=obj),
                    "model_answer": model_answer,
                    "answer_text": answer_text,
                    "correct": model_answer == ("yes" if present else "no"),
                    "lens_gaps": lens_gaps,
                }
            )
        asked_present = [obj for obj, present in plan if present]
        asked_absent = [obj for obj, present in plan if not present]
        print(
            f"  [{index + 1}/{len(sources)}] {file_name}: "
            f"present={asked_present or '-'} absent={asked_absent or '-'}",
            flush=True,
        )

    # The per-item convenience gap is the best layer's gap, by the ground-truth AUROC.
    per_layer: dict[str, dict[str, Any]] = {}
    for layer in layers:
        key = str(layer)
        gaps = [item["lens_gaps"][key] for item in items]
        present_gaps = [item["lens_gaps"][key] for item in items if item["present"]]
        absent_gaps = [item["lens_gaps"][key] for item in items if not item["present"]]
        correct_gaps = [item["lens_gaps"][key] for item in items if item["correct"]]
        wrong_gaps = [item["lens_gaps"][key] for item in items if not item["correct"]]
        halluc_yes_gaps = [
            item["lens_gaps"][key]
            for item in items
            if not item["present"] and item["model_answer"] == "yes"
        ]
        grounded_yes_gaps = [
            item["lens_gaps"][key]
            for item in items
            if item["present"] and item["model_answer"] == "yes"
        ]
        per_layer[key] = {
            "n_questions": len(gaps),
            "n_present": len(present_gaps),
            "n_absent": len(absent_gaps),
            "auroc_present_vs_absent": auroc(present_gaps, absent_gaps),
            "auroc_model_correct_vs_wrong": auroc(correct_gaps, wrong_gaps),
            "auroc_halluc_yes_vs_grounded_yes": auroc(halluc_yes_gaps, grounded_yes_gaps),
            "n_model_yes_halluc": len(halluc_yes_gaps),
            "n_model_yes_grounded": len(grounded_yes_gaps),
            "mean_gap_present": _mean_or_none(present_gaps),
            "mean_gap_absent": _mean_or_none(absent_gaps),
        }
    ranked = [
        (per_layer[str(layer)]["auroc_present_vs_absent"], layer)
        for layer in layers
        if per_layer[str(layer)]["auroc_present_vs_absent"] is not None
    ]
    best_layer = max(ranked, key=lambda pair: (pair[0], -pair[1]))[1] if ranked else None
    if best_layer is not None:
        best_key = str(best_layer)
        for item in items:
            item["lens_gap"] = item["lens_gaps"][best_key]

    present_items = [item for item in items if item["present"]]
    absent_items = [item for item in items if not item["present"]]
    answer_counts = {
        verdict: sum(1 for item in items if item["model_answer"] == verdict)
        for verdict in ("yes", "no", "unparsed")
    }
    summary: dict[str, Any] = {
        "n_images": len(sources) - n_skipped,
        "n_questions": len(items),
        "n_present_questions": len(present_items),
        "n_absent_questions": len(absent_items),
        "model_accuracy": {
            "present": _mean_or_none(
                [float(item["model_answer"] == "yes") for item in present_items]
            ),
            "absent": _mean_or_none(
                [float(item["model_answer"] == "no") for item in absent_items]
            ),
            "overall": _mean_or_none(
                [float(item["correct"]) for item in items]
            ),
        },
        "model_answers": answer_counts,
        "best_layer": best_layer,
        "auroc": (
            {
                "present_vs_absent": per_layer[str(best_layer)]["auroc_present_vs_absent"],
                "model_correct_vs_wrong": per_layer[str(best_layer)]["auroc_model_correct_vs_wrong"],
                "halluc_yes_vs_grounded_yes": per_layer[str(best_layer)][
                    "auroc_halluc_yes_vs_grounded_yes"
                ],
            }
            if best_layer is not None
            else {}
        ),
    }

    meta: dict[str, Any] = {
        "experiment": "x18_pope_handoff",
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
        "question_template": QUESTION_TEMPLATE,
        "prompt_template": PROMPT_TEMPLATE,
        "prompt_template_hash": prompt_template_hash(),
        "prompt_example": prompt_text(QUESTION_TEMPLATE.format(object="dog")),
        "present_questions_per_image": PRESENT_QUESTIONS_PER_IMAGE,
        "absent_questions_per_image": ABSENT_QUESTIONS_PER_IMAGE,
        "images_dir": str(args.images_dir),
        "annotations": str(args.annotations),
        "synthetic_data": synthetic_data,
        "synthetic_note": SYNTHETIC_NOTE if synthetic_data or not lens_from_dir else None,
        "n_images": len(sources),
        "n_images_skipped_no_annotations": n_skipped,
        "seed": args.seed,
        "layers": layers,
        "answer_tokens": ANSWER_TOKENS,
        "max_seq_len": MAX_SEQ_LEN,
        "handoff_position": HANDOFF_NOTE,
        "gap_rule": GAP_RULE,
        "yesno": yesno_note,
        "auroc_rule": AUROC_RULE,
    }

    print("\n== X18 POPE handoff: per-layer digest ==")
    print(
        f"{'layer':>6}{'n_q':>5}{'n_pres':>8}{'au_pres':>10}{'au_corr':>10}"
        f"{'au_hall':>10}{'gap_pres':>10}{'gap_abs':>10}"
    )
    for layer in layers:
        row = per_layer[str(layer)]
        print(
            f"L{layer:<5}{row['n_questions']:>5}{row['n_present']:>8}"
            f"{_cell(row['auroc_present_vs_absent'])}"
            f"{_cell(row['auroc_model_correct_vs_wrong'])}"
            f"{_cell(row['auroc_halluc_yes_vs_grounded_yes'])}"
            f"{_cell(row['mean_gap_present'])}"
            f"{_cell(row['mean_gap_absent'])}"
        )
    print(
        f"\nmodel answers: yes={answer_counts['yes']} no={answer_counts['no']} "
        f"unparsed={answer_counts['unparsed']}"
    )
    accuracy = summary["model_accuracy"]
    print(
        f"accuracy: present={accuracy['present']} absent={accuracy['absent']} "
        f"overall={accuracy['overall']}"
    )
    if best_layer is not None:
        row = per_layer[str(best_layer)]
        print(
            f"best layer: L{best_layer} (AUROC present-vs-absent "
            f"{row['auroc_present_vs_absent']:.3f}; halluc-yes detector AUROC "
            f"{row['auroc_halluc_yes_vs_grounded_yes']} — <0.5 means the calibrated gap "
            f"runs lower on the model's hallucinated yes)"
        )
    else:
        print("best layer: none (no scoreable questions)")

    report = {"meta": meta, "items": items, "per_layer": per_layer, "summary": summary}
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.json} ({len(items)} questions over {summary['n_images']} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
