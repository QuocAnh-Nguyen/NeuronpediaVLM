#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""X24: the POPE yes/no microscope — the internal belief trajectory behind every
answer, layer by layer, with and without the image.

For every question of a POPE split (the ``<split>-open.jsonl`` manifests built by
``build_pope.py``) the arm encodes the prompt WITHOUT the answer through the registry's
established image-encode path, reads the residual at the final prompt position (the
answer position; x18's handoff convention) at EVERY lens-fitted layer, and turns the
calibrated J-lens logits into ``P_yes(l) = sigmoid(z_yes - z_no)`` — a binary softmax
over the two resolved answer-token ids, so the number is tokenizer-agnostic and
comparable across layers. The same question is then re-encoded through the registry's
text-only chat path with NO image, giving ``P_yes_free(l)``; the per-layer evidence gap
``g_l = P_yes(l) - P_yes_free(l)`` is how much the image itself moves the yes-belief.

Around that measurement the arm computes:

* the model's own answer + margin from the model's final-position logits (raw unembed
  of the final-layer residual — never transported, never calibrated; the T26 calibration
  applies only to the lens side);
* the commitment layer: the smallest fitted layer whose ``P_yes`` already sits on the
  model's final-answer side of 0.5 and never crosses back — where the answer became an
  internal fact (distribution recorded per split);
* error prediction: rank-AUC of per-layer features (answer-aligned ``P_yes`` margins and
  answer-aligned gaps) for predicting model ERROR (model answer vs the POPE ground-truth
  label), against the best final-logit baseline (``-|margin|`` and answer-aligned final
  ``P_yes``);
* the POPE scoreboard per split: accuracy, yes-class precision/recall/F1, model vs
  ground-truth yes-rate (the hallucination-relevant bias), gap curves, and the AUC
  table.

Calibration: ``--bias`` points at a ``bias-<mask>.pt`` payload (T26's
``artifacts.save_bias`` family; ``logit_scale``/``logit_bias`` fitted on the calibrated
readout, plus the schema-required zero residual bias and unit scale). It is applied in
the canonical order ``z = unembed(transport(h_l)) / temp * logit_scale + logit_bias``
inside a local mirror of x14's ``_calibrated_lens_logits`` (x14's helper has no
``logit_scale`` slot). Missing/unreadable/dimension-mismatched payloads are tolerated:
the arm runs raw and records the drop in ``meta.bias``. ``meta.lens.fingerprint`` (x21's
J-content sha256) and the payload's own ``base_lens.fingerprint`` land in the report, so
bias<->lens binding is verifiable from the JSON alone.

Backends: ``--backend hf-llava`` (default, CUDA, ``--model`` dispatch through the
registry) or ``--backend tiny`` — the tiny CPU fixture has no real yes/no semantics, so
the smoke asserts mechanics (shapes, trajectories, JSON schema, determinism), never
accuracy values.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch  # noqa: E402  # the vlm_lens import must precede jlens: it installs the path


def _find_repo_root() -> Path:
    """The repo owning ``src/vlm_lens`` (walk up from this file)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src" / "vlm_lens").is_dir():
            return parent
    return here.parents[3] if len(here.parents) > 3 else Path(here.anchor)


REPO = _find_repo_root()
CODE = Path(__file__).resolve().parent
for _path in (str(REPO / "src"), str(CODE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import vlm_lens  # noqa: E402, F401  # installs the vendored jlens path: must precede jlens
from jlens.hooks import ActivationRecorder  # noqa: E402
from jlens.lens import JacobianLens  # noqa: E402

from vlm_lens.artifacts import load_bias, load_lens_set  # noqa: E402
from vlm_lens.data.manifest import read_manifest  # noqa: E402

import vlm_registry as registry  # noqa: E402

#: Any registry-built lens model (``LlavaLensModel`` for the llava kinds,
#: ``Qwen2VLLensModel`` for qwen2vl -- same public surface).
LensModel = Any

from x18_pope_handoff import auroc  # noqa: E402
from x21_verbal_report import lens_fingerprint  # noqa: E402

DEFAULT_LENS_DIR = "/data/anhnq/vlm-lens-out/validation/fits/llava-1.5-7b-hf/merged"
DEFAULT_POPE_DIR = "/data/anhnq/vlm-lens-out/validation/pope"
MASK = "text"
DEFAULT_BIAS_PATH = str(Path(DEFAULT_LENS_DIR) / "bias-text.pt")


P_YES_RULE = (
    "P_yes(l) = sigmoid(z_yes - z_no) on the calibrated J-lens logits at the FINAL "
    "PROMPT position (the answer position; with llava the prompt ends 'ASSISTANT:', "
    "qwen2vl carries the appended assistant header) — a binary softmax over the two "
    "resolved answer-token ids, comparable across layers and tokenizers"
)
GAP_RULE = (
    "evidence gap g_l = P_yes(l) - P_yes_free(l); P_yes_free is the same question "
    "encoded through the registry's text-only chat path (NO image tokens), so g_l is "
    "how much the image itself moves the yes-belief at layer l"
)
IMAGE_FREE_NOTE = (
    "the image-free run is a DIFFERENT sequence (no placeholder tokens), so only the "
    "answer-position distribution is compared, never intermediate positions"
)
COMMITMENT_RULE = (
    "commitment layer = smallest fitted layer where sign(P_yes(l) - 0.5) equals the "
    "model's final answer and stays there for every later fitted layer; null when the "
    "readout never stabilizes on the answer side"
)
ERROR_AUC_RULE = (
    "rank AUROC (Mann-Whitney U) predicting model ERROR, positive group = model answer "
    "!= POPE label; per-layer features, both scored so 'the committed answer is not "
    "supported' ranks high: (a) 1 - answer-aligned P_yes(l), where alignment is "
    "confidence (P_yes itself for a yes answer, 1 - P_yes for a no answer); "
    "(b) -answer-aligned gap, where alignment is sign (g_l as-is for a yes answer, "
    "negated for a no answer)"
)
BASELINE_RULE = (
    "final-logit baselines from the model's OWN final-position logits (raw unembed, "
    "never calibrated): -|margin| and 1 - answer-aligned P_yes(final); the best of the "
    "two is the number the lens AUCs must beat"
)
TINY_NOTE = (
    "the tiny fixture is a random-weight model with no yes/no semantics: accuracy, F1 "
    "and AUC values are meaningless there; the smoke asserts mechanics only"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("hf-llava", "tiny"), default="hf-llava",
        help="model backend: 'hf-llava' (default, CUDA) or the tiny CPU smoke fixture",
    )
    parser.add_argument(
        "--model", choices=registry.MODEL_KINDS, default=registry.DEFAULT_KIND,
        help="checkpoint kind (HF backend); --backend tiny implies llava-7b geometry",
    )
    parser.add_argument("--pope-dir", default=DEFAULT_POPE_DIR,
                        help="directory holding <split>-open.jsonl manifests "
                             "(build_pope.py output)")
    parser.add_argument("--splits", default="random,popular,adversarial",
                        help="comma-separated subset of: random,popular,adversarial")
    parser.add_argument("--lens-dir", default=DEFAULT_LENS_DIR,
                        help="fitted J-lens dir; its fitted layers are read at the "
                             "answer position (identity fallback over ALL layers when "
                             "it cannot be loaded, recorded in meta.lens)")
    parser.add_argument("--mask", default=MASK, help="which lens-<mask>.pt to read")
    parser.add_argument(
        "--bias", default=DEFAULT_BIAS_PATH,
        help="optional bias-<mask>.pt payload (artifacts.load_bias; T26 calibration: "
             "logit_scale/logit_bias on the lens readout). Missing/unusable -> the arm "
             "runs raw and records the drop; pass '' to force raw",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="max questions per split (default: all)")
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--json", required=True)
    return parser.parse_args()


# ----------------------------------------------------------------- yes/no token ids


def _surface_forms(word: str) -> tuple[str, ...]:
    """The checked surface forms for one answer word (leading-space first)."""
    return (
        f" {word.capitalize()}", f" {word}", word.capitalize(), word,
    )


def _encoded_ids(tokenizer: Any, form: str) -> list[int]:
    """Token ids of one surface form, tolerating the tiny stand-in's tensor return."""
    encoded = tokenizer(form, add_special_tokens=False, return_tensors=None)["input_ids"]
    if torch.is_tensor(encoded):  # the tiny processor returns [1, n] tensors
        encoded = encoded[0].tolist()
    return [int(token_id) for token_id in encoded]


def select_yes_no_ids(model: LensModel, kind: str) -> tuple[int, int, dict[str, Any]]:
    """Resolve the yes/no token ids, tokenizer-agnostically.

    Checked surface forms per word: ' Yes'/' No' (leading-space capitalized -- the
    repo's teacher-forced POPE answer form), ' yes'/' no', 'Yes'/'No', 'yes'/'no'.
    Preferred: the first surface-form PAIR whose yes and no ids both exist, differ, and
    are single tokens (leading-space capitalized on real checkpoints). Fallback (x20's
    canonical-word convention): the FIRST non-whitespace piece of the word's
    leading-space continuation form via ``registry.word_piece_ids`` — the piece the
    model must emit to start the word. Every candidate is recorded in the note.
    """
    forms = {"yes": _surface_forms("yes"), "no": _surface_forms("no")}
    candidates = {
        word: {
            form: {"ids": _encoded_ids(model.tokenizer, form), "single_token": None}
            for form in word_forms
        }
        for word, word_forms in forms.items()
    }
    for word in candidates:
        for entry in candidates[word].values():
            entry["single_token"] = len(entry["ids"]) == 1

    def form_id(word: str, form: str) -> int | None:
        ids = candidates[word][form]["ids"]
        return ids[-1] if ids else None

    chosen: tuple[int, int, int] | None = None
    fallback: tuple[int, int, int] | None = None
    for index in range(len(forms["yes"])):
        yes_id = form_id("yes", forms["yes"][index])
        no_id = form_id("no", forms["no"][index])
        if yes_id is None or no_id is None or yes_id == no_id:
            continue
        both_single = (
            candidates["yes"][forms["yes"][index]]["single_token"]
            and candidates["no"][forms["no"][index]]["single_token"]
        )
        if both_single:
            chosen = (yes_id, no_id, index)
            break
        if fallback is None:
            fallback = (yes_id, no_id, index)
    if chosen is None:
        # x20's canonical-word fallback: first non-whitespace piece of ' ' + word.
        yes_pieces = registry.word_piece_ids(model, kind, "yes")
        no_pieces = registry.word_piece_ids(model, kind, "no")
        if yes_pieces and no_pieces and yes_pieces[0] != no_pieces[0]:
            yes_id, no_id = yes_pieces[0], no_pieces[0]
            note = {
                "rule": "single-token surface-form pair unavailable; first-piece "
                        "fallback (registry.word_piece_ids on ' ' + word, x20's "
                        "canonical-word convention)",
                "yes": {"form": " 'yes' first piece", "id": yes_id,
                        "single_token": False},
                "no": {"form": " 'no' first piece", "id": no_id,
                       "single_token": False},
                "candidates": candidates,
            }
            return yes_id, no_id, note
        chosen = fallback
    if chosen is None:
        raise ValueError(
            "no usable yes/no token pair: every evaluated surface form "
            f"({forms['yes']}) is empty or collides with its 'no' counterpart"
        )
    yes_id, no_id, index = chosen
    note = {
        "rule": "first surface-form pair with both ids present, distinct, and single "
                "tokens (leading-space capitalized preferred); otherwise the x20 "
                "first-piece fallback",
        "yes": {"form": forms["yes"][index], "id": yes_id,
                "single_token": candidates["yes"][forms["yes"][index]]["single_token"]},
        "no": {"form": forms["no"][index], "id": no_id,
               "single_token": candidates["no"][forms["no"][index]]["single_token"]},
        "candidates": candidates,
    }
    return yes_id, no_id, note


# ----------------------------------------------------------------- T26 affine wiring


def _affine_readout_kwargs(payload: dict[str, Any]) -> dict[str, dict[int, Any]]:
    """The payload's logit-side corrections as int-keyed dicts (T26 canonical).

    Delegates to ``vlm_lens.calibrate.affine_readout_kwargs`` when T26's module has
    landed; the local extraction below implements the same documented contract
    (``logit_scale`` float multiplier + ``logit_bias`` tensor per layer str key, old
    entries without logit keys omitted) so the arm does not depend on the module's
    landing order.
    """
    try:
        from vlm_lens.calibrate import affine_readout_kwargs as t26_extract
    except ImportError:
        t26_extract = None
    if t26_extract is not None:
        extracted = t26_extract(payload)
        return {
            "logit_scale": {
                int(k): float(v) for k, v in (extracted.get("logit_scale") or {}).items()
            },
            "logit_bias": {
                int(k): v for k, v in (extracted.get("logit_bias") or {}).items()
            },
        }
    return {
        "logit_scale": {
            int(k): float(entry["logit_scale"])
            for k, entry in payload.items()
            if k != "meta" and "logit_scale" in entry
        },
        "logit_bias": {
            int(k): entry["logit_bias"]
            for k, entry in payload.items()
            if k != "meta" and "logit_bias" in entry
        },
    }


def load_affine_optional(
    args: argparse.Namespace, model: LensModel
) -> tuple[dict[int, torch.Tensor] | None, dict[int, float] | None,
           dict[int, float] | None, dict[int, torch.Tensor] | None,
           dict[int, float] | None, dict[str, Any]]:
    """The optional ``--bias`` payload as the readout's five dicts, or all-None raw.

    Tolerates absence (unset path, missing file, unreadable file) and fails closed on a
    dimension mismatch (x23's drop convention): a payload saved for another checkpoint
    would crash the residual add or the vocab-wide logit add, so the WHOLE payload is
    dropped and the arm runs uncalibrated with the drop recorded.
    """
    raw = {
        "loaded": False, "path": str(args.bias or ""),
        "note": "no --bias path given; the lens readout runs raw" if not args.bias
        else None,
    }
    if not args.bias:
        return None, None, None, None, None, raw
    try:
        payload, payload_meta = load_bias(args.bias)
    except Exception as exc:  # noqa: BLE001 - any load failure takes the raw run
        raw["error"] = f"{exc.__class__.__name__}: {exc}"
        raw["note"] = "bias payload unreadable; the lens readout runs raw"
        print(f"note: no usable bias payload at {args.bias} ({exc}); running raw")
        return None, None, None, None, None, raw
    logit_side = _affine_readout_kwargs(payload)
    bias = {int(k): entry["bias"] for k, entry in payload.items() if k != "meta"}
    scale = {int(k): float(entry["scale"]) for k, entry in payload.items() if k != "meta"}
    temp = {
        int(k): float(entry["temp"])
        for k, entry in payload.items()
        if k != "meta" and "temp" in entry
    }
    logit_bias = logit_side["logit_bias"]
    logit_scale = logit_side["logit_scale"]
    vocab = int(model.unembed_weight().shape[0])
    mismatches = []
    for name, expected, tensors in (
        ("bias", model.d_model, bias.values()),
        ("logit_bias", vocab, logit_bias.values()),
    ):
        wrong = sorted({int(t.shape[-1]) for t in tensors if int(t.shape[-1]) != expected})
        if wrong:
            mismatches.append(f"{name} last dim {wrong} != {expected}")
    if mismatches:
        detail = "; ".join(mismatches)
        print(
            f"note: dropping bias payload at {args.bias} for model kind "
            f"{registry.kind_for(args)!r} (d_model {model.d_model}, vocab {vocab}): "
            f"{detail}; the lens readout runs raw"
        )
        return None, None, None, None, None, {
            **raw,
            "loaded": False,
            "dropped": "dim_mismatch",
            "model_d_model": int(model.d_model),
            "model_vocab": vocab,
            "note": f"payload dims do not match this model; dropped, arm runs raw: {detail}",
        }
    meta = {
        "loaded": True,
        "path": str(args.bias),
        "bias_layers": sorted(bias),
        "scale_layers": sorted(scale),
        "temp_layers": sorted(temp),
        "logit_bias_layers": sorted(logit_bias),
        "logit_scale_layers": sorted(logit_scale),
        "payload_meta": payload_meta,
    }
    return bias, scale, temp, logit_bias, logit_scale, meta


def calibrated_lens_logits(
    model: LensModel,
    lens: JacobianLens,
    hidden: torch.Tensor,
    layer: int,
    aff: "Affine",
) -> torch.Tensor:
    """``lens_readout``'s per-layer math for one cached residual vector.

    Mirrors x14's ``_calibrated_lens_logits`` (transport -> +bias inside the fitted
    branch -> scale -> unembed -> temp) plus T26's ``logit_scale`` slot applied AFTER
    temp as ``z = z / temp * logit_scale + logit_bias`` (the raw multiplier; the
    calibration semantics is z' = scale * x + bias on the lens logits). Model logits
    are never touched. With no affine payload this is the raw ``unembed(J_l @ h_l)``.
    """
    residual = hidden
    if layer in lens.jacobians:
        residual = lens.transport(residual, layer)
        if aff.bias is not None and layer in aff.bias:
            residual = residual + aff.bias[layer].to(residual.device)
    if aff.scale is not None and layer in aff.scale:
        residual = residual * aff.scale[layer]
    logits = model.unembed(residual).float().cpu()
    if aff.temp is not None and layer in aff.temp:
        logits = logits / aff.temp[layer]
    if aff.logit_scale is not None and layer in aff.logit_scale:
        logits = logits * aff.logit_scale[layer]
    if aff.logit_bias is not None and layer in aff.logit_bias:
        logits = logits + aff.logit_bias[layer].to(logits.device)
    return logits


@dataclass(frozen=True)
class Affine:
    """The five optional per-layer correction dicts of the calibrated readout."""

    bias: dict[int, torch.Tensor] | None
    scale: dict[int, float] | None
    temp: dict[int, float] | None
    logit_bias: dict[int, torch.Tensor] | None
    logit_scale: dict[int, float] | None



# --------------------------------------------------------------------------- lens


def load_lens(
    args: argparse.Namespace, model: LensModel
) -> tuple[JacobianLens, dict[str, Any]]:
    """The fitted J-lens on the model's device (x21's convention: the FITTED transport,
    which is also what the T26 calibration payload is fitted against), or the identity
    over ALL layers when ``--lens-dir`` cannot serve one (tiny smoke without a dir, or
    any load failure — recorded, never fatal). The meta carries the fitted layer list
    and the J-content fingerprint, so bias<->lens binding is checkable from the JSON.
    """
    device = model.unembed_weight().device
    source: JacobianLens | None = None
    fallback = ""
    try:
        lenses, _ = load_lens_set(args.lens_dir)
        if args.mask not in lenses:
            fallback = f"mask {args.mask!r} not in {args.lens_dir} (found {sorted(lenses)})"
        else:
            candidate = lenses[args.mask]
            if not candidate.source_layers:
                fallback = f"lens {args.mask!r} in {args.lens_dir} has no fitted layers"
            elif candidate.d_model != model.d_model:
                fallback = (
                    f"lens d_model {candidate.d_model} != model d_model {model.d_model}"
                )
            else:
                source = candidate
    except Exception as exc:  # noqa: BLE001 - any load failure takes the identity
        fallback = f"{exc.__class__.__name__}: {exc}"
    if source is None:
        lens = identity_lens_layers(model, device)
        meta = {
            "loaded": False,
            "dir": str(args.lens_dir),
            "mask": args.mask,
            "fallback": fallback,
            "note": "identity lens over the model's own layers; the readout is the raw "
                    "per-layer unembed",
            "source_layers": list(lens.source_layers),
            "fingerprint": lens_fingerprint(lens),
        }
        return lens, meta
    lens = JacobianLens(
        jacobians={layer: J.to(device) for layer, J in source.jacobians.items()},
        n_prompts=source.n_prompts,
        d_model=source.d_model,
    )
    meta = {
        "loaded": True,
        "dir": str(args.lens_dir),
        "mask": args.mask,
        "n_prompts": int(source.n_prompts),
        "source_layers": list(lens.source_layers),
        "fingerprint": lens_fingerprint(lens),
    }
    return lens, meta


def identity_lens_layers(model: LensModel, device: torch.device) -> JacobianLens:
    """``torch.eye`` JacobianLens over every model layer (the x23 tiny fallback)."""
    return JacobianLens(
        jacobians={
            layer: torch.eye(model.d_model, dtype=torch.float32, device=device)
            for layer in range(model.n_layers)
        },
        n_prompts=0,
        d_model=model.d_model,
    )


# ----------------------------------------------------------------------- workspace


@dataclass
class Microscope:
    """Everything the arm shares: model, lens, affine payload, ids, layer sets."""

    kind: str
    model: LensModel
    lens: JacobianLens
    aff: Affine
    aff_meta: dict[str, Any]
    lens_meta: dict[str, Any]
    layers: list[int]  # fitted (sorted) — the per-layer P_yes readout set
    record_layers: list[int]  # fitted ∪ {final}: the residual capture set
    final_layer: int
    yes_id: int
    no_id: int
    yesno_note: dict[str, Any]
    band_layers: list[int]  # the workspace band ∩ fitted (the mid-band summary set)


def build_microscope(args: argparse.Namespace) -> Microscope:
    """One model load + lens/affine loads + id and layer-set resolution."""
    kind = registry.kind_for(args)
    model = registry.load_lens_model(args)
    device = model.unembed_weight().device
    lens, lens_meta = load_lens(args, model)
    bias, scale, temp, logit_bias, logit_scale, aff_meta = load_affine_optional(args, model)
    aff = Affine(
        bias=None if bias is None else {l: t.to(device) for l, t in bias.items()},
        scale=scale,
        temp=temp,
        logit_bias=(
            None if logit_bias is None else {l: t.to(device) for l, t in logit_bias.items()}
        ),
        logit_scale=logit_scale,
    )
    final_layer = model.n_layers - 1
    layers = sorted(lens.source_layers)
    if final_layer not in layers:
        layers.append(final_layer)
    yes_id, no_id, yesno_note = select_yes_no_ids(model, kind)
    band = [layer for layer in registry.resolve_band(kind, lens) if layer in set(layers)]
    return Microscope(
        kind=kind, model=model, lens=lens, aff=aff, aff_meta=aff_meta,
        lens_meta=lens_meta, layers=layers, record_layers=sorted(set(layers)),
        final_layer=final_layer, yes_id=yes_id, no_id=no_id, yesno_note=yesno_note,
        band_layers=band,
    )


# --------------------------------------------------------------------- measurement


def p_yes_of(logits: torch.Tensor, yes_id: int, no_id: int) -> float:
    """Binary softmax over the yes/no logits: ``sigmoid(z_yes - z_no)`` (stable form)."""
    z_yes = float(logits[yes_id])
    z_no = float(logits[no_id])
    return 1.0 / (1.0 + math.exp(min(500.0, max(-500.0, z_no - z_yes))))


@torch.no_grad()
def answer_position_residuals(
    microscope: Microscope, batch: Any
) -> dict[int, torch.Tensor]:
    """One forward pass; the residual at the FINAL position for every recorded layer.

    Returns ``{layer: [d_model] cpu float32}`` — the answer-position residual the
    per-layer readouts consume (x18's handoff position: the position whose distribution
    predicts the first answer token).
    """
    with ActivationRecorder(
        microscope.model.layers, at=microscope.record_layers
    ) as recorder:
        microscope.model.forward_mm(batch)
    return {
        layer: recorder.activations[layer][0, -1].detach().float().cpu()
        for layer in microscope.record_layers
    }


def measure_question(
    microscope: Microscope, question: str, image: Any, *, max_seq_len: int
) -> dict[str, Any]:
    """One question's full trajectory: model answer + per-layer P_yes with and without
    the image (see the module docstring and the *_RULE notes)."""
    model = microscope.model
    kind = microscope.kind
    prompt = registry.image_prompt(kind, question)
    batch = registry.image_encode(model, kind, prompt, image, max_length=max_seq_len)
    residuals = answer_position_residuals(microscope, batch)

    model_logits = model.unembed(residuals[microscope.final_layer])
    margin = float(model_logits[microscope.yes_id] - model_logits[microscope.no_id])
    answer = "yes" if margin > 0.0 else "no"
    argmax_id = int(torch.argmax(model_logits).item())

    free_batch = registry.chat_encode(model, kind, question, max_length=max_seq_len)
    free_residuals = answer_position_residuals(microscope, free_batch)

    p_yes: dict[str, float] = {}
    p_yes_free: dict[str, float] = {}
    for layer in microscope.layers:
        lens_logits = calibrated_lens_logits(
            model, microscope.lens, residuals[layer], layer, microscope.aff
        )
        free_logits = calibrated_lens_logits(
            model, microscope.lens, free_residuals[layer], layer, microscope.aff
        )
        p_yes[str(layer)] = round(p_yes_of(lens_logits, microscope.yes_id, microscope.no_id), 4)
        p_yes_free[str(layer)] = round(
            p_yes_of(free_logits, microscope.yes_id, microscope.no_id), 4
        )
    gap = {
        layer: round(p_yes[layer] - p_yes_free[layer], 4) for layer in p_yes
    }
    commitment = commitment_layer(p_yes, microscope.layers, answer)
    return {
        "model_answer": answer,
        "margin": round(margin, 4),
        "p_yes_final": round(p_yes_of(model_logits, microscope.yes_id, microscope.no_id), 4),
        "argmax_token": str(model.tokenizer.decode([argmax_id])),
        "n_tokens": int(batch.seq_len),
        "n_tokens_free": int(free_batch.seq_len),
        "p_yes": p_yes,
        "p_yes_free": p_yes_free,
        "gap": gap,
        "commitment_layer": commitment,
    }


def commitment_layer(
    p_yes: dict[str, float], layers: list[int], answer: str
) -> int | None:
    """Smallest layer whose P_yes sits on the answer's side of 0.5 through the end
    (COMMITMENT_RULE); ``None`` when the readout never stabilizes."""
    target_yes = answer == "yes"

    def committed(layer: int) -> bool:
        value = p_yes[str(layer)]
        return value > 0.5 if target_yes else value < 0.5

    for index, layer in enumerate(layers):
        if all(committed(later) for later in layers[index:]):
            return layer
    return None


def aligned_confidence(p_yes: float, answer: str) -> float:
    """Answer-aligned P_yes as a CONFIDENCE: itself for a yes answer, 1 - P_yes for a
    no answer (ERROR_AUC_RULE's feature (a) alignment)."""
    return p_yes if answer == "yes" else 1.0 - p_yes


def aligned_sign(value: float, answer: str) -> float:
    """Answer-aligned SIGN: the value as-is for a yes answer, negated for no (the gap's
    alignment: positive = the image supports the committed answer)."""
    return value if answer == "yes" else -value


def error_auc_pyes(
    p_yes: list[float], answers: list[str], correct: list[bool]
) -> float | None:
    """Rank AUC of ``1 - aligned confidence`` for predicting model error (positive
    group = wrong); ``None`` without both groups (ERROR_AUC_RULE)."""
    wrong = [
        1.0 - aligned_confidence(p, a)
        for p, a, ok in zip(p_yes, answers, correct) if not ok
    ]
    right = [
        1.0 - aligned_confidence(p, a)
        for p, a, ok in zip(p_yes, answers, correct) if ok
    ]
    return auroc(wrong, right)


def error_auc_gap(
    gaps: list[float], answers: list[str], correct: list[bool]
) -> float | None:
    """Rank AUC of ``-answer-aligned gap`` for predicting model error; ``None``
    without both groups (ERROR_AUC_RULE)."""
    wrong = [
        -aligned_sign(g, a) for g, a, ok in zip(gaps, answers, correct) if not ok
    ]
    right = [
        -aligned_sign(g, a) for g, a, ok in zip(gaps, answers, correct) if ok
    ]
    return auroc(wrong, right)


# ---------------------------------------------------------------------- aggregation


def _mean_or_none(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _prf1(answers: list[str], labels: list[str]) -> dict[str, float | None]:
    """Yes-class precision/recall/F1 plus accuracy (POPE's scoreboard convention)."""
    n = len(labels)
    tp = sum(1 for a, l in zip(answers, labels) if a == "yes" and l == "yes")
    fp = sum(1 for a, l in zip(answers, labels) if a == "yes" and l == "no")
    fn = sum(1 for a, l in zip(answers, labels) if a == "no" and l == "yes")
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    accuracy = (
        sum(1 for a, l in zip(answers, labels) if a == l) / n if n else None
    )
    return {
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "precision_yes": round(precision, 4) if precision is not None else None,
        "recall_yes": round(recall, 4) if recall is not None else None,
        "f1_yes": round(f1, 4) if f1 is not None else None,
    }


def split_layer_table(
    microscope: Microscope, questions: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Per fitted layer: the gap curve, the label separation, and the error AUCs.

    Only labeled questions (label in {yes, no}) enter the label-separation and
    error-AUC columns; the means cover every measured question.
    """
    scored = [q for q in questions if q["label"] in ("yes", "no")]
    table: dict[str, dict[str, Any]] = {}
    for layer in microscope.layers:
        key = str(layer)
        p_label_yes = [q["p_yes"][key] for q in scored if q["label"] == "yes"]
        p_label_no = [q["p_yes"][key] for q in scored if q["label"] == "no"]
        table[key] = {
            "mean_p_yes": _mean_or_none([q["p_yes"][key] for q in questions]),
            "mean_p_yes_free": _mean_or_none([q["p_yes_free"][key] for q in questions]),
            "mean_gap": _mean_or_none([q["gap"][key] for q in questions]),
            "mean_p_yes_label_yes": _mean_or_none(p_label_yes),
            "mean_p_yes_label_no": _mean_or_none(p_label_no),
            "auroc_yes_vs_no": auroc(p_label_yes, p_label_no),
            "auc_error_pyes": error_auc_pyes(
                [q["p_yes"][key] for q in scored],
                [q["model_answer"] for q in scored],
                [q["correct"] for q in scored],
            ),
            "auc_error_gap": error_auc_gap(
                [q["gap"][key] for q in scored],
                [q["model_answer"] for q in scored],
                [q["correct"] for q in scored],
            ),
        }
    return table


def split_summary(
    microscope: Microscope,
    questions: list[dict[str, Any]],
    table: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Accuracy/F1, yes-rate bias, commitment histogram, and the error-AUC digest."""
    scored = [q for q in questions if q["label"] in ("yes", "no")]
    answers = [q["model_answer"] for q in scored]
    labels = [q["label"] for q in scored]
    scoreboard = _prf1(answers, labels)
    commitment_counts: dict[str, int] = {"none": 0}
    for q in scored:
        layer = q["commitment_layer"]
        commitment_counts[str(layer) if layer is not None else "none"] = (
            commitment_counts.get(str(layer) if layer is not None else "none", 0) + 1
        )
    best = {"auc": None, "layer": None, "feature": None}
    mid_best = {"auc": None, "layer": None, "feature": None}
    band = set(microscope.band_layers)
    for layer in microscope.layers:
        key = str(layer)
        for feature, value in (
            ("pyes", table[key]["auc_error_pyes"]),
            ("gap", table[key]["auc_error_gap"]),
        ):
            if value is None:
                continue
            if best["auc"] is None or value > best["auc"]:
                best = {"auc": value, "layer": layer, "feature": feature}
            if layer in band and (mid_best["auc"] is None or value > mid_best["auc"]):
                mid_best = {"auc": value, "layer": layer, "feature": feature}
    margins = [-abs(q["margin"]) for q in scored]
    finals = [
        1.0 - aligned_confidence(q["p_yes_final"], q["model_answer"]) for q in scored
    ]
    wrong = [not q["correct"] for q in scored]
    baseline = {
        "auc_margin_abs": auroc(
            [m for m, w in zip(margins, wrong) if w],
            [m for m, w in zip(margins, wrong) if not w],
        ),
        "auc_pyes_final": auroc(
            [f for f, w in zip(finals, wrong) if w],
            [f for f, w in zip(finals, wrong) if not w],
        ),
    }
    baseline["best"] = max(
        (v for k, v in baseline.items() if k != "best" and v is not None),
        default=None,
    )
    return {
        "n_questions": len(scored),
        "n_unlabeled": len(questions) - len(scored),
        "n_yes_label": sum(1 for l in labels if l == "yes"),
        "n_no_label": sum(1 for l in labels if l == "no"),
        "model_yes_rate": (
            round(sum(1 for a in answers if a == "yes") / len(answers), 4)
            if answers else None
        ),
        "gt_yes_rate": (
            round(sum(1 for l in labels if l == "yes") / len(labels), 4)
            if labels else None
        ),
        "model_answer_counts": {
            verdict: sum(1 for a in answers if a == verdict)
            for verdict in ("yes", "no")
        },
        "mean_margin": _mean_or_none([abs(q["margin"]) for q in scored]),
        **scoreboard,
        "commitment_histogram": commitment_counts,
        "n_committed": len(scored) - commitment_counts.get("none", 0),
        "error_auc": {
            "best": best,
            "mid_band": {
                "layers": microscope.band_layers,
                "best": mid_best,
                "note": "workspace band ∩ fitted layers (registry.resolve_band); None "
                        "when empty",
            },
            "final_logit_baseline": baseline,
            "rule": ERROR_AUC_RULE,
        },
    }


# ---------------------------------------------------------------------------- main


def run_split(
    args: argparse.Namespace, microscope: Microscope, split: str
) -> dict[str, Any]:
    """One split end to end: open-manifest questions -> trajectories -> aggregates."""
    manifest = Path(args.pope_dir) / f"{split}-open.jsonl"
    samples = read_manifest(manifest)
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError(f"--limit must be >= 0, got {args.limit}")
        samples = samples[: args.limit]
    questions: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        meta = sample.meta or {}
        question = str(meta.get("question") or "")
        label = str(meta.get("label", "")).strip().lower()
        if not sample.images:
            raise ValueError(
                f"{manifest} sample {sample.sample_id!r} carries no image path"
            )
        measurement = measure_question(
            microscope, question, sample.images[0], max_seq_len=args.max_seq_len
        )
        questions.append(
            {
                "question_id": meta.get("question_id", sample.sample_id),
                "image": meta.get("image") or Path(sample.images[0]).name,
                "question": question,
                "label": label,
                "correct": (
                    measurement["model_answer"] == label if label in ("yes", "no") else None
                ),
                **measurement,
            }
        )
        if (index + 1) % 25 == 0 or index + 1 == len(samples):
            print(
                f"  [{split} {index + 1}/{len(samples)}] answered="
                f"{measurement['model_answer']} margin={measurement['margin']}",
                flush=True,
            )
    table = split_layer_table(microscope, questions)
    return {
        "summary": split_summary(microscope, questions, table),
        "gap_curve": [
            {"layer": layer, **{k: table[str(layer)][k] for k in
                                ("mean_p_yes", "mean_p_yes_free", "mean_gap")}}
            for layer in microscope.layers
        ],
        "per_layer": table,
        "questions": questions,
    }


def main() -> int:
    args = parse_args()
    splits = [s for s in (part.strip() for part in args.splits.split(",")) if s]
    unknown = [s for s in splits if s not in ("random", "popular", "adversarial")]
    if unknown:
        raise ValueError(f"unknown splits {unknown} (expected random/popular/adversarial)")
    microscope = build_microscope(args)

    yes = microscope.yesno_note
    print(
        f"model ready: kind={microscope.kind} layers={microscope.model.n_layers} "
        f"d_model={microscope.model.d_model} backend={args.backend}; "
        f"lens fitted {microscope.lens_meta['source_layers']} "
        f"(loaded={microscope.lens_meta['loaded']}); calibrated="
        f"{microscope.aff_meta['loaded']}; yes/no ids "
        f"{microscope.yes_id}/{microscope.no_id} ({yes['yes']['form']!r}/"
        f"{yes['no']['form']!r})"
    )
    print(f"== X24 POPE microscope: splits={splits} limit={args.limit} ==")

    results = {split: run_split(args, microscope, split) for split in splits}
    digest_lines = []
    for split in splits:
        summary = results[split]["summary"]
        digest_lines.append(
            f"[{split}] acc={summary['accuracy']} f1={summary['f1_yes']} "
            f"yes_rate(model/gt)={summary['model_yes_rate']}/{summary['gt_yes_rate']} "
            f"best_error_auc={summary['error_auc']['best']} "
            f"baseline={summary['error_auc']['final_logit_baseline']['best']}"
        )
        digest_lines.append(
            f"{'layer':>6}{'p_yes':>8}{'p_free':>8}{'gap':>8}{'au_y/n':>8}"
            f"{'auc_p':>8}{'auc_g':>8}"
        )
        for layer in microscope.layers:
            row = results[split]["per_layer"][str(layer)]
            cells = "".join(
                f"{row[c]:>8}" if row[c] is not None else f"{'-':>8}"
                for c in ("mean_p_yes", "mean_p_yes_free", "mean_gap",
                          "auroc_yes_vs_no", "auc_error_pyes", "auc_error_gap")
            )
            digest_lines.append(f"L{layer:<5}{cells}")

    report: dict[str, Any] = {
        "script": Path(__file__).name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": args.backend,
        "model": {
            "kind": microscope.kind,
            "model_id": registry.MODEL_SPECS[microscope.kind]["model_id"],
            "n_layers": microscope.model.n_layers,
            "d_model": microscope.model.d_model,
        },
        "pope_dir": str(args.pope_dir),
        "splits": splits,
        "limit_per_split": args.limit,
        "max_seq_len": args.max_seq_len,
        "prompt_template": registry.image_prompt_template(microscope.kind),
        "prompt_template_hash": registry.prompt_template_sha256(microscope.kind),
        "lens": {
            **microscope.lens_meta,
            "fingerprint_note": (
                "sha256 over the loaded lens's fitted J_l values (x21 pattern): equal "
                "metrics with equal fingerprints = the same fitted lens was read"
            ),
        },
        "bias": {
            **microscope.aff_meta,
            "calibration_note": (
                "T26 affine calibration on the lens readout only "
                "(z = unembed(transport(h_l)) / temp * logit_scale + logit_bias); the "
                "model's own logits are never touched"
                if microscope.aff_meta.get("loaded") else None
            ),
            "base_lens_fingerprint": (
                (microscope.aff_meta.get("payload_meta") or {}).get("base_lens", {})
                .get("fingerprint")
                if microscope.aff_meta.get("loaded") else None
            ),
        },
        "yesno": microscope.yesno_note,
        "rules": {
            "p_yes": P_YES_RULE,
            "gap": GAP_RULE,
            "image_free": IMAGE_FREE_NOTE,
            "commitment": COMMITMENT_RULE,
            "error_auc": ERROR_AUC_RULE,
            "baseline": BASELINE_RULE,
        },
        "results": {split: results[split] for split in splits},
        "digest": "\n".join(digest_lines),
        "interpretation_limits": [
            IMAGE_FREE_NOTE,
            "the commitment layer is defined over the FITTED layer set only; layers "
            "between fitted layers are invisible to it",
            "error AUCs need both correct and wrong questions; a split the model "
            "aces (or fails) completely reports None there",
            TINY_NOTE if args.backend == "tiny" else
            "the tiny fixture note does not apply to this run",
        ],
    }
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + "\n".join(digest_lines))
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
