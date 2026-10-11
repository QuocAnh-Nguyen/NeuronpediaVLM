#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Single dispatch point for the x20-x23 validation scripts' model conventions.

Every checkpoint-specific constant the four scripts used to hardwire lives here: the
model id and wrapper class per ``--model`` kind, the workspace band in layers, the
chat convention, the vision-projector module, and the leading-space tokenization
convention. The scripts keep their own flags (``--layers``/``--band``/``--layer-lo``,
``--device``, ``--dtype``) as overrides; the registry supplies the defaults.

Kinds (``MODEL_SPECS``):

* ``llava-7b``  -- llava-hf/llava-1.5-7b-hf, 32 layers, workspace band [12, 29]
  (the default; byte-for-byte the scripts' previous behavior).
* ``llava-13b`` -- llava-hf/llava-1.5-13b-hf, 40 layers, band [15, 36] (same
  ``LlavaLensModel`` wrapper class, same llava-1.5 chat convention).
* ``qwen2vl``   -- Qwen/Qwen2-VL-7B-Instruct, 28 layers, band [11, 26] (the
  ``Qwen2VLLensModel`` wrapper in ``src/vlm_lens/models/qwen2vl.py``).

Qwen2-VL chat convention: the wrapper's ``encode_mm`` applies
``processor.apply_chat_template`` to the RAW body itself (``<image>`` placeholders
spell the image slots, exactly as in llava), so for qwen2vl the registry passes the
body through un-wrapped -- pre-wrapping would double-template. The wrapper renders
the user turn only (``add_generation_prompt=False``; the generation scaffold is the
caller's choice, as the llava prompt string carries ``ASSISTANT:`` itself), so the
registry's generation-capable encoders (:func:`chat_encode`, :func:`image_encode`)
append the Qwen2 assistant header ``<|im_start|>assistant\\n`` at the token level.

``--backend tiny`` implies llava-7b geometry: the tiny CPU fixture is llava-only and
the other kinds have no tiny stand-in (x20-x23 deliberately build no tiny qwen2vl
path), so :func:`kind_for` collapses tiny to ``llava-7b``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
from typing import Any

import torch

from vlm_lens.data.captions import PROMPT_TEMPLATE as LLAVA_PROMPT_TEMPLATE
from vlm_lens.models.llava import LlavaLensModel

DEFAULT_KIND = "llava-7b"

#: kind -> checkpoint identity + geometry. ``cls_name`` resolves through
#: ``_CLASS_MODULES`` (lazy, so an unbuilt qwen2vl module never breaks llava runs).
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "llava-7b": {
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "cls_name": "LlavaLensModel",
        "image_token_id": 32000,
        "band_lo": 12,
        "band_hi": 29,
        "n_layers": 32,
    },
    "llava-13b": {
        "model_id": "llava-hf/llava-1.5-13b-hf",
        "cls_name": "LlavaLensModel",
        "image_token_id": 32000,
        "band_lo": 15,
        "band_hi": 36,
        "n_layers": 40,
    },
    "qwen2vl": {
        "model_id": "Qwen/Qwen2-VL-7B-Instruct",
        "cls_name": "Qwen2VLLensModel",
        "image_token_id": 151655,
        "band_lo": 11,
        "band_hi": 26,
        "n_layers": 28,
    },
}

#: The ``--model`` choices, in MODEL_SPECS order.
MODEL_KINDS: tuple[str, ...] = tuple(MODEL_SPECS)

#: cls_name -> (module, attribute); imported on demand in :func:`spec_class`.
_CLASS_MODULES: dict[str, tuple[str, str]] = {
    "LlavaLensModel": ("vlm_lens.models.llava", "LlavaLensModel"),
    "Qwen2VLLensModel": ("vlm_lens.models.qwen2vl", "Qwen2VLLensModel"),
}

#: The Qwen2 chat rendering of one text-only user turn -- what
#: ``Qwen2VLLensModel.encode_mm`` produces from the raw body (documented here for
#: provenance; the wrapper's ``processor.apply_chat_template`` is the authority).
QWEN_TEXT_TEMPLATE = "<|im_start|>user\n{body}<|im_end|>\n"

#: The Qwen2 assistant header a generation prompt must end with (the template's
#: ``add_generation_prompt=True`` tail); appended token-level by
#: :func:`_with_generation_prompt`.
QWEN_GENERATION_SUFFIX = "<|im_start|>assistant\n"


# ------------------------------------------------------------------ dispatch


def kind_for(args: argparse.Namespace) -> str:
    """The dispatch kind for a parsed arg namespace: ``--model``, tiny forced to llava.

    ``--backend tiny`` implies llava-7b geometry (the tiny fixture is llava-only),
    whatever ``--model`` says.
    """
    if getattr(args, "backend", "hf-llava") == "tiny":
        return DEFAULT_KIND
    return getattr(args, "model", DEFAULT_KIND)


def spec_class(kind: str) -> type:
    """The wrapper class for ``kind``, imported lazily so an unbuilt qwen2vl module
    only errors when actually dispatched."""
    cls_name = MODEL_SPECS[kind]["cls_name"]
    module_name, attr = _CLASS_MODULES[cls_name]
    return getattr(importlib.import_module(module_name), attr)


def resolve_device(args: argparse.Namespace) -> str:
    """The scripts' HF-path device conventions, unified.

    x21/x23 hardwire cuda (no ``--device`` flag); x20 defaults '' -> cuda with an
    override; x22's 'auto' resolves to cuda when available, else cpu.
    """
    device = getattr(args, "device", None)
    if device in (None, ""):
        return "cuda"
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


def resolve_dtype(args: argparse.Namespace, device: str) -> torch.dtype:
    """The scripts' HF-path dtype conventions, unified.

    x22's ``--dtype`` flag wins when present (bfloat16/float32); otherwise the repo
    default: bf16 on cuda, fp32 elsewhere.
    """
    flag = getattr(args, "dtype", None)
    if flag == "bfloat16":
        return torch.bfloat16
    if flag == "float32":
        return torch.float32
    return torch.bfloat16 if str(device).startswith("cuda") else torch.float32


def load_lens_model(args: argparse.Namespace) -> Any:
    """The scripts' model loader: ``--model`` dispatch (default llava-7b).

    ``--backend tiny`` builds the tiny llava CPU fixture exactly as before. The HF
    path loads the kind's checkpoint through its wrapper class with each script's
    own dtype/device convention (:func:`resolve_dtype`/:func:`resolve_device`) and
    ``local_files_only=True``. The wrapper's ``image_token_id`` is checked against
    the spec so checkpoint/wrapper drift fails loudly at load.

    Returns:
        The lens-model instance (``LlavaLensModel`` for the llava kinds;
        ``Qwen2VLLensModel`` for qwen2vl -- same public surface).
    """
    if getattr(args, "backend", "hf-llava") == "tiny":
        from vlm_lens.models.tiny_llava import TinyLlavaConfig, build_tiny_llava

        hf_model, processor = build_tiny_llava(TinyLlavaConfig())
        return LlavaLensModel(hf_model, processor)
    kind = kind_for(args)
    spec = MODEL_SPECS[kind]
    device = resolve_device(args)
    dtype = resolve_dtype(args, device)
    model = spec_class(kind).from_pretrained(
        spec["model_id"], dtype=dtype, device=device, local_files_only=True
    )
    expected_id = int(spec["image_token_id"])
    if int(getattr(model, "image_token_id", expected_id)) != expected_id:
        raise ValueError(
            f"{kind}: wrapper image_token_id {model.image_token_id!r} != spec "
            f"{expected_id} for {spec['model_id']}"
        )
    return model


# ------------------------------------------------------------------ chat template


def chat_wrap(kind: str, body: str) -> str:
    """The ``encode_mm`` prompt string for one text-only user turn.

    llava: ``USER: {body}\\nASSISTANT:`` (the repo's llava-1.5 convention, x21's old
    CHAT_USER/CHAT_ASSISTANT). qwen2vl: the RAW body, passed through -- the wrapper's
    ``encode_mm`` applies ``processor.apply_chat_template`` itself (rendering
    ``QWEN_TEXT_TEMPLATE``), and pre-wrapping here would double-template. The qwen2vl
    generation scaffold is appended token-level by :func:`chat_encode`, not in the
    string (see :func:`_with_generation_prompt`).
    """
    if kind == "qwen2vl":
        return body
    return f"USER: {body}\nASSISTANT:"


def chat_encode(
    model: Any, kind: str, body: str, *, max_length: int | None = None
) -> Any:
    """Encode a text-only chat turn through the model's canonical path.

    llava: the wrapped string through ``encode_mm``, exactly as x21 did inline.
    qwen2vl: the raw body through the wrapper (its template), then the assistant
    header appended token-level so generation starts at the answer position.
    """
    batch = model.encode_mm(chat_wrap(kind, body), None, max_length=max_length)
    if kind == "qwen2vl":
        batch = _with_generation_prompt(model, batch)
    return batch


def chat_turn_frames(kind: str) -> dict[str, str]:
    """Frame pieces for segmented multi-turn assembly (x21's exact-span prompts).

    Keys map onto x21's ``_segment_prompt`` pieces: ``user_open`` (before the first
    user turn), ``assistant_open`` (after each user turn), ``turn_sep`` (between
    assistant and the next user turn), ``assistant_close`` (final assistant header).
    The llava values are byte-identical to x21's old CHAT_USER/CHAT_ASSISTANT
    composition. The qwen2vl frames render the same multi-turn template the
    processor would (eos ``<|im_end|>`` included as a literal piece the tokenizer
    maps to its eos id, mirroring the llava path's ``</s>`` handling), ending with
    the assistant header so the segment path is generation-ready like
    :func:`chat_encode`.
    """
    if kind == "qwen2vl":
        return {
            "user_open": "<|im_start|>user\n",
            "assistant_open": "<|im_end|>\n<|im_start|>assistant\n",
            "turn_sep": "<|im_end|>\n<|im_start|>user\n",
            "assistant_close": "<|im_end|>\n<|im_start|>assistant\n",
        }
    return {
        "user_open": "USER: ",
        "assistant_open": "\nASSISTANT: ",
        "turn_sep": "</s>\nUSER: ",
        "assistant_close": "\nASSISTANT:",
    }


def image_prompt_template(kind: str) -> str:
    """The COCO-captioning image-prompt skeleton with a ``{question}`` slot, as the
    model sees it (the V8 provenance string).

    llava: the repo's ``vlm_lens.data.captions.PROMPT_TEMPLATE`` verbatim, so its
    hash stays consistent. qwen2vl: the rendered Qwen2 user turn with the
    ``<image>`` slot before the question and the assistant header (what the wrapper
    produces from the :func:`image_prompt` body plus :func:`image_encode`'s
    scaffold).
    """
    if kind == "qwen2vl":
        return "<|im_start|>user\n<image>{question}<|im_end|>\n<|im_start|>assistant\n"
    return LLAVA_PROMPT_TEMPLATE


def image_prompt(kind: str, question: str) -> str:
    """The ``encode_mm`` prompt (body) for one image question (x13's prompt_text,
    kind-aware). llava carries the full chat scaffold in the string; qwen2vl is the
    raw ``<image>``-placeholder body the wrapper templates."""
    if kind == "qwen2vl":
        return f"<image>{question}"
    return image_prompt_template(kind).format(question=question)


def image_encode(
    model: Any, kind: str, prompt: str, images: Any, *, max_length: int | None = None
) -> Any:
    """Encode an image prompt through the model's canonical path.

    llava: ``encode_mm(prompt, images)`` verbatim (the string carries ``ASSISTANT:``).
    qwen2vl: the raw body through the wrapper, then the assistant header appended
    token-level (:func:`_with_generation_prompt`) so the prefill's handoff position
    is the about-to-answer state, like llava's.
    """
    batch = model.encode_mm(prompt, images, max_length=max_length)
    if kind == "qwen2vl":
        batch = _with_generation_prompt(model, batch)
    return batch


def prompt_template_sha256(kind: str) -> str:
    """SHA-256 of the kind's rendered prompt skeleton (the V8 provenance rule)."""
    return hashlib.sha256(image_prompt_template(kind).encode("utf-8")).hexdigest()


def _with_generation_prompt(model: Any, batch: Any) -> Any:
    """Append the Qwen2 assistant-header tokens to an encoded qwen2vl batch.

    ``Qwen2VLLensModel.encode_mm`` renders the user turn only
    (``apply_chat_template(..., add_generation_prompt=False)`` -- the scaffold is
    the caller's choice, as the llava prompt string carries ``ASSISTANT:`` itself),
    so generation-capable callers append ``QWEN_GENERATION_SUFFIX`` here: the exact
    tokens the template's generation prompt would add, resolved through the
    wrapper's own tokenizer, spliced onto ``input_ids``/``attention_mask`` (and
    ``mm_token_type_ids`` as text labels) via a frozen-dataclass copy. Text-only
    batches carry no ``mm_token_type_ids`` and stay None. The 3-token suffix is
    added after ``encode_mm``'s length guard, so ``--max-seq-len`` effectively has
    a 3-token slack on qwen2vl.
    """
    suffix_ids = model.tokenizer(
        QWEN_GENERATION_SUFFIX, add_special_tokens=False, return_tensors=None
    )["input_ids"]
    suffix = torch.tensor(
        [suffix_ids], dtype=batch.input_ids.dtype, device=batch.input_ids.device
    )
    fields: dict[str, Any] = {
        "input_ids": torch.cat([batch.input_ids, suffix], dim=1),
        "attention_mask": torch.cat(
            [batch.attention_mask, torch.ones_like(suffix)], dim=1
        ),
        "text": batch.text + QWEN_GENERATION_SUFFIX,
    }
    labels = getattr(batch, "mm_token_type_ids", None)
    if labels is not None:
        zeros = torch.zeros(
            (labels.shape[0], suffix.shape[1]),
            dtype=labels.dtype,
            device=labels.device,
        )
        fields["mm_token_type_ids"] = torch.cat([labels, zeros], dim=1)
    return dataclasses.replace(batch, **fields)


# ------------------------------------------------------------------ vision side


def vision_projector(model: Any, kind: str) -> torch.nn.Module:
    """The module whose OUTPUT is the projected visual features merged into the LM.

    llava: ``hf_model.model.multi_modal_projector`` (its output is scattered into
    the input embeddings by ``LlavaModel.forward``). qwen2vl: ``hf_model.visual``,
    the vision tower whose output (the merged embeds) is the same-role tensor.
    Hooks on the returned module therefore capture/blend the identical injection
    point for both kinds (x23's projector capture and mixer).
    """
    if kind == "qwen2vl":
        return model.hf_model.visual
    return model.hf_model.model.multi_modal_projector


def vision_only(model: Any, kind: str, batch: Any) -> None:
    """Run the standalone vision path (no LM forward) so ``vision_projector`` hooks
    fire, yielding the projected features for the batch's image.

    llava: ``LlavaModel.get_image_features`` (tower + feature selection +
    projector), the call x23 used inline. qwen2vl:
    ``Qwen2VLModel.get_image_features`` (tower + merger), which needs the
    processor's per-image ``image_grid_thw`` -- read from the wrapper's batch
    (``batch.image_grid_thw``; llava-built batches carry None, which fails with a
    clear error here).
    """
    hf_model = model.hf_model
    if kind == "qwen2vl":
        grid = getattr(batch, "image_grid_thw", None)
        if grid is None:
            raise TypeError(
                "qwen2vl vision-only capture needs the batch's image_grid_thw (the "
                "processor's per-image grid); encode the image through "
                "Qwen2VLLensModel.encode_mm so the batch carries it"
            )
        hf_model.model.get_image_features(pixel_values=batch.pixel_values, image_grid_thw=grid)
        return
    hf_model.model.get_image_features(pixel_values=batch.pixel_values, return_dict=True)


# ------------------------------------------------------------------ tokenization


def word_piece_ids(model: Any, kind: str, word: str) -> list[int]:
    """Non-whitespace piece ids of a word's leading-space continuation form.

    llava (SentencePiece): the tokenizer emits the leading space as its own
    whitespace piece (``' Brazil' -> [29871, 'Brazil-piece']``), so whitespace-only
    pieces are dropped before judging. qwen2vl (byte-level BPE): ``' ' + word``
    merges into the word's own token(s) with no standalone whitespace piece, so a
    single-token form is returned as-is; multi-piece forms fall back to the llava
    shape (drop whitespace-only pieces). The tiny fixture's word-hash tokenizer
    yields exactly one id per word.
    """
    tokenizer = model.tokenizer
    if hasattr(tokenizer, "encode_text"):  # TinyLlavaProcessor: one id per word
        return [int(tokenizer.encode_text(word)[1])]
    pieces = tokenizer(" " + word, add_special_tokens=False, return_tensors=None)["input_ids"]
    ids = [int(piece) for piece in pieces]
    if kind == "qwen2vl" and len(ids) == 1:
        return ids
    return [piece for piece in ids if tokenizer.decode([piece]).strip()]


# ------------------------------------------------------------------ band


def spec_band(kind: str) -> tuple[int, int]:
    """(band_lo, band_hi): the workspace band in layers, from MODEL_SPECS."""
    spec = MODEL_SPECS[kind]
    return int(spec["band_lo"]), int(spec["band_hi"])


def resolve_band(kind: str, lens: Any | None = None) -> list[int]:
    """The workspace band's layers, clamped to the lens's fitted layers when given.

    Returns the possibly-EMPTY intersection; callers own their empty-band fallback
    (x21: all fitted layers; x22: the fitted middle half). With ``lens=None`` the
    raw spec band is returned.
    """
    lo, hi = spec_band(kind)
    band = list(range(lo, hi + 1))
    if lens is None:
        return band
    fitted = set(lens.source_layers)
    return [layer for layer in band if layer in fitted]


def band_layers(kind: str, n_layers: int) -> tuple[int, ...]:
    """Four stride-4 readout layers inside the band (x23 report/modulation default).

    ``band_lo + 4k`` for k in 1..4 while inside the band: llava-7b reproduces the
    old hardcoded ``(16, 20, 24, 28)`` exactly.
    """
    lo, hi = spec_band(kind)
    return tuple(lo + 4 * step for step in range(1, 5) if lo + 4 * step <= hi)


def sweep_layers(kind: str, n_layers: int) -> tuple[int, ...]:
    """The wider stride-4 grid from band_lo to the final layer (x23 ignition/band).

    llava-7b reproduces the old hardcoded ``(12, 16, 20, 24, 28, 31)`` exactly:
    stride-4 from band_lo, with the final layer always included (it is the readout
    layer's own block).
    """
    lo, _hi = spec_band(kind)
    grid = list(range(lo, n_layers, 4))
    final = n_layers - 1
    if grid[-1] != final:
        grid.append(final)
    return tuple(grid)


__all__ = [
    "DEFAULT_KIND",
    "MODEL_KINDS",
    "MODEL_SPECS",
    "QWEN_GENERATION_SUFFIX",
    "QWEN_TEXT_TEMPLATE",
    "band_layers",
    "chat_encode",
    "chat_turn_frames",
    "chat_wrap",
    "image_encode",
    "image_prompt",
    "image_prompt_template",
    "kind_for",
    "load_lens_model",
    "prompt_template_sha256",
    "resolve_band",
    "resolve_device",
    "resolve_dtype",
    "spec_band",
    "spec_class",
    "sweep_layers",
    "vision_only",
    "vision_projector",
    "word_piece_ids",
]
