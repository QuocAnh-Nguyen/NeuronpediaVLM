# SPDX-License-Identifier: Apache-2.0
"""Multimodal ``LensModel`` over HuggingFace LLaVA (``llava-hf/llava-1.5-7b-hf``).

Design notes
------------
The Jacobian lens is defined on the *LLM* residual stream: the vision tower and the
projector are frozen input providers that merge their output into the input embeddings
before block 0. We therefore do **not** reimplement the image->embedding fusion. The
forward pass calls the model's own :class:`LlavaModel` (``hf_model.model``), which runs
``get_image_features`` -> ``get_placeholder_mask`` -> ``masked_scatter`` -> LM stack, so
token counts, dtypes and masking are exactly HuggingFace's. We only add what the lens
needs: tokenization that keeps image placeholders addressable, modality masks, and a
``[d_model, d_model]``-friendly forward that stops before the LM head.

Because ``LlavaModel.forward`` is used verbatim, ``scripts/check_equivalence.py`` can
assert bit-level agreement against ``LlavaForConditionalGeneration.forward``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from jlens.hf import HFLensModel
from jlens.hooks import ActivationRecorder
from PIL import Image

IMAGE_PLACEHOLDER = "<image>"


@dataclass(frozen=True)
class MultimodalBatch:
    """One encoded sample (batch size 1; the fit loop expands along the batch axis)."""

    input_ids: torch.Tensor  # [1, seq_len] long
    attention_mask: torch.Tensor  # [1, seq_len] long
    pixel_values: torch.Tensor | None  # [n_images, 3, H, W] on the vision device
    image_token_mask: torch.Tensor  # [1, seq_len] bool
    text: str

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.shape[1])

    @property
    def n_image_tokens(self) -> int:
        return int(self.image_token_mask.sum())

    def to(self, device: torch.device | str) -> MultimodalBatch:
        return MultimodalBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            pixel_values=None if self.pixel_values is None else self.pixel_values.to(device),
            image_token_mask=self.image_token_mask.to(device),
            text=self.text,
        )

    def hf_kwargs(self) -> dict[str, torch.Tensor]:
        """Keyword arguments for ``LlavaForConditionalGeneration.forward`` / ``generate``."""
        kwargs: dict[str, torch.Tensor] = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }
        if self.pixel_values is not None:
            kwargs["pixel_values"] = self.pixel_values
        return kwargs

    def expand(self, n: int) -> MultimodalBatch:
        """Replicate the sample along the batch axis (the fit's ``dim_batch`` trick).

        Views only — no data is copied; the fit relies on every batch element being the
        identical prompt. HuggingFace flattens images across the batch
        (``[n_images_total, 3, H, W]``), so replication is only expressible when each
        sample carries at most one image: with several images, fit at ``dim_batch=1``.
        """
        if n == 1:
            return self
        if self.pixel_values is not None and self.pixel_values.shape[0] != 1:
            raise ValueError(
                f"dim_batch replication needs one image per sample, got "
                f"{self.pixel_values.shape[0]} images; use dim_batch=1 for multi-image samples"
            )
        return MultimodalBatch(
            input_ids=self.input_ids.expand(n, -1),
            attention_mask=self.attention_mask.expand(n, -1),
            pixel_values=(
                None
                if self.pixel_values is None
                else self.pixel_values.expand(n, *self.pixel_values.shape[1:])
            ),
            image_token_mask=self.image_token_mask.expand(n, -1),
            text=self.text,
        )


def resolve_image_seq_length(config: Any) -> tuple[int, str]:
    """LM tokens one image expands to, and where the number came from (V10).

    The merged ``config.json`` does not carry ``image_seq_length`` (F5): the value is a
    class default of the installed ``transformers``. A missing or zero value silently
    disabled :meth:`LlavaLensModel.encode_mm`'s placeholder guard (fail-open, F6), so
    derive it from the vision geometry instead and raise when even that is unavailable.
    """
    merged = int(getattr(config, "image_seq_length", 0) or 0)
    if merged > 0:
        return merged, "config"
    vision = getattr(config, "vision_config", None)
    image_size = int(getattr(vision, "image_size", 0) or 0)
    patch_size = int(getattr(vision, "patch_size", 0) or 0)
    if image_size > 0 and patch_size > 0 and image_size % patch_size == 0:
        patches = (image_size // patch_size) ** 2
        if patches > 0:
            return patches, f"derived:({image_size}//{patch_size})**2"
    raise ValueError(
        "cannot resolve image_seq_length: config.image_seq_length is "
        f"{merged!r}, and the vision geometry is image_size={image_size!r}, "
        f"patch_size={patch_size!r}; the placeholder guard would be fail-open"
    )


class LlavaLensModel(HFLensModel):
    """``LensModel`` over a loaded ``LlavaForConditionalGeneration``.

    Inherits the text-only protocol (``encode``/``forward``/``unembed``) from
    :class:`jlens.hf.HFLensModel` so text-only fits and upstream ``JacobianLens.apply``
    keep working, and adds the multimodal path (``encode_mm``/``forward_mm``).

    Args:
        hf_model: Loaded ``LlavaForConditionalGeneration`` (eval mode, device-placed).
        processor: Matching ``LlavaProcessor`` (or a stand-in exposing the same call).
        compile: Wrap each residual block in ``torch.compile`` (see upstream docs).
    """

    def __init__(
        self,
        hf_model: Any,
        processor: Any,
        *,
        compile: bool = False,
        force_bos: bool = True,
    ) -> None:
        super().__init__(
            hf_model,
            getattr(processor, "tokenizer", processor),
            compile=compile,
            force_bos=force_bos,
        )
        self.hf_model = hf_model
        self.processor = processor
        config = hf_model.config
        self.image_token_id: int = int(config.image_token_id)
        self.image_seq_length, self.image_seq_length_source = resolve_image_seq_length(config)
        self.vision_feature_layer = config.vision_feature_layer
        self.vision_feature_select_strategy = config.vision_feature_select_strategy

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "llava-hf/llava-1.5-7b-hf",
        *,
        dtype: torch.dtype | str | None = torch.bfloat16,
        device: torch.device | str | None = None,
        attn_implementation: str | None = None,
        processor: Any | None = None,
        compile: bool = False,
        local_files_only: bool = False,
    ) -> LlavaLensModel:
        """Load model + processor and wrap them.

        Kept separate from :func:`jlens.hf.from_hf` because LLaVA needs the processor
        (image preprocessing) in addition to the tokenizer, and defaults to bf16.
        """
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        kwargs: dict[str, Any] = {"local_files_only": local_files_only}
        if dtype is not None:
            kwargs["dtype"] = dtype
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        hf_model = LlavaForConditionalGeneration.from_pretrained(model_id, **kwargs)
        if device is not None:
            hf_model = hf_model.to(device)
        if processor is None:
            processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
        return cls(hf_model, processor, compile=compile)

    # ------------------------------------------------------------------ encoding
    @property
    def vision_device(self) -> torch.device:
        return self.hf_model.model.vision_tower.device

    @property
    def vision_dtype(self) -> torch.dtype:
        try:
            return self.hf_model.model.vision_tower.dtype
        except AttributeError:  # pragma: no cover - defensive for exotic towers
            return next(self.hf_model.model.vision_tower.parameters()).dtype

    def load_image(self, image: Any) -> Any:
        """Pass through PIL images / arrays; resolve strings as file paths."""
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        return image

    def encode_mm(
        self,
        text: str,
        images: Any | None = None,
        *,
        max_length: int | None = None,
    ) -> MultimodalBatch:
        """Encode ``text`` (with ``<image>`` placeholders) and images into a batch.

        Args:
            text: Prompt containing one ``<image>`` placeholder per image.
            images: ``None``, a single image (path/PIL/array), or a list of them.
            max_length: If given, reject (``ValueError``) samples longer than this
                instead of truncating — truncation would silently drop image
                placeholders and corrupt the lens average.

        Raises:
            ValueError: placeholder/image count mismatch, or sequence too long.
        """
        if images is None:
            image_list: list[Any] = []
        elif isinstance(images, (str, Path, Image.Image)) or hasattr(images, "shape"):
            image_list = [self.load_image(images)]
        else:
            image_list = [self.load_image(image) for image in images]

        expected_placeholders = text.count(IMAGE_PLACEHOLDER)
        if expected_placeholders != len(image_list):
            raise ValueError(
                f"prompt has {expected_placeholders} {IMAGE_PLACEHOLDER} placeholders "
                f"but {len(image_list)} images were given"
            )

        inputs = self.processor(
            images=image_list or None,
            text=text,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(self.input_device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(self.input_device)

        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(device=self.vision_device, dtype=self.vision_dtype)

        image_token_mask = input_ids == self.image_token_id
        n_image_tokens = int(image_token_mask.sum())
        if image_list:
            if self.image_seq_length <= 0:  # pragma: no cover - __init__ refuses first
                raise ValueError(
                    "image_seq_length is unknown; refusing to skip the placeholder guard"
                )
            expected = len(image_list) * self.image_seq_length
            if n_image_tokens != expected:
                raise ValueError(
                    f"tokenizer expanded to {n_image_tokens} image tokens but "
                    f"{len(image_list)} images x {self.image_seq_length} were expected; "
                    "check the placeholder spelling and processor configuration"
                )

        batch = MultimodalBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_token_mask=image_token_mask,
            text=text,
        )
        if max_length is not None and batch.seq_len > max_length:
            raise ValueError(
                f"sample is {batch.seq_len} tokens, over the {max_length}-token limit "
                "(truncation is refused for multimodal samples)"
            )
        return batch

    def vision_fingerprint(self) -> dict[str, Any]:
        """Plain-type vision-side identity, recorded in fit provenance (V10).

        ``image_seq_length`` resolution is fail-closed (see
        :func:`resolve_image_seq_length`), so config drift (missing value, different
        ``vision_feature_select_strategy`` or patch geometry) shows up in the
        fingerprint instead of silently changing the number of LM positions per image.
        """
        vision = getattr(self.hf_model.config, "vision_config", None)
        return {
            "image_seq_length": int(self.image_seq_length),
            "image_seq_length_source": str(self.image_seq_length_source),
            "vision_feature_layer": self.vision_feature_layer,
            "vision_feature_select_strategy": self.vision_feature_select_strategy,
            "vision_image_size": int(getattr(vision, "image_size", 0) or 0),
            "vision_patch_size": int(getattr(vision, "patch_size", 0) or 0),
            "vision_hidden_size": int(getattr(vision, "hidden_size", 0) or 0),
            "vision_model_type": getattr(vision, "model_type", None),
        }


    # ------------------------------------------------------------------ forward
    def forward_mm(self, batch: MultimodalBatch) -> torch.Tensor:
        """Run the LM stack on a multimodal batch; returns HF's ``last_hidden_state``.

        ``LlavaModel.forward`` is used verbatim: image features are computed by the
        model's own vision tower + projector and scattered into the input embeddings, so
        this is numerically the HuggingFace forward pass. ``last_hidden_state`` is the
        **post-final-norm** tensor (HF convention); the lens is defined on the *pre-norm*
        residual stream, see :meth:`forward_residual`.
        """
        output = self.hf_model.model(
            **batch.hf_kwargs(),
            use_cache=False,
        )
        return output.last_hidden_state

    def forward_residual(self, batch: MultimodalBatch) -> torch.Tensor:
        """Final-layer residual stream *before* the final norm: ``[1, seq_len, d]``.

        This is the tensor the lens is defined on and the one ``unembed`` maps to logits:
        ``unembed(self.forward_residual(batch))`` equals HF's ``logits``. Captured with a
        forward hook on the last residual block — the same activation the fit records.
        """
        last = len(self.layers) - 1
        with ActivationRecorder(self.layers, at=[last]) as recorder:
            self.forward_mm(batch)
            return recorder.activations[last]

    def unembed_weight(self, *, layer_agnostic: bool = True) -> torch.Tensor:
        """The unembedding matrix ``W_U`` as ``[vocab_size, d_model]``."""
        del layer_agnostic
        return self._lm_head.weight  # type: ignore[attr-defined]

    def predict_caption(
        self,
        batch: MultimodalBatch,
        *,
        max_new_tokens: int = 64,
        **generate_kwargs: Any,
    ) -> str:
        """Greedy caption for a prompt-only batch (used by the manifest builder)."""
        with torch.no_grad():
            ids = self.hf_model.generate(
                **batch.hf_kwargs(),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                **generate_kwargs,
            )
        new_tokens = ids[0, batch.seq_len :]
        return self._decode(new_tokens)

    def _decode(self, token_ids: torch.Tensor) -> str:
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None or not hasattr(tokenizer, "decode"):
            return " ".join(str(int(t)) for t in token_ids)
        return tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)


__all__ = [
    "IMAGE_PLACEHOLDER",
    "LlavaLensModel",
    "MultimodalBatch",
    "resolve_image_seq_length",
]
