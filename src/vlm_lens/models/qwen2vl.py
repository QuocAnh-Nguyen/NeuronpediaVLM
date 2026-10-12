# SPDX-License-Identifier: Apache-2.0
"""Multimodal ``LensModel`` over HuggingFace Qwen2-VL (``Qwen/Qwen2-VL-7B-Instruct``).

Design notes
------------
Same shape as the LLaVA adapter: the Jacobian lens lives on the *LLM* residual stream,
the vision tower and merger are frozen input providers, and the forward pass calls the
model's own :class:`Qwen2VLModel` (``hf_model.model``) verbatim —
``get_image_features`` -> ``get_placeholder_mask`` -> ``masked_scatter`` -> LM stack —
so token counts, dtypes and masking are exactly HuggingFace's. What this adapter adds is
only what the lens needs: placeholder-safe encoding, modality masks, and a residual-only
forward.

Where Qwen2-VL genuinely differs from LLaVA, and how it is absorbed:

- **Variable image-token count.** One ``<|vision_start|><|image_pad|><|vision_end|>``
  block expands to ``prod(image_grid_thw[i]) // spatial_merge_size**2`` LM tokens for
  image *i* (2x2 patch merge), so there is no fixed ``image_seq_length`` to guard: the
  placeholder check compares the tokenized image-token count against the count the
  processor's own ``image_grid_thw`` implies.
- **M-RoPE needs modality labels.** ``Qwen2VLModel.compute_3d_position_ids`` raises when
  ``image_grid_thw`` is passed without ``mm_token_type_ids`` (transformers 5.x), so
  :class:`MultimodalBatch` carries both and ``hf_kwargs`` forwards them; ``encode_mm``
  fails closed when the processor omits ``mm_token_type_ids`` for an image prompt.
- **Prompts go through the chat template.** The user text (with ``<image>``
  placeholders, spelled exactly as in the LLaVA adapter) is split into ``user`` content
  parts — text segments and ``{"type": "image"}`` parts — and rendered with
  ``processor.apply_chat_template(..., tokenize=False, add_generation_prompt=False)``.
  Whether to append the generation prompt is the caller's choice, as with LLaVA, where
  the prompt string carries the ``ASSISTANT:`` scaffold itself.
- **The processor loads without the video stack.** ``AutoProcessor.from_pretrained``
  also builds ``Qwen2VLVideoProcessor``, whose module imports torchvision at module
  level and fails on torchvision-less boxes (see AGENTS invariants); transformers 5.17
  additionally rejects ``video_processor=None`` when constructing the composite
  directly. :func:`_load_processor` therefore builds a ``Qwen2VLProcessor`` subclass
  carrying only the image processor and tokenizer — the generic ``__call__`` guards
  the video branch with ``hasattr`` — with the PIL image processor pinned so loading
  and image preprocessing never touch torchvision.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
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
    pixel_values: torch.Tensor | None  # [n_patches, channels*temporal*ph*pw], flattened
    image_grid_thw: torch.Tensor | None  # [n_images, 3] (t, h, w) in patch units
    mm_token_type_ids: torch.Tensor | None  # [1, seq_len] int: 0 text, 1 image
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
            image_grid_thw=(
                None if self.image_grid_thw is None else self.image_grid_thw.to(device)
            ),
            mm_token_type_ids=(
                None if self.mm_token_type_ids is None else self.mm_token_type_ids.to(device)
            ),
            image_token_mask=self.image_token_mask.to(device),
            text=self.text,
        )

    def hf_kwargs(self) -> dict[str, torch.Tensor]:
        """Keyword arguments for ``Qwen2VLForConditionalGeneration.forward``/``generate``.

        ``mm_token_type_ids`` is required alongside ``image_grid_thw``: the model builds
        M-RoPE position ids from the modality labels and raises without them.
        """
        kwargs: dict[str, torch.Tensor] = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }
        if self.pixel_values is not None:
            kwargs["pixel_values"] = self.pixel_values
        if self.image_grid_thw is not None:
            kwargs["image_grid_thw"] = self.image_grid_thw
        if self.mm_token_type_ids is not None:
            kwargs["mm_token_type_ids"] = self.mm_token_type_ids
        return kwargs

    def expand(self, n: int) -> MultimodalBatch:
        """Replicate the sample along the batch axis (the fit's ``dim_batch`` trick).

        The fit relies on every batch element being the identical prompt. HuggingFace
        consumes ``pixel_values`` as a flattened ``[n_patches, dim]`` stack shared across
        the batch and indexes it by ``image_grid_thw``, so one image per sample is the
        only expressible layout: the single image's patch rows are block-repeated ``n``
        times (a copy — the one place :meth:`expand` cannot use views) and the grid rows
        are replicated. With several images per sample, fit at ``dim_batch=1``.
        """
        if n == 1:
            return self
        if self.pixel_values is not None and int(self.image_grid_thw.shape[0]) != 1:
            raise ValueError(
                f"dim_batch replication needs one image per sample, got "
                f"{int(self.image_grid_thw.shape[0])} images; use dim_batch=1 for "
                "multi-image samples"
            )
        return MultimodalBatch(
            input_ids=self.input_ids.expand(n, -1),
            attention_mask=self.attention_mask.expand(n, -1),
            pixel_values=(
                None if self.pixel_values is None else self.pixel_values.repeat(n, 1)
            ),
            image_grid_thw=(
                None if self.image_grid_thw is None else self.image_grid_thw.expand(n, -1)
            ),
            mm_token_type_ids=(
                None if self.mm_token_type_ids is None else self.mm_token_type_ids.expand(n, -1)
            ),
            image_token_mask=self.image_token_mask.expand(n, -1),
            text=self.text,
        )


def _load_processor(model_id: str, *, local_files_only: bool = False) -> Any:
    """Assemble the Qwen2-VL processor without ever touching the video stack.

    ``AutoProcessor.from_pretrained`` builds every ``Qwen2VLProcessor`` sub-component,
    including ``Qwen2VLVideoProcessor``, whose module imports torchvision at module
    level — an ``ImportError`` on torchvision-less environments (the fit box). The
    direct-construction escape hatch is closed too: transformers 5.17 class-checks
    ``video_processor`` against a torchvision-gated base, so ``video_processor=None``
    is rejected. The only torchvision-free construction is a subclass whose ``__init__``
    signature simply has no video component; ``ProcessorMixin.__call__`` guards the
    video branch with ``hasattr(self, "video_processor")``, and the token-label
    attributes the image/text path reads are set exactly as ``Qwen2VLProcessor.__init__``
    sets them.

    The image processor is resolved through ``AutoImageProcessor`` with
    ``backend="pil"`` — imported from its submodule, because the top-level lazy
    attribute is a ``DummyObject`` that raises without torchvision. Pinning PIL keeps
    the wrapper torchvision-independent everywhere; the PIL and torchvision Qwen2-VL
    processors share the same ``smart_resize`` and patchify math, so ``image_grid_thw``
    and the ``<|image_pad|>`` expansion come out identical. The chat template is read
    with the same ``get_processor_dict`` helper ``ProcessorMixin.from_pretrained``
    uses, so ``apply_chat_template`` sees the checkpoint's processor-level template.
    """
    from transformers import AutoTokenizer, Qwen2VLProcessor
    from transformers.models.auto.image_processing_auto import AutoImageProcessor
    from transformers.processing_utils import ProcessorMixin

    class _Qwen2VLImageOnlyProcessor(Qwen2VLProcessor):
        """``Qwen2VLProcessor`` carrying only ``image_processor`` and ``tokenizer``."""

        def __init__(self, image_processor=None, tokenizer=None, chat_template=None):
            # Same token attributes as Qwen2VLProcessor.__init__; only the mandatory
            # video_processor argument is dropped (see _load_processor docstring).
            self.image_token = (
                "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
            )
            self.video_token = (
                "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
            )
            self.image_token_id = (
                tokenizer.image_token_id
                if getattr(tokenizer, "image_token_id", None)
                else tokenizer.convert_tokens_to_ids(self.image_token)
            )
            self.video_token_id = (
                tokenizer.video_token_id
                if getattr(tokenizer, "video_token_id", None)
                else tokenizer.convert_tokens_to_ids(self.video_token)
            )
            ProcessorMixin.__init__(self, image_processor, tokenizer, chat_template=chat_template)

    processor_dict, _ = ProcessorMixin.get_processor_dict(model_id, local_files_only=local_files_only)
    return _Qwen2VLImageOnlyProcessor(
        image_processor=AutoImageProcessor.from_pretrained(
            model_id, local_files_only=local_files_only, backend="pil"
        ),
        tokenizer=AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only),
        chat_template=processor_dict.get("chat_template"),
    )


class Qwen2VLLensModel(HFLensModel):
    """``LensModel`` over a loaded ``Qwen2VLForConditionalGeneration``.

    Inherits the text-only protocol (``encode``/``forward``/``unembed``) from
    :class:`jlens.hf.HFLensModel` so text-only fits and upstream ``JacobianLens.apply``
    keep working, and adds the multimodal path (``encode_mm``/``forward_mm``/
    :meth:`forward_residual`).

    Args:
        hf_model: Loaded ``Qwen2VLForConditionalGeneration`` (eval mode, device-placed).
        processor: Matching ``Qwen2VLProcessor`` (or a stand-in exposing the same call
            and ``apply_chat_template`` surfaces).
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
        self.spatial_merge_size: int = int(config.vision_config.spatial_merge_size)

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
        *,
        dtype: torch.dtype | str | None = torch.bfloat16,
        device: torch.device | str | None = None,
        attn_implementation: str | None = None,
        processor: Any | None = None,
        compile: bool = False,
        local_files_only: bool = False,
    ) -> Qwen2VLLensModel:
        """Load model + processor and wrap them.

        Kept separate from :func:`jlens.hf.from_hf` because Qwen2-VL needs the processor
        (image preprocessing + chat template) in addition to the tokenizer, and defaults
        to bf16 for the real-fit path. The processor is assembled image-only by
        :func:`_load_processor`, so loading never requires a video/torchvision backend.
        """
        from transformers import Qwen2VLForConditionalGeneration

        kwargs: dict[str, Any] = {"local_files_only": local_files_only}
        if dtype is not None:
            kwargs["dtype"] = dtype
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        hf_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        if device is not None:
            hf_model = hf_model.to(device)
        if processor is None:
            processor = _load_processor(model_id, local_files_only=local_files_only)
        return cls(hf_model, processor, compile=compile)

    # ------------------------------------------------------------------ encoding
    @property
    def vision_device(self) -> torch.device:
        return self.hf_model.model.visual.device

    @property
    def vision_dtype(self) -> torch.dtype:
        try:
            return self.hf_model.model.visual.dtype
        except AttributeError:  # pragma: no cover - defensive for exotic towers
            return next(self.hf_model.model.visual.parameters()).dtype

    def load_image(self, image: Any) -> Any:
        """Pass through PIL images / arrays; resolve strings as file paths.

        Images must reach the processor as PIL objects — ``pathlib.Path`` is rejected by
        transformers 5.x and str/bytes go through ``torchvision.io.decode_image``, which
        dies on a torchvision built without libjpeg (see AGENTS invariants).
        """
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        return image

    def expected_image_tokens(self, image_grid_thw: torch.Tensor) -> int:
        """LM tokens ``image_grid_thw`` expands to: ``prod(grid_i) // merge**2`` per image.

        Mirrors ``Qwen2VLProcessor.replace_image_token`` /
        ``Qwen2VLModel.get_image_features``, which both derive the per-image token count
        this way (the 2x2 spatial merge). Returns the sum over images.
        """
        merge = self.spatial_merge_size**2
        total = 0
        for grid in image_grid_thw:
            t, h, w = (int(value) for value in grid)
            total += (t * h * w) // merge
        return total

    _CHAT_SHAPE = re.compile(r"^\s*USER:\s*(.*?)\s*ASSISTANT:\s*(.*)$", re.DOTALL)

    def _chat_messages(self, text: str) -> list[dict[str, Any]]:
        """Chat messages for a corpus prompt, Qwen2-VL native.

        The shared corpus spells prompts in the LLaVA shape,
        ``USER: <image>\\n{question}\\nASSISTANT: {caption}``; rendering that literally
        would make the caption part of the *user* turn. When the shape matches, the
        question (with its ``<image>`` placeholders) becomes the user turn and a
        non-empty caption becomes the assistant turn (teacher-forced fits); an empty
        caption is a generation prompt and stays user-only (the caller appends the
        assistant opener via ``add_generation_prompt``). Anything else — text-only
        corpora, free-form prompts — renders as a single user message exactly as
        before.
        """
        match = self._CHAT_SHAPE.match(text)
        if match is None:
            return [{"role": "user", "content": self._content_parts(text)}]
        messages = [
            {"role": "user", "content": self._content_parts(match.group(1))}
        ]
        caption = match.group(2).strip()
        if caption:
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": caption}]}
            )
        return messages

    @staticmethod
    def _content_parts(text: str) -> list[dict[str, Any]]:
        """Content parts interleaving text segments and ``{"type": "image"}`` parts."""
        content: list[dict[str, Any]] = []
        for index, segment in enumerate(text.split(IMAGE_PLACEHOLDER)):
            if index:
                content.append({"type": "image"})
            if segment:
                content.append({"type": "text", "text": segment})
        if not content:  # empty prompt
            content = [{"type": "text", "text": text}]
        return content

    def encode_mm(
        self,
        text: str,
        images: Any | None = None,
        *,
        max_length: int | None = None,
    ) -> MultimodalBatch:
        """Encode ``text`` (with ``<image>`` placeholders) and images into a batch.

        The text is wrapped in a one-turn chat template (``processor.apply_chat_template``,
        ``add_generation_prompt=False`` — appending the generation prompt is the caller's
        choice, as in the LLaVA adapter) and run through the processor, which expands each
        ``<|image_pad|>`` to the per-image grid-derived token count.

        Args:
            text: Prompt containing one ``<image>`` placeholder per image.
            images: ``None``, a single image (path/PIL/array), or a list of them.
            max_length: If given, reject (``ValueError``) samples longer than this
                instead of truncating — truncation would silently drop image
                placeholders and corrupt the lens average.

        Raises:
            ValueError: placeholder/image count mismatch, image-token count vs
                ``image_grid_thw`` mismatch, a processor that omits the M-RoPE modality
                labels, or sequence too long.
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

        prompt = self.processor.apply_chat_template(
            self._chat_messages(text),
            tokenize=False,
            add_generation_prompt=False,
        )
        inputs = self.processor(
            text=prompt,
            images=image_list or None,
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
        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(self.input_device)
        mm_token_type_ids = inputs.get("mm_token_type_ids")
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids.to(self.input_device)

        image_token_mask = input_ids == self.image_token_id
        n_image_tokens = int(image_token_mask.sum())
        if image_list:
            expected = self.expected_image_tokens(image_grid_thw)
            if n_image_tokens != expected:
                raise ValueError(
                    f"tokenizer expanded to {n_image_tokens} image tokens but "
                    f"{expected} were expected from image_grid_thw "
                    f"{image_grid_thw.tolist()} (spatial merge "
                    f"{self.spatial_merge_size}x{self.spatial_merge_size}); check the "
                    "placeholder spelling and processor configuration"
                )
            if mm_token_type_ids is None:
                raise ValueError(
                    "processor returned no mm_token_type_ids for an image prompt; the "
                    "model needs the modality labels to build M-RoPE position ids "
                    "(transformers 5.x raises without them)"
                )

        batch = MultimodalBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
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
        """Plain-type vision-side identity, recorded in fit provenance.

        With variable-resolution images there is no ``image_seq_length`` to pin; the
        merge/patch geometry and the placeholder token id are what silently change the
        number of LM positions per image, so those are what the fingerprint records.
        """
        vision = self.hf_model.config.vision_config
        return {
            "image_token_id": int(self.image_token_id),
            "spatial_merge_size": int(self.spatial_merge_size),
            "vision_patch_size": int(vision.patch_size),
            "vision_temporal_patch_size": int(vision.temporal_patch_size),
            "vision_depth": int(vision.depth),
            "vision_hidden_size": int(vision.hidden_size),
            "vision_embed_dim": int(vision.embed_dim),
            "vision_model_type": getattr(vision, "model_type", None),
        }

    # ------------------------------------------------------------------ forward
    def forward_mm(self, batch: MultimodalBatch) -> torch.Tensor:
        """Run the LM stack on a multimodal batch; returns HF's ``last_hidden_state``.

        ``Qwen2VLModel.forward`` is used verbatim: image features are computed by the
        model's own vision tower + merger and scattered into the input embeddings, and
        M-RoPE position ids are derived from ``image_grid_thw``/``mm_token_type_ids``, so
        this is numerically the HuggingFace forward pass. ``last_hidden_state`` is the
        **post-final-norm** tensor (HF convention); the lens is defined on the *pre-norm*
        residual stream, see :meth:`forward_residual`.
        """
        output = self.hf_model.model(
            **batch.hf_kwargs(),
            use_cache=False,
        )
        return output.last_hidden_state

    def forward_residual(
        self,
        batch: MultimodalBatch,
        layers: Any | None = None,
    ) -> dict[int, torch.Tensor]:
        """Block-output residual streams *before* the final norm, keyed by layer index.

        The lens is defined on these pre-norm block outputs — the same activations the
        fit records with ``ActivationRecorder`` — **not** on HF's post-final-norm
        ``last_hidden_state`` (applying the final norm twice is wrong).
        ``unembed(self.forward_residual(batch)[last])`` equals HF's ``logits``.

        Unlike ``LlavaLensModel.forward_residual`` (which returns only the last block's
        tensor), this returns every requested layer's block output in one hooked pass;
        ``layers=None`` records all blocks.

        Raises:
            ValueError: a requested layer index is outside ``[0, n_layers)``.
        """
        if layers is None:
            record_at = list(range(len(self.layers)))
        else:
            requested = sorted({int(layer) for layer in layers})
            bad = [layer for layer in requested if not 0 <= layer < len(self.layers)]
            if bad:
                raise ValueError(
                    f"layer indices {bad} out of range for {len(self.layers)} blocks"
                )
            record_at = requested
        with ActivationRecorder(self.layers, at=record_at) as recorder:
            self.forward_mm(batch)
            return {layer: recorder.activations[layer] for layer in record_at}

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

    def encode_generate_batch(
        self,
        texts: Sequence[str],
        images: Sequence[Any],
    ) -> list[dict[str, Any]]:
        """Per-sample processor inputs for the shared caption builder's ``generate``.

        The builder hands raw prompt strings to the processor directly; Qwen2-VL needs
        its chat skeleton — a bare ``<image>`` never becomes ``<|image_pad|>``, so
        ``input_ids`` would carry zero image tokens while the features still compute,
        and HF's forward rejects the mismatch. Each (text, image) pair is rendered
        through :meth:`_chat_messages` (which parses the corpus's LLaVA shape into
        native turns) with ``add_generation_prompt=True``. One image per call, no
        padding: image expansions differ per image, so a padded batch would mis-align
        the continuation slice the builder takes.

        Returns:
            One plain kwargs dict per pair, values still on the processor's device —
            the caller moves them (see ``data.captions.generate_captions``).
        """
        if len(texts) != len(images):
            raise ValueError(
                f"{len(texts)} prompts but {len(images)} images for generate encoding"
            )
        batches: list[dict[str, Any]] = []
        for text, image in zip(texts, images, strict=True):
            prompt = self.processor.apply_chat_template(
                self._chat_messages(text),
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
            batches.append(dict(inputs))
        return batches

    def _decode(self, token_ids: torch.Tensor) -> str:
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None or not hasattr(tokenizer, "decode"):
            return " ".join(str(int(t)) for t in token_ids)
        return tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)


__all__ = [
    "IMAGE_PLACEHOLDER",
    "MultimodalBatch",
    "Qwen2VLLensModel",
]
