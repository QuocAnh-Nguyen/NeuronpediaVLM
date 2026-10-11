# SPDX-License-Identifier: Apache-2.0
"""Tiny random-weight Qwen2-VL on CPU, for end-to-end smokes and tests.

Uses the real HuggingFace classes (``Qwen2VLConfig`` / ``Qwen2VLForConditionalGeneration``)
with miniature dimensions, so the production pipeline — chat template -> per-image
placeholder expansion -> M-RoPE -> image fusion -> residual stack — executes the real
code paths without a GPU or download. The processor is a minimal stand-in (the role
``tiny_llava.TinyLlavaProcessor`` plays for LLaVA) that reproduces the *production
placeholder logic*:

- the rendered chat template carries one ``<|vision_start|><|image_pad|><|vision_end|>``
  block per image, and the i-th ``<|image_pad|>`` expands to
  ``prod(image_grid_thw[i]) // spatial_merge_size**2`` copies of the image token —
  exactly ``Qwen2VLProcessor.replace_image_token``;
- ``pixel_values`` are the production flattened patches ``[total_patches,
  channels*temporal*ph*pw]`` with the temporal frame repeated and patch rows grouped by
  2x2 merge cell — exactly ``Qwen2VLImageProcessor.patchify``;
- ``mm_token_type_ids`` label image (1) vs text (0) positions, as the production
  processor returns for M-RoPE.

Unlike production there is no smart-resize: images keep their given size, so the merged
token count is fixed per image size (56x56 -> 4 tokens, 336x336 -> 144); sizes must be
multiples of ``patch_size * spatial_merge_size`` or the 2x2 merge grouping is undefined
(``ValueError``). Non-special tokenization is a deterministic byte hash into a small
vocab, as in ``tiny_llava``. Qwen2-VL tokenizers prepend no BOS token (the chat template
opens with ``<|im_start|>``), so neither does this stand-in — ``force_bos`` is a no-op
here, matching production.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import (
    Qwen2VLConfig,
    Qwen2VLForConditionalGeneration,
    Qwen2VLTextConfig,
    Qwen2VLVisionConfig,
)

IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"
IMAGE_PAD_TOKEN = "<|image_pad|>"

#: Special-token strings this stand-in emits. ``<|image_pad|>`` is last so its
#: enumerated id is overridden by ``image_token_id`` in ``__init__``.
SPECIAL_TOKENS: tuple[str, ...] = (
    "<|bos|>",
    IM_START_TOKEN,
    IM_END_TOKEN,
    VISION_START_TOKEN,
    VISION_END_TOKEN,
    IMAGE_PAD_TOKEN,
)
_SPECIAL_PATTERN = re.compile("(" + "|".join(re.escape(token) for token in SPECIAL_TOKENS) + ")")


class TinyQwen2VLProcessor:
    """Minimal stand-in for ``Qwen2VLProcessor`` with the same call surface.

    ``apply_chat_template`` renders the production Qwen2-VL chat skeleton (default
    system turn, ``<|vision_start|><|image_pad|><|vision_end|>`` per image part);
    ``__call__`` expands each ``<|image_pad|>`` to its image's grid-derived token count
    and produces the production pixel/label tensors. ``__call__`` accepts ``text``
    positionally as well, because ``jlens.hf.HFLensModel.encode`` calls the tokenizer
    that way for the text-only path.
    """

    def __init__(
        self,
        *,
        image_token_id: int = 100,
        vocab_size: int = 128,
        bos_token_id: int = 1,
        im_start_token_id: int = 2,
        im_end_token_id: int = 3,
        vision_start_token_id: int = 4,
        vision_end_token_id: int = 5,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        seed: int = 0,
    ) -> None:
        if not 6 < image_token_id < vocab_size:
            raise ValueError(
                f"image_token_id={image_token_id} must leave room for the five special "
                f"ids (1-5) and hash ids (6..{image_token_id - 1}) below vocab_size="
                f"{vocab_size}"
            )
        self.image_token_id = image_token_id
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.im_start_token_id = im_start_token_id
        self.im_end_token_id = im_end_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.seed = seed
        self.image_token = IMAGE_PAD_TOKEN
        self.tokenizer = self  # so Qwen2VLLensModel(..., processor) finds .tokenizer
        self._special_to_id = {
            token: 1 + index for index, token in enumerate(SPECIAL_TOKENS)
        }
        self._special_to_id[IMAGE_PAD_TOKEN] = image_token_id
        self._id_to_special = {value: key for key, value in self._special_to_id.items()}

    # -- tokenizer surface -------------------------------------------------
    def _token_id(self, piece: str) -> int:
        digest = hashlib.blake2b(piece.encode("utf-8"), digest_size=4).digest()
        return 6 + (int.from_bytes(digest, "big") % (self.image_token_id - 6))

    def encode_text(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in _SPECIAL_PATTERN.split(text):
            if not piece:
                continue
            if piece in self._special_to_id:
                ids.append(self._special_to_id[piece])
            else:
                ids.extend(self._token_id(word) for word in piece.split())
        return ids

    def convert_ids_to_tokens(
        self, ids: Any, skip_special_tokens: bool = False
    ) -> list[str]:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [ids]
        tokens: list[str] = []
        for token_id in ids:
            token_id = int(token_id)
            special = self._id_to_special.get(token_id)
            if special is not None:
                if not skip_special_tokens:
                    tokens.append(special)
            else:
                tokens.append(f"t{token_id}")
        return tokens

    def decode(self, ids: Any, skip_special_tokens: bool = False, **_kwargs: Any) -> str:
        return " ".join(self.convert_ids_to_tokens(ids, skip_special_tokens))

    def apply_chat_template(
        self,
        messages: Any,
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **_kwargs: Any,
    ) -> str:
        """Render the production Qwen2-VL chat skeleton (tokenize=False only)."""
        if tokenize:
            raise ValueError("TinyQwen2VLProcessor only supports tokenize=False")
        rendered = f"{IM_START_TOKEN}system\nYou are a helpful assistant.{IM_END_TOKEN}\n"
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if isinstance(content, str):
                body = content
            else:
                body = ""
                for part in content:
                    if part.get("type") == "image":
                        body += f"{VISION_START_TOKEN}{IMAGE_PAD_TOKEN}{VISION_END_TOKEN}"
                    else:
                        body += str(part.get("text", ""))
            rendered += f"{IM_START_TOKEN}{role}\n{body}{IM_END_TOKEN}\n"
        if add_generation_prompt:
            rendered += f"{IM_START_TOKEN}assistant\n"
        return rendered

    # -- image surface -----------------------------------------------------
    def _coerce_image(self, image: Any, index: int) -> np.ndarray:
        """``ndarray|Tensor|PIL|path|other -> uint8 HWC``, as the PIL path delivers."""
        if isinstance(image, np.ndarray):
            array = image
        elif isinstance(image, torch.Tensor):
            array = image.detach().cpu().numpy()
        elif hasattr(image, "convert"):  # PIL
            array = np.asarray(image.convert("RGB"))
        elif isinstance(image, (str, bytes)):
            from PIL import Image as _Image

            array = np.asarray(_Image.open(image).convert("RGB"))
        else:
            array = random_image(seed=index, image_size=56)
        if array.dtype != np.uint8:
            array = (np.clip(array, 0, 1) * 255).astype(np.uint8)
        return np.ascontiguousarray(array)

    def _patchify(self, image: np.ndarray) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """uint8 HWC image -> production patch rows + ``(t, h, w)`` grid.

        Same reshape/permute/temporal-repeat as ``Qwen2VLImageProcessor.patchify``: rows
        are ordered by merge cell first, and each row flattens as
        ``[channel, temporal, patch_h, patch_w]`` for ``PatchEmbed``'s ``view``.
        """
        ph = pw = self.patch_size
        merge = self.spatial_merge_size
        temporal = self.temporal_patch_size
        height, width = image.shape[:2]
        factor = self.patch_size * merge
        if height % factor or width % factor:
            raise ValueError(
                f"image is {height}x{width}; sizes must be multiples of "
                f"patch_size*spatial_merge_size={factor} for the merge grouping"
            )
        grid_h, grid_w = height // ph, width // pw
        patches = image.transpose(2, 0, 1).reshape(
            1, 3, grid_h // merge, merge, ph, grid_w // merge, merge, pw
        )
        # [1, gh/m, gw/m, m, m, C, ph, pw] -> repeat the frame along temporal
        patches = patches.transpose(0, 2, 5, 3, 6, 1, 4, 7)
        patches = np.broadcast_to(
            patches[:, :, :, :, :, :, None, :, :],
            (1, grid_h // merge, grid_w // merge, merge, merge, 3, temporal, ph, pw),
        )
        rows = (
            torch.from_numpy(np.ascontiguousarray(patches))
            .reshape(grid_h * grid_w, 3 * temporal * ph * pw)
            .float()
            / 255.0
        )
        return rows, (1, grid_h, grid_w)

    def _expand_image_pads(self, text: str, token_counts: list[int]) -> str:
        """Production expansion: i-th ``<|image_pad|>`` -> ``token_counts[i]`` copies."""
        parts = text.split(IMAGE_PAD_TOKEN)
        if len(parts) - 1 != len(token_counts):
            raise ValueError(
                f"text has {len(parts) - 1} {IMAGE_PAD_TOKEN} placeholders but "
                f"{len(token_counts)} images were given"
            )
        expanded = parts[0]
        for count, part in zip(token_counts, parts[1:]):
            expanded += IMAGE_PAD_TOKEN * count + part
        return expanded

    def __call__(
        self,
        text: Any = None,
        *,
        images: Any = None,
        return_tensors: str = "pt",
        truncation: bool = False,
        max_length: int | None = None,
        padding: bool = False,
        add_special_tokens: bool = True,
        **_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del return_tensors, padding, add_special_tokens
        texts = [text] if isinstance(text, str) else list(text or [])
        if images is None:
            image_list: list[Any] = []
        elif isinstance(images, (list, tuple)):
            image_list = list(images)
        else:
            image_list = [images]

        patch_rows: list[torch.Tensor] = []
        grids: list[tuple[int, int, int]] = []
        for index, image in enumerate(image_list):
            rows, grid = self._patchify(self._coerce_image(image, index))
            patch_rows.append(rows)
            grids.append(grid)
        merge = self.spatial_merge_size**2
        token_counts = [(t * h * w) // merge for t, h, w in grids]

        encoded = [
            self.encode_text(self._expand_image_pads(item, token_counts))
            for item in texts
        ]
        if truncation and max_length is not None:
            encoded = [ids[:max_length] for ids in encoded]
        input_ids = torch.tensor(encoded, dtype=torch.long)
        batch: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            # production processors return these labels for M-RoPE by default
            "mm_token_type_ids": (input_ids == self.image_token_id).long(),
        }
        if image_list:
            batch["pixel_values"] = torch.cat(patch_rows, dim=0)
            batch["image_grid_thw"] = torch.tensor(grids, dtype=torch.long)
        return batch


def random_image(seed: int, *, image_size: int = 56) -> np.ndarray:
    """Deterministic random ``uint8`` HWC image, as the PIL path would deliver."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)


@dataclass(frozen=True)
class TinyQwen2VLConfig:
    """Sizes for :func:`build_tiny_qwen2vl` (kept small; placeholder logic stays production)."""

    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 4
    vision_depth: int = 2
    vision_embed_dim: int = 64
    vision_heads: int = 4
    image_size: int = 56  # multiple of patch_size * spatial_merge_size
    patch_size: int = 14
    spatial_merge_size: int = 2
    vocab_size: int = 128
    image_token_id: int = 100
    seed: int = 0

    @property
    def image_tokens_per_image(self) -> int:
        """Merged LM tokens one ``image_size`` image expands to (2x2 merge)."""
        patches = (self.image_size // self.patch_size) ** 2
        return patches // self.spatial_merge_size**2


def build_tiny_qwen2vl(
    config: TinyQwen2VLConfig | None = None,
) -> tuple[Qwen2VLForConditionalGeneration, TinyQwen2VLProcessor]:
    """Random-weight ``Qwen2VLForConditionalGeneration`` + matching processor stand-in."""
    config = config or TinyQwen2VLConfig()
    torch.manual_seed(config.seed)

    # M-RoPE splits the rotary dimensions across (temporal, height, width); the
    # sections must sum to head_dim//2 (the rotary frequency count), mirroring the
    # real checkpoint's mrope_section which sums to 128//2.
    half_dim = (config.d_model // config.n_heads) // 2
    if half_dim < 3:
        raise ValueError(
            f"d_model/n_heads/4={half_dim} is too small to split into three mrope sections"
        )
    section = half_dim // 3
    mrope_section = [section, section, half_dim - 2 * section]

    vision_config = Qwen2VLVisionConfig(
        depth=config.vision_depth,
        embed_dim=config.vision_embed_dim,
        hidden_size=config.d_model,
        num_heads=config.vision_heads,
        patch_size=config.patch_size,
        spatial_merge_size=config.spatial_merge_size,
        temporal_patch_size=2,
        in_channels=3,
        hidden_act="quick_gelu",
        mlp_ratio=2,
    )
    text_config = Qwen2VLTextConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.d_model,
        intermediate_size=config.d_model * 2,
        num_hidden_layers=config.n_layers,
        num_attention_heads=config.n_heads,
        num_key_value_heads=max(1, config.n_heads // 2),
        max_position_embeddings=2048,
        bos_token_id=1,
        eos_token_id=3,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1000000.0,
            "mrope_section": mrope_section,
        },
    )
    model_config = Qwen2VLConfig(
        vision_config=vision_config,
        text_config=text_config,
        image_token_id=config.image_token_id,
        video_token_id=config.vocab_size - 1,
        vision_start_token_id=4,
        vision_end_token_id=5,
        tie_word_embeddings=False,
    )
    model = Qwen2VLForConditionalGeneration(model_config)
    model.eval()
    processor = TinyQwen2VLProcessor(
        image_token_id=config.image_token_id,
        vocab_size=config.vocab_size,
        patch_size=config.patch_size,
        spatial_merge_size=config.spatial_merge_size,
        seed=config.seed,
    )
    return model, processor
