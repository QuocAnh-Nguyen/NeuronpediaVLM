# SPDX-License-Identifier: Apache-2.0
"""Tiny random-weight LLaVA on CPU, for end-to-end dry runs and tests.

Uses the real HuggingFace classes (``LlavaConfig`` / ``LlavaForConditionalGeneration``)
with miniature dimensions, so the entire production pipeline — processor -> placeholder
expansion -> image fusion -> residual stack -> lens fit -> readout -> interventions —
executes the real code paths without a GPU or any download. The 576-token image block
is kept at the production size (336/14 patches) because placeholder expansion and the
position-mask logic depend on it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration

from vlm_lens.models.llava import IMAGE_PLACEHOLDER


class TinyLlavaProcessor:
    """Minimal stand-in for ``LlavaProcessor`` with the same call surface.

    Tokenization is a deterministic byte hash into a small vocab; the image pipeline is
    the real placeholder expansion (576 tokens per ``<image>``). ``__call__`` accepts
    ``text`` positionally as well, because ``jlens.hf.HFLensModel.encode`` calls the
    tokenizer that way for the text-only path.
    """

    def __init__(
        self,
        *,
        image_token_id: int = 100,
        num_image_tokens: int = 576,
        vocab_size: int = 128,
        bos_token_id: int = 1,
        image_size: int = 336,
        seed: int = 0,
    ) -> None:
        self.image_token_id = image_token_id
        self.num_image_tokens = num_image_tokens
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.image_size = image_size
        self.add_bos_token = True  # HFLensModel sets this; kept for API parity
        self.seed = seed
        self.tokenizer = self  # so LlavaLensModel(..., processor) finds .tokenizer

    # -- tokenizer surface -------------------------------------------------
    def _token_id(self, piece: str) -> int:
        digest = hashlib.blake2b(piece.encode("utf-8"), digest_size=4).digest()
        return 2 + (int.from_bytes(digest, "big") % (self.vocab_size - 2 - self.image_token_id))

    def encode_text(self, text: str) -> list[int]:
        ids = [self.bos_token_id]
        for word in text.split(" "):
            if IMAGE_PLACEHOLDER in word:
                head, _, tail = word.partition(IMAGE_PLACEHOLDER)
                if head:
                    ids.append(self._token_id(head))
                ids.extend([self.image_token_id] * self.num_image_tokens)
                if tail:
                    ids.append(self._token_id(tail))
            else:
                ids.append(self._token_id(word))
        return ids

    def decode(self, ids: Any, **_kwargs: Any) -> str:
        return " ".join(self.convert_ids_to_tokens(ids))

    def convert_ids_to_tokens(self, ids: Any) -> list[str]:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [ids]
        return [
            IMAGE_PLACEHOLDER if int(i) == self.image_token_id else f"t{int(i)}" for i in ids
        ]

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

        encoded = [self.encode_text(t) for t in texts]
        if truncation and max_length is not None:
            encoded = [ids[:max_length] for ids in encoded]
        input_ids = torch.tensor(encoded, dtype=torch.long)
        batch: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }
        if image_list:
            batch["pixel_values"] = stack_images(image_list, image_size=self.image_size)
        return batch


def random_image(seed: int, *, image_size: int = 336) -> np.ndarray:
    """Deterministic random ``uint8`` HWC image, as the PIL path would deliver."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)


def stack_images(images: list[Any], *, image_size: int = 336) -> torch.Tensor:
    """``list[ndarray|Tensor|PIL|path] -> [n, 3, H, W] float32`` in [0, 1]."""
    tensors = []
    for index, image in enumerate(images):
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
            array = random_image(seed=index, image_size=image_size)
        if array.dtype != np.uint8:
            array = (np.clip(array, 0, 1) * 255).astype(np.uint8)
        tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float() / 255.0
        tensors.append(tensor)
    return torch.stack(tensors)


@dataclass(frozen=True)
class TinyLlavaConfig:
    """Sizes for :func:`build_tiny_llava` (kept small; the image block stays 576)."""

    d_model: int = 32
    n_layers: int = 4
    n_heads: int = 4
    vision_hidden: int = 32
    vision_layers: int = 2
    image_size: int = 336
    patch_size: int = 14
    vocab_size: int = 128
    image_token_id: int = 100
    seed: int = 0


def build_tiny_llava(
    config: TinyLlavaConfig | None = None,
) -> tuple[LlavaForConditionalGeneration, TinyLlavaProcessor]:
    """Random-weight ``LlavaForConditionalGeneration`` + matching processor stand-in."""
    config = config or TinyLlavaConfig()
    torch.manual_seed(config.seed)

    vision_config = CLIPVisionConfig(
        hidden_size=config.vision_hidden,
        intermediate_size=config.vision_hidden * 2,
        num_hidden_layers=config.vision_layers,
        num_attention_heads=2,
        image_size=config.image_size,
        patch_size=config.patch_size,
        projection_dim=config.vision_hidden,
        num_channels=3,
    )
    text_config = LlamaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.d_model,
        intermediate_size=config.d_model * 2,
        num_hidden_layers=config.n_layers,
        num_attention_heads=config.n_heads,
        num_key_value_heads=max(1, config.n_heads // 2),
        max_position_embeddings=2048,
        tie_word_embeddings=False,
    )
    llava_config = LlavaConfig(
        vision_config=vision_config,
        text_config=text_config,
        image_token_index=config.image_token_id,
        image_seq_length=(config.image_size // config.patch_size) ** 2,
        vision_feature_layer=-2,
        vision_feature_select_strategy="default",
        projector_hidden_act="gelu",
        tie_word_embeddings=False,
    )
    model = LlavaForConditionalGeneration(llava_config)
    model.eval()
    processor = TinyLlavaProcessor(
        image_token_id=config.image_token_id,
        num_image_tokens=(config.image_size // config.patch_size) ** 2,
        vocab_size=config.vocab_size,
        seed=config.seed,
    )
    return model, processor
