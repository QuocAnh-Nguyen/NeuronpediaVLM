# SPDX-License-Identifier: Apache-2.0
"""Qwen2-VL processor loading: image-only assembly without the video stack.

``AutoProcessor.from_pretrained`` builds every ``Qwen2VLProcessor`` sub-component,
including ``Qwen2VLVideoProcessor`` — whose module imports torchvision at module level,
an ``ImportError`` on the fit box (AGENTS invariants). transformers 5.17 also class-
checks ``video_processor`` at construction, so ``video_processor=None`` is rejected and
nothing torchvision-free can fill the slot. These tests pin the torchvision-free
construction in :func:`vlm_lens.models.qwen2vl._load_processor`:

- it works with the video stack poisoned (the box condition, simulated),
- it is bit-identical to the plain ``Qwen2VLProcessor`` composite the AutoProcessor
  route assembles (video component stood in; it is never called for image/text),
- text-only behaviour is unchanged.

Everything runs offline against a local fixture directory carrying the real Qwen2-VL
processor configuration values (hub ``preprocessor_config.json``), the checkpoint's
chat-template skeleton and a minimal trained BPE tokenizer; no downloads, no GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from vlm_lens.models.qwen2vl import _load_processor

IMAGE_SIZE = 56  # 14px patch, 2x2 spatial merge -> 4x4 patches -> 4 LM tokens per image
IMAGE_TOKENS = 4
MERGE = 2

# Exact hub values (Qwen/Qwen2-VL-7B-Instruct preprocessor_config.json).
PREPROCESSOR_CONFIG: dict[str, Any] = {
    "min_pixels": 3136,
    "max_pixels": 12845056,
    "patch_size": 14,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [0.48145466, 0.4578275, 0.40821073],
    "image_std": [0.26862954, 0.26130258, 0.27577711],
    "image_processor_type": "Qwen2VLImageProcessor",
    "processor_class": "Qwen2VLProcessor",
}

SPECIAL_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
]

# Chat-template skeleton of the checkpoint's processor-level template (chat_template.json):
# default system turn, one <|vision_start|><|image_pad|><|vision_end|> block per image part.
CHAT_TEMPLATE = (
    "{% set image_count = namespace(value=0) %}"
    "{% for message in messages %}"
    "{% if loop.first and message['role'] != 'system' %}"
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{% for content in message['content'] %}"
    "{% if content['type'] == 'image' %}"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif 'text' in content %}{{ content['text'] }}{% endif %}"
    "{% endfor %}"
    "<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

MESSAGES = [
    {"role": "user", "content": [{"type": "text", "text": "Describe this image."}, {"type": "image"}]}
]


def _write_minimal_bpe_tokenizer(path: Path) -> None:
    from tokenizers import AddedToken, Tokenizer, models, trainers
    from tokenizers.pre_tokenizers import ByteLevel

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=350,
        initial_alphabet=ByteLevel.alphabet(),
        special_tokens=[AddedToken(token, special=True) for token in SPECIAL_TOKENS],
    )
    tokenizer.train_from_iterator(
        ["describe this image in one sentence", "the capital of france is", "a cat sits on a mat"],
        trainer,
    )
    tokenizer.save(str(path))


@pytest.fixture()
def processor_dir(tmp_path: Path) -> Path:
    """Offline stand-in for the checkpoint's processor files."""
    d = tmp_path / "qwen2vl-processor-fixture"
    d.mkdir()
    (d / "preprocessor_config.json").write_text(json.dumps(PREPROCESSOR_CONFIG))
    (d / "chat_template.json").write_text(json.dumps({"chat_template": CHAT_TEMPLATE}))
    (d / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "Qwen2Tokenizer",
                "model_max_length": 1000000,
                "eos_token": "<|im_end|>",
            }
        )
    )
    _write_minimal_bpe_tokenizer(d / "tokenizer.json")
    return d


@pytest.fixture()
def sample_image() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))


# ------------------------------------------------------------------ box condition
def test_load_processor_never_touches_the_video_stack(processor_dir, monkeypatch) -> None:
    """With the video stack poisoned (the box condition), the loader still works.

    Poisons both paths the AutoProcessor route would take: the video auto class's
    ``from_pretrained`` and the import of the qwen2_vl video module (torchvision-less
    boxes fail the import itself).
    """
    import transformers.models.auto.video_processing_auto as video_auto
    from transformers import Qwen2VLProcessor

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the video stack was touched")

    monkeypatch.setattr(video_auto.AutoVideoProcessor, "from_pretrained", _boom)
    monkeypatch.setitem(sys.modules, "transformers.models.qwen2_vl.video_processing_qwen2_vl", None)

    processor = _load_processor(str(processor_dir), local_files_only=True)

    assert isinstance(processor, Qwen2VLProcessor)
    assert not hasattr(processor, "video_processor")
    assert processor.get_attributes() == ["image_processor", "tokenizer"]
    assert processor.chat_template  # checkpoint-level template loaded, as AutoProcessor does


# ------------------------------------------------------------------ parity
def test_load_processor_matches_autoprocessor_composite(processor_dir, sample_image) -> None:
    """Same sub-components + the generic ``__call__`` -> bit-identical outputs.

    The reference is what the AutoProcessor route assembles on a full environment,
    with the video component stood in (class-checked at construction, never called for
    image/text inputs).
    """
    from transformers import AutoTokenizer, Qwen2VLProcessor
    from transformers.models.auto.image_processing_auto import AutoImageProcessor
    from transformers.processing_utils import ProcessorMixin

    ours = _load_processor(str(processor_dir), local_files_only=True)
    video_stand_in = object.__new__(ProcessorMixin.get_possibly_dynamic_module("BaseVideoProcessor"))
    reference = Qwen2VLProcessor(
        image_processor=AutoImageProcessor.from_pretrained(
            str(processor_dir), local_files_only=True, backend="pil"
        ),
        tokenizer=AutoTokenizer.from_pretrained(str(processor_dir), local_files_only=True),
        video_processor=video_stand_in,
        chat_template=ours.chat_template,
    )

    prompt = ours.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=False)
    assert prompt == reference.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=False)
    assert prompt.count("<|image_pad|>") == 1

    ours_out = ours(text=prompt, images=[sample_image], return_tensors="pt")
    ref_out = reference(text=prompt, images=[sample_image], return_tensors="pt")
    for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids"):
        assert torch.equal(ours_out[key], ref_out[key]), key

    # <|image_pad|> expands to exactly the grid-derived token count (2x2 spatial merge)
    ids = ours_out["input_ids"][0]
    grid = ours_out["image_grid_thw"][0].tolist()
    n_pads = int((ids == ours.image_token_id).sum())
    assert n_pads == int(np.prod(grid)) // MERGE**2 == IMAGE_TOKENS
    # M-RoPE labels mark exactly the image positions
    assert torch.equal(ours_out["mm_token_type_ids"][0].bool(), ids == ours.image_token_id)
    # production flattened patches: one row per 14px patch, channels*temporal*ph*pw wide
    assert tuple(ours_out["pixel_values"].shape) == (int(np.prod(grid)), 3 * 2 * 14 * 14)


def test_text_only_call_is_unchanged(processor_dir) -> None:
    processor = _load_processor(str(processor_dir), local_files_only=True)
    out = processor(text="The capital of France is", return_tensors="pt")
    assert "pixel_values" not in out
    assert "image_grid_thw" not in out
    assert int(out["mm_token_type_ids"].sum()) == 0
    assert "attention_mask" in out


# ------------------------------------------------------------------ real checkpoint
def _qwen2vl_processor_cached() -> bool:
    cache = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots"
    return cache.exists() and any((snapshot / "preprocessor_config.json").exists() for snapshot in cache.iterdir())


@pytest.mark.skipif(not _qwen2vl_processor_cached(), reason="Qwen2-VL-7B-Instruct processor files not cached")
def test_load_processor_accepts_the_real_repo_id(sample_image) -> None:
    """The exact id the box gate uses resolves offline from the HF cache."""
    processor = _load_processor("Qwen/Qwen2-VL-7B-Instruct", local_files_only=True)
    prompt = processor.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=False)
    out = processor(text=prompt, images=[sample_image], return_tensors="pt")
    grid = out["image_grid_thw"][0].tolist()
    assert grid == [1, 4, 4]
    assert int((out["input_ids"][0] == processor.image_token_id).sum()) == IMAGE_TOKENS
    assert tuple(out["pixel_values"].shape) == (int(np.prod(grid)), 3 * 2 * 14 * 14)
