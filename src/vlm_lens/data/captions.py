# SPDX-License-Identifier: Apache-2.0
"""COCO-captioning corpus: image + question + LLaVA's own greedy caption.

Why on-policy: hallucination analysis needs the lens to describe the residual stream in
the contexts where the model actually generates captions, including its own continuation
tokens. We therefore fit on the deployment-matched, teacher-forced sequence
``USER: <image>\\n{question}\\nASSISTANT: {greedy caption}`` and diversify the question
phrasing (the fitting average is over prompts; a single template over-weights one
phrasing).

Generation runs once, here, and is frozen into the manifest — the fit loop never calls
the generator.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from vlm_lens.data.manifest import FitSample, write_manifest

DEFAULT_QUESTIONS: tuple[str, ...] = (
    "Describe this image in detail.",
    "What is this image about?",
    "What do you see in this image?",
    "Write a short caption for this image.",
    "Summarize the scene in one sentence.",
    "Describe the scene and the objects in it.",
    "What is happening in this picture?",
    "Give a detailed account of the image contents.",
    "List the main objects in this image.",
    "Describe everything visible in this image.",
)


#: The LLaVA-1.5 chat template used by every captioning prompt. Changing it shifts the
#: distribution the lens averages over (V8), so the skeleton is hashed into provenance.
PROMPT_TEMPLATE = "USER: <image>\n{question}\nASSISTANT:"


def prompt_template_hash() -> str:
    """SHA-256 of the prompt skeleton, recorded in manifest and fit provenance (V8)."""
    return hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()


def prompt_text(question: str, caption: str | None = None) -> str:
    """LLaVA-1.5 chat format, teacher-forced on the caption when given."""
    base = PROMPT_TEMPLATE.format(question=question)
    if caption:
        return f"{base} {caption}"
    return base


def sample_questions(
    n_images: int,
    *,
    questions_per_image: int = 1,
    seed: int = 0,
    bank: Sequence[str] = DEFAULT_QUESTIONS,
) -> list[list[str]]:
    """Per-image question sets: consecutive images cycle through the bank (diverse),
    then a seeded shuffle breaks the correlation between index and phrasing."""
    rng = random.Random(seed)
    bank = list(bank)
    per_image: list[list[str]] = []
    cursor = rng.randrange(len(bank))
    for _ in range(n_images):
        picked: list[str] = []
        for _ in range(questions_per_image):
            picked.append(bank[cursor % len(bank)])
            cursor += 1
        per_image.append(picked)
    rng.shuffle(per_image)
    return per_image


def list_coco_images(images_dir: str | Path, *, pattern: str = "*.jpg") -> list[Path]:
    files = sorted(Path(images_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"no images matching {pattern} under {images_dir}")
    return files


def select_images(
    images: Sequence[Path],
    n_images: int,
    *,
    seed: int = 0,
    stride: int | None = None,
) -> list[Path]:
    """Deterministic selection: an evenly spaced slice by default, else seeded sampling."""
    if n_images >= len(images):
        return list(images)
    if stride is not None:
        return list(images)[::stride][:n_images]
    rng = random.Random(seed)
    return sorted(rng.sample(list(images), n_images))


def _open_image(path: str | Path) -> Image.Image:
    """RGB PIL image for ``path`` - the decoder HF and we agree on.

    Passed as PIL, the processor decodes with transformers' own image utilities; passed
    as ``str``/``bytes`` it goes through ``torchvision.io.decode_image``, which dies on
    torchvision builds without libjpeg (the cluster env carries an editable torchvision
    from another project that lacks it), and ``pathlib.Path`` is rejected outright.
    ``LlavaLensModel.load_image`` resolves images the same way.
    """
    with Image.open(path) as image:
        return image.convert("RGB")


@torch.no_grad()
def generate_captions(
    model: Any,
    image_paths: Sequence[Path],
    question: str,
    *,
    batch_size: int = 4,
    max_new_tokens: int = 64,
) -> list[str]:
    """Greedy captions for one question over a batch of images (uniform lengths).

    Uses the model's own processor + ``generate`` so prompts, image preprocessing and
    chat format are exactly the deployment path. Images are handed over as PIL objects
    (see :func:`_open_image`) rather than paths, for the decoder reason documented there.
    """
    captions: list[str] = []
    for start in range(0, len(image_paths), batch_size):
        # PIL, not paths: HF decodes str/bytes through torchvision's jpeg decoder.
        chunk = [_open_image(path) for path in image_paths[start : start + batch_size]]
        texts = [prompt_text(question) for _ in chunk]
        inputs = model.processor(images=chunk, text=texts, return_tensors="pt", padding=True)
        inputs = {
            key: value.to(model.input_device if key != "pixel_values" else model.vision_device)
            for key, value in inputs.items()
        }
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.vision_dtype)
        padded_len = int(inputs["input_ids"].shape[1])
        generated = model.hf_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        new_tokens = generated[:, padded_len:]
        for row in new_tokens:
            captions.append(model._decode(row).strip())
    return captions


def _question_slug(question: str, *, length: int = 8) -> str:
    """Stable per-question id component (``hash()`` is salted per process)."""
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:length]


def build_coco_caption_manifest(
    out_path: str | Path,
    *,
    images_dir: str | Path,
    image_list: Sequence[str | Path] | None = None,
    model: Any,
    n_images: int = 100,
    questions_per_image: int = 1,
    max_new_tokens: int = 64,
    batch_size: int = 4,
    seed: int = 0,
    prompt_mode: str = "prompt+caption",
    max_seq_len: int | None = None,
    questions: Sequence[str] = DEFAULT_QUESTIONS,
) -> Path:
    """Generate captions and freeze the fitting prompts into a manifest.

    Args:
        image_list: Use exactly these images (paths) instead of sampling from
            ``images_dir``; lets a caller own the fit/eval split.
        prompt_mode: ``"prompt+caption"`` (deployment-matched, default),
            ``"prompt"`` (image + question only), or ``"both"`` (both variants as
            separate samples).
        max_seq_len: Drop samples whose tokenized length exceeds this (keeps the fit
            free of truncation surprises); reported in the header meta.
    """
    if prompt_mode not in {"prompt", "prompt+caption", "both"}:
        raise ValueError(f"unknown prompt_mode {prompt_mode!r}")

    if image_list is not None:
        chosen = [Path(path) for path in image_list]
        names = [path.name for path in chosen]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate images in image_list: {duplicates[:5]}")
        missing = [str(path) for path in chosen if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"image_list entries do not exist: {missing[:5]}")
    else:
        chosen = select_images(list_coco_images(images_dir), n_images, seed=seed)
    question_sets = sample_questions(len(chosen), questions_per_image=questions_per_image, seed=seed, bank=questions)

    samples: list[FitSample] = []
    dropped: list[str] = []
    by_question: dict[str, list[int]] = {}
    for index, question_list in enumerate(question_sets):
        for question in question_list:
            by_question.setdefault(question, []).append(index)

    captions_by_pair: dict[tuple[int, str], str] = {}
    for question, indices in by_question.items():
        captions = generate_captions(
            model,
            [chosen[i] for i in indices],
            question,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )
        for index, caption in zip(indices, captions, strict=True):
            captions_by_pair[(index, question)] = caption

    for index, question_list in enumerate(question_sets):
        for question in question_list:
            caption = captions_by_pair[(index, question)]
            variants: list[tuple[str, str]] = []
            if prompt_mode in {"prompt", "both"}:
                variants.append(("prompt", prompt_text(question)))
            if prompt_mode in {"prompt+caption", "both"}:
                variants.append(("prompt+caption", prompt_text(question, caption)))
            for variant, text in variants:
                sample_id = (
                    f"{chosen[index].stem}::{variant}::{_question_slug(question)}"
                )
                if max_seq_len is not None:
                    n_tokens = len(
                        model.processor.tokenizer(text, add_special_tokens=True)["input_ids"]
                    ) + getattr(model, "image_seq_length", 0)
                    if n_tokens > max_seq_len:
                        dropped.append(sample_id)
                        continue
                samples.append(
                    FitSample(
                        sample_id=sample_id,
                        text=text,
                        images=(str(chosen[index]),),
                        meta={
                            "corpus": "coco_val2014",
                            "question": question,
                            "caption": caption,
                            "variant": variant,
                            "image": chosen[index].name,
                            "generation": {
                                "do_sample": False,
                                "max_new_tokens": max_new_tokens,
                                "model": type(model).__name__,
                            },
                        },
                    )
                )

    meta = {
        "corpus": "coco_val2014_captions",
        "images_dir": str(images_dir),
        "n_images": len(chosen),
        "image_list_given": image_list is not None,
        "image_names": [path.name for path in chosen],
        "questions_per_image": questions_per_image,
        "question_bank": list(questions),
        "question_sampling": "cycle+shuffle (seeded)",
        "seed": seed,
        "prompt_mode": prompt_mode,
        "prompt_template": PROMPT_TEMPLATE,
        "prompt_template_sha256": prompt_template_hash(),
        "max_new_tokens": max_new_tokens,
        "max_seq_len": max_seq_len,
        "n_dropped_over_length": len(dropped),
    }
    return write_manifest(out_path, samples, meta=meta)
