# SPDX-License-Identifier: Apache-2.0
"""Synthetic image+prompt corpus for CPU dry runs and tests.

The fitting loop and every analysis path can be exercised end-to-end without COCO or a
real model: this builder writes deterministic random RGB images to disk and a manifest of
LLaVA-style prompts that reference them. It uses the same chat formatting as the caption
corpus, so prompt/token accounting matches production. Nothing here imports a model.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from vlm_lens.data.captions import DEFAULT_QUESTIONS, prompt_text
from vlm_lens.data.manifest import FitSample, write_manifest


def random_rgb_image(seed: int, *, image_size: int = 336) -> np.ndarray:
    """Deterministic random ``uint8`` HWC image (what a real photo loading path yields)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)


def build_dummy_manifest(
    out_dir: str | Path,
    *,
    n_samples: int = 4,
    image_size: int = 336,
    seed: int = 0,
    questions: Sequence[str] = DEFAULT_QUESTIONS,
    overwrite: bool = True,
) -> Path:
    """Write ``<out_dir>/images/dummy-*.png`` plus ``<out_dir>/manifest.jsonl``.

    Images are deterministic in ``seed``, so a dry run is reproducible and a manifest
    diff across runs only reflects code changes.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if not questions:
        raise ValueError("questions must not be empty")
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    samples: list[FitSample] = []
    for index in range(n_samples):
        path = images_dir / f"dummy-{index:03d}.png"
        if overwrite or not path.exists():
            Image.fromarray(random_rgb_image(seed + index, image_size=image_size)).save(path)
        question = questions[index % len(questions)]
        samples.append(
            FitSample(
                sample_id=f"dummy-{index:03d}",
                text=prompt_text(question),
                images=(str(path),),
                meta={"corpus": "dummy", "question": question, "image_seed": seed + index},
            )
        )
    return write_manifest(
        out_dir / "manifest.jsonl",
        samples,
        meta={
            "corpus": "dummy",
            "image_size": image_size,
            "seed": seed,
            "question_bank": list(questions),
        },
    )


__all__ = ["build_dummy_manifest", "random_rgb_image"]
