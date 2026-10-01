# SPDX-License-Identifier: Apache-2.0
"""Text-only corpora for the S1 control fit (WikiText-103 by default).

The control fit validates that our port of the estimator reproduces the paper's
behaviour on the corpus they used, before any multimodal machinery is involved.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from vlm_lens.data.manifest import FitSample, write_manifest


def load_wikitext_prompts(n_prompts: int, *, min_chars: int = 600, split: str = "train") -> list[str]:
    """First ``n_prompts`` WikiText-103 records of at least ``min_chars`` characters.

    Mirrors the reference implementation's loader (streaming, requires ``datasets``).
    """
    if n_prompts <= 0:
        return []
    from datasets import load_dataset

    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split=split, streaming=True
    )
    prompts: list[str] = []
    for record in dataset:
        text = record["text"]
        if len(text) >= min_chars and not text.lstrip().startswith("="):
            prompts.append(text.strip())
            if len(prompts) >= n_prompts:
                break
    if len(prompts) < n_prompts:
        raise RuntimeError(f"only found {len(prompts)} WikiText records >= {min_chars} chars")
    return prompts


def load_text_file_prompts(path: str | Path, n_prompts: int, *, min_chars: int = 600) -> list[str]:
    """Prompts from a ``.txt`` (blank-line separated blocks) or ``.jsonl`` (``text``)."""
    path = Path(path)
    prompts: list[str] = []
    if path.suffix == ".jsonl":
        import json

        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                text = str(json.loads(line).get("text", "")).strip()
                if len(text) >= min_chars:
                    prompts.append(text)
                if len(prompts) >= n_prompts:
                    break
    else:
        block: list[str] = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    block.append(line.rstrip())
                elif block:
                    text = "\n".join(block).strip()
                    if len(text) >= min_chars:
                        prompts.append(text)
                    block = []
                    if len(prompts) >= n_prompts:
                        break
        if block and len(prompts) < n_prompts:
            text = "\n".join(block).strip()
            if len(text) >= min_chars:
                prompts.append(text)
    if len(prompts) < n_prompts:
        raise RuntimeError(f"only found {len(prompts)} text blocks >= {min_chars} chars in {path}")
    return prompts[:n_prompts]


def build_text_manifest(
    out_path: str | Path,
    *,
    n_prompts: int,
    source: str = "wikitext",
    path: str | Path | None = None,
    min_chars: int = 600,
    split: str = "train",
    max_tokens: int | None = None,
    tokenizer: Any | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Freeze a text-only corpus into a manifest (no images)."""
    if source == "wikitext":
        prompts = load_wikitext_prompts(n_prompts, min_chars=min_chars, split=split)
    elif source == "file":
        if path is None:
            raise ValueError("source='file' requires path=")
        prompts = load_text_file_prompts(path, n_prompts, min_chars=min_chars)
    else:
        raise ValueError(f"unknown text source {source!r}")

    samples: list[FitSample] = []
    dropped = 0
    for index, text in enumerate(prompts):
        if max_tokens is not None and tokenizer is not None:
            n_tokens = len(tokenizer(text, add_special_tokens=True)["input_ids"])
            if n_tokens > max_tokens:
                dropped += 1
                continue
        samples.append(
            FitSample(
                sample_id=f"{source}-{index:05d}",
                text=text,
                images=(),
                meta={"corpus": source, "source_path": str(path) if path else None},
            )
        )
    meta = {
        "corpus": source,
        "source_path": str(path) if path else None,
        "min_chars": min_chars,
        "split": split if source == "wikitext" else None,
        "max_tokens": max_tokens,
        "n_dropped_over_length": dropped,
        **(extra_meta or {}),
    }
    return write_manifest(out_path, samples, meta=meta)


def iter_prompts(samples: Iterable[FitSample]) -> list[str]:
    return [sample.text for sample in samples]
