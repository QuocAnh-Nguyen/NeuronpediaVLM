# SPDX-License-Identifier: Apache-2.0
"""POPE / yes-no samples (the VQA entry point).

The fit protocol is task-agnostic: a sample is an image plus a teacher-forced prompt.
For POPE that is ``USER: <image>\\n{question}\\nASSISTANT: {label}`` so the lens is
averaged over answer-position contexts; analysis then reads the lens log-probabilities
of "Yes"/"No" per layer to compare internal belief with the emitted answer.

Regenerated locally (per project decision D-06) rather than reading the sibling
``VLM_Hallu_VCD`` artifacts, so provenance stays inside this repo.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from vlm_lens.data.manifest import FitSample, write_manifest

POPE_TEMPLATE = "USER: <image>\n{question}\nASSISTANT:"


def read_pope_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read POPE records: ``question_id``, ``image``, ``text``, ``label``."""
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def iter_pope_samples(
    records: list[dict[str, Any]],
    *,
    images_dir: str | Path,
    include_answer: bool = True,
    limit: int | None = None,
) -> Iterator[FitSample]:
    images_dir = Path(images_dir)
    count = 0
    for record in records:
        if limit is not None and count >= limit:
            break
        image_name = str(record["image"])
        question = str(record.get("text") or record.get("question") or "")
        label = str(record.get("label", "")).strip().lower()
        text = POPE_TEMPLATE.format(question=question)
        if include_answer and label in {"yes", "no"}:
            text = f"{text} {label.capitalize()}"
        yield FitSample(
            sample_id=f"pope-{record.get('question_id', count)}",
            text=text,
            images=(str(images_dir / image_name),),
            meta={
                "corpus": "pope",
                "question_id": record.get("question_id"),
                "question": question,
                "label": label,
                "image": image_name,
            },
        )
        count += 1


def build_pope_manifest(
    out_path: str | Path,
    *,
    pope_jsonl: str | Path,
    images_dir: str | Path,
    include_answer: bool = True,
    limit: int | None = None,
) -> Path:
    records = read_pope_jsonl(pope_jsonl)
    samples = list(
        iter_pope_samples(records, images_dir=images_dir, include_answer=include_answer, limit=limit)
    )
    meta = {
        "corpus": "pope",
        "pope_jsonl": str(pope_jsonl),
        "images_dir": str(images_dir),
        "include_answer": include_answer,
        "limit": limit,
        "n_records_available": len(records),
        "template": POPE_TEMPLATE,
    }
    return write_manifest(out_path, samples, meta=meta)
