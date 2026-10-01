# SPDX-License-Identifier: Apache-2.0
"""On-disk prompt manifests: a JSONL header plus one line per ``FitSample``.

A manifest is the contract between corpus construction (which may run the model, e.g.
to generate on-policy captions) and the fitting loop (which must not regenerate data).
The header carries the corpus provenance and the sample count, so a fit can verify it is
reproducing the run it claims to.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1


@dataclass(frozen=True)
class FitSample:
    """One fitting sample: a prompt, its images, and whatever produced them."""

    sample_id: str
    text: str
    images: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "text": self.text,
            "images": list(self.images),
            "meta": self.meta,
        }

    @classmethod
    def from_json(cls, record: dict[str, Any]) -> FitSample:
        return cls(
            sample_id=str(record["sample_id"]),
            text=str(record["text"]),
            images=tuple(record.get("images") or ()),
            meta=dict(record.get("meta") or {}),
        )


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_manifest(
    path: str | Path,
    samples: Iterable[FitSample],
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write a manifest; the first line is a header, the rest are samples."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_list = list(samples)
    ids = [sample.sample_id for sample in sample_list]
    if len(set(ids)) != len(ids):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate sample_id(s) in manifest: {duplicates[:5]}")
    header = {
        "type": "header",
        "version": MANIFEST_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_samples": len(sample_list),
        "meta": meta or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(header, ensure_ascii=False) + "\n")
        for sample in sample_list:
            handle.write(json.dumps(sample.to_json(), ensure_ascii=False) + "\n")
    tmp.replace(path)
    return path


def read_manifest(path: str | Path) -> list[FitSample]:
    """Read samples and verify the header count (catches truncated manifests)."""
    samples: list[FitSample] = []
    header: dict[str, Any] | None = None
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if line_number == 1:
                header = record
                if record.get("type") != "header":
                    raise ValueError(f"{path}: first line is not a manifest header")
                if int(record.get("version", 0)) != MANIFEST_VERSION:
                    raise ValueError(
                        f"{path}: manifest version {record.get('version')!r} "
                        f"!= supported {MANIFEST_VERSION}"
                    )
                continue
            samples.append(FitSample.from_json(record))
    if header is not None and len(samples) != int(header.get("n_samples", -1)):
        raise ValueError(
            f"{path}: header says {header.get('n_samples')} samples, found {len(samples)}"
        )
    return samples


def manifest_meta(path: str | Path) -> dict[str, Any]:
    """Header metadata (provenance) of a manifest."""
    with open(path, encoding="utf-8") as handle:
        header = json.loads(handle.readline())
    if header.get("type") != "header":
        raise ValueError(f"{path}: first line is not a manifest header")
    return {"version": header.get("version"), "n_samples": header.get("n_samples"), **header.get("meta", {})}


def sample_token_lengths(tokenizer, text: str) -> int:  # pragma: no cover - convenience
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])
