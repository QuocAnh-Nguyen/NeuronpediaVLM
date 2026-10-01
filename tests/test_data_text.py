# SPDX-License-Identifier: Apache-2.0
"""Text-corpus manifests: local-file round trip and the WikiText split provenance.

The S1 control fit needs a held-out shard disjoint from the fitting prompts; WikiText
provides that via ``split``, which must therefore reach the loader and be recorded in the
manifest header (the fit's provenance copies the header verbatim).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vlm_lens.data.manifest import manifest_meta, read_manifest
from vlm_lens.data.text import build_text_manifest, load_text_file_prompts


@pytest.fixture()
def text_file(tmp_path: Path) -> Path:
    """Four blank-line-separated blocks, each long enough for ``min_chars=100``."""
    path = tmp_path / "corpus.txt"
    blocks = [f"Block {index} " + "word " * 200 for index in range(4)]
    path.write_text("\n\n".join(blocks), encoding="utf-8")
    return path


def test_build_text_manifest_round_trip(text_file: Path, tmp_path: Path):
    out = build_text_manifest(
        tmp_path / "m.jsonl", n_prompts=3, source="file", path=text_file, min_chars=100
    )
    samples = read_manifest(out)
    assert [sample.sample_id for sample in samples] == ["file-00000", "file-00001", "file-00002"]
    assert samples[0].text.startswith("Block 0")
    assert all(sample.images == () for sample in samples)
    meta = manifest_meta(out)
    assert meta["corpus"] == "file"
    assert meta["n_dropped_over_length"] == 0
    assert meta["split"] is None


def test_wikitext_split_reaches_loader_and_header(monkeypatch, tmp_path: Path):
    seen: dict[str, object] = {}

    def fake_loader(n_prompts: int, *, min_chars: int = 600, split: str = "train"):
        seen["n_prompts"] = n_prompts
        seen["split"] = split
        return ["x" * (min_chars + 1)] * n_prompts

    monkeypatch.setattr("vlm_lens.data.text.load_wikitext_prompts", fake_loader)
    out = build_text_manifest(
        tmp_path / "m.jsonl", n_prompts=2, source="wikitext", min_chars=100, split="validation"
    )
    assert seen == {"n_prompts": 2, "split": "validation"}
    meta = manifest_meta(out)
    assert meta["corpus"] == "wikitext"
    assert meta["split"] == "validation"
    assert len(read_manifest(out)) == 2


def test_too_few_blocks_raises(text_file: Path):
    with pytest.raises(RuntimeError, match="text blocks"):
        load_text_file_prompts(text_file, 99, min_chars=100)
