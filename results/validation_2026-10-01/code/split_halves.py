#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Split the 100-sample fit manifest into the two instruction-pool halves (A4/X8, D4).

Each image carries exactly one question, and the 10-question bank cycles, so the halves
are also image-disjoint: every sample's ``meta['question']`` decides its side.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from vlm_lens.data.manifest import manifest_meta, read_manifest, write_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split-json", required=True, help="corpus-split.json with half_a_questions")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
    half_a = set(split["half_a_questions"])
    samples = read_manifest(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    halves = {"a": [], "b": []}
    for sample in samples:
        halves["a" if sample.meta.get("question") in half_a else "b"].append(sample)
    source_meta = manifest_meta(args.manifest)
    for name, subset in halves.items():
        if not subset:
            raise SystemExit(f"half {name} is empty; check half_a_questions")
        path = write_manifest(
            out_dir / f"manifest-half-{name}.jsonl",
            subset,
            meta={
                "corpus": f"{source_meta.get('corpus')}:half-{name}",
                "parent_manifest": str(args.manifest),
                "half_a_questions": sorted(half_a),
                "question": sorted({sample.meta["question"] for sample in subset}),
                "prompt_template_sha256": source_meta.get("prompt_template_sha256"),
            },
        )
        images = sorted({sample.meta["image"] for sample in subset})
        print(f"half {name}: {len(subset)} samples, {len(images)} images -> {path}")
    overlap = {
        sample.meta["image"] for sample in halves["a"]
    } & {
        sample.meta["image"] for sample in halves["b"]
    }
    print(f"image overlap between halves: {len(overlap)} (must be 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
