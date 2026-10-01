#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Step 2 corpus: WikiText-103 text-only manifests (fit = train, held out = validation).

The held-out shard must be disjoint from the fitting prompts, which is why ``split`` is
plumbed into the manifest header. Only the tokenizer is loaded (no 15 GB model): the
manifest builder needs it for the over-length filter. The WikiText stream needs the
network, the fit itself does not.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

MODEL_ID = "llava-hf/llava-1.5-7b-hf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="run directory for manifests")
    parser.add_argument("--n-fit", type=int, default=100)
    parser.add_argument("--n-heldout", type=int, default=30)
    parser.add_argument("--min-chars", type=int, default=600)
    parser.add_argument("--max-tokens", type=int, default=1536)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.pop("HF_HUB_OFFLINE", None)  # the WikiText stream is a download
    from transformers import AutoTokenizer

    from vlm_lens.data.manifest import manifest_meta, read_manifest
    from vlm_lens.data.text import build_text_manifest

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)

    fit_path = build_text_manifest(
        out / "manifest-text-fit.jsonl",
        n_prompts=args.n_fit,
        source="wikitext",
        split="train",
        min_chars=args.min_chars,
        max_tokens=args.max_tokens,
        tokenizer=tokenizer,
    )
    heldout_path = build_text_manifest(
        out / "manifest-text-heldout.jsonl",
        n_prompts=args.n_heldout,
        source="wikitext",
        split="validation",
        min_chars=args.min_chars,
        max_tokens=args.max_tokens,
        tokenizer=tokenizer,
    )

    report = {"fit": manifest_meta(fit_path), "heldout": manifest_meta(heldout_path)}
    for name, path in (("fit", fit_path), ("heldout", heldout_path)):
        lengths = sorted(
            len(tokenizer(sample.text, add_special_tokens=True)["input_ids"])
            for sample in read_manifest(path)
        )
        report[name]["token_lengths"] = {
            "min": lengths[0],
            "median": lengths[len(lengths) // 2],
            "max": lengths[-1],
        }
        print(f"{name}: n={len(lengths)} tokens min/median/max = {lengths[0]}/{lengths[len(lengths) // 2]}/{lengths[-1]}")
    (out / "text-corpus.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
