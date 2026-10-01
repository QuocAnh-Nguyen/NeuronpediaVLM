#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Step 3 corpus: freeze the COCO caption manifests (fit images and held-out images).

Run once per campaign; the resulting JSONL files are the only inputs the fits see, so the
fit runs are offline and reproducible. The image split is owned here (D4), not by the
builder's internal sampling: 100 fit images and 30 held-out images, disjoint by
construction, recorded in a sidecar JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
import torch  # noqa: E402

from vlm_lens.data.captions import (  # noqa: E402
    DEFAULT_QUESTIONS,
    build_coco_caption_manifest,
    list_coco_images,
    select_images,
)
from vlm_lens.data.manifest import manifest_meta, read_manifest  # noqa: E402
from vlm_lens.models.llava import LlavaLensModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--out", required=True, help="run directory for manifests")
    parser.add_argument("--n-fit", type=int, default=100)
    parser.add_argument("--n-eval", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    images = select_images(list_coco_images(args.images_dir), args.n_fit + args.n_eval, seed=args.seed)
    fit_images, eval_images = images[: args.n_fit], images[args.n_fit :]
    print(f"selected {len(images)} images: {len(fit_images)} fit / {len(eval_images)} held out")

    model = LlavaLensModel.from_pretrained(
        dtype=torch.bfloat16, device="cuda", local_files_only=True
    )

    fit_path = build_coco_caption_manifest(
        out / "manifest-fit.jsonl",
        images_dir=args.images_dir,
        model=model,
        image_list=fit_images,
        questions_per_image=1,
        seed=args.seed,
        prompt_mode="prompt+caption",
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    eval_path = build_coco_caption_manifest(
        out / "manifest-heldout.jsonl",
        images_dir=args.images_dir,
        model=model,
        image_list=eval_images,
        questions_per_image=len(DEFAULT_QUESTIONS),
        seed=args.seed,
        prompt_mode="prompt+caption",
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    sidecar = {
        "images_dir": args.images_dir,
        "seed": args.seed,
        "fit_images": [path.name for path in fit_images],
        "held_out_images": [path.name for path in eval_images],
        "half_a_questions": list(DEFAULT_QUESTIONS[:5]),
        "half_b_questions": list(DEFAULT_QUESTIONS[5:]),
        "fit_manifest": str(fit_path),
        "heldout_manifest": str(eval_path),
        "fit_meta": manifest_meta(fit_path),
        "heldout_meta": manifest_meta(eval_path),
    }
    (out / "corpus-split.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    fit_samples = read_manifest(fit_path)
    eval_samples = read_manifest(eval_path)
    print(f"fit manifest: {len(fit_samples)} samples, meta={json.dumps(manifest_meta(fit_path))}")
    print(f"held-out manifest: {len(eval_samples)} samples")
    questions = sorted({sample.meta["question"] for sample in fit_samples})
    print(f"fit questions ({len(questions)}): {questions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
