#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Build a model-generated caption manifest with a controlled caption budget.

The 64-token default truncates the model's description of the image; the budget round
(D48) re-fits at ``--max-new-tokens 256`` on the SAME seeded image sample, so the only
changed variable is the caption budget. Uses the registry dispatch (same wrapper the
fits use) so each model captions with its own chat template.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402

import vlm_lens  # noqa: E402, F001

from vlm_lens.data.captions import build_coco_caption_manifest  # noqa: E402
from vlm_lens.models.registry import MODEL_SPECS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--images-dir", default="/data/baodq/coco2014/val2014")
    parser.add_argument("--n-images", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    spec = MODEL_SPECS[args.model]
    model = spec.lens_model_cls().from_pretrained(
        spec.hf_id, dtype=torch.bfloat16, device="cuda", local_files_only=True
    )
    out = build_coco_caption_manifest(
        args.out,
        images_dir=args.images_dir,
        model=model,
        n_images=args.n_images,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    print(f"manifest: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
