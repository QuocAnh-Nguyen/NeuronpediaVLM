#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Step 0d: measured seconds/sample and peak GPU memory, plus the rule-3 cost table.

Measures one or two real samples on the image corpus and (optionally) the text corpus
with the exact fit settings used later (all source layers, three masks, the corpus's
``skip_first``). The projection table is what autonomy rule 3 consumes: any run whose
projected cost exceeds the remaining budget must shrink ``n_samples`` instead.
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

from vlm_lens.data.manifest import read_manifest  # noqa: E402
from vlm_lens.fitting import jacobian_for_sample  # noqa: E402
from vlm_lens.models.llava import LlavaLensModel  # noqa: E402

TARGET_SAMPLE_COUNTS = (30, 50, 100, 200, 500)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-manifest")
    parser.add_argument("--text-manifest")
    parser.add_argument("--n", type=int, default=1, help="samples per corpus")
    parser.add_argument("--dim-batch", type=int, default=8)
    parser.add_argument("--masks", default="text,image,all")
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--json", required=True)
    return parser.parse_args()


def measure(model, manifest: str, *, n: int, skip_first: int, args) -> dict[str, object]:
    samples = read_manifest(manifest)[:n]
    masks = [part.strip() for part in args.masks.split(",") if part.strip()]
    torch.cuda.reset_peak_memory_stats()
    seconds: list[float] = []
    for sample in samples:
        _, info = jacobian_for_sample(
            model,
            sample,
            source_layers=None,
            dim_batch=args.dim_batch,
            max_seq_len=args.max_seq_len,
            skip_first=skip_first,
            masks=masks,
        )
        seconds.append(info.seconds)
        print(
            f"  {sample.sample_id}: seq={info.seq_len} images={info.n_image_tokens} "
            f"positions={info.mask_positions} {info.seconds:.1f}s"
        )
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    mean_seconds = sum(seconds) / len(seconds)
    return {
        "manifest": manifest,
        "n": len(samples),
        "skip_first": skip_first,
        "seconds_per_sample": [round(value, 1) for value in seconds],
        "mean_seconds": round(mean_seconds, 1),
        "peak_gib": round(peak_gib, 2),
        "projection_hours": {
            str(count): round(mean_seconds * count / 3600, 2) for count in TARGET_SAMPLE_COUNTS
        },
    }


def main() -> int:
    args = parse_args()
    model = LlavaLensModel.from_pretrained(dtype=torch.bfloat16, device="cuda", local_files_only=True)
    report: dict[str, object] = {
        "dim_batch": args.dim_batch,
        "masks": args.masks,
        "layers": "all below target",
        "dtype": "bfloat16",
        "image_seq_length": model.image_seq_length,
        "n_layers": model.n_layers,
        "d_model": model.d_model,
    }
    for name, manifest, skip_first in (
        ("image", args.image_manifest, 1),
        ("text", args.text_manifest, 16),
    ):
        if not manifest:
            continue
        print(f"measuring {name} corpus: {manifest}")
        report[name] = measure(model, manifest, n=args.n, skip_first=skip_first, args=args)

    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\npeak_gib by corpus: " + ", ".join(
        f"{name}={report[name]['peak_gib']}" for name in ("image", "text") if name in report
    ))
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
