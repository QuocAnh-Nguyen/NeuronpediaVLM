#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Data-scaling corpus: caption manifests at n=100/500/1000, disjoint from the held-out.

D40 (P2) measured the translator scaling flat from n=20 to n=100 - the limiter was the
estimator's model class, not sample count - so extending the range needs bigger corpora:
the translator experiment refits at n=100/500/1000. Only ~420 caption rows exist in the
step manifests today and the 300-sample held-out must stay disjoint from every fit, so
the image split is owned here, not by the builder's internal sampling: list the images
directory, drop every filename any exclude manifest already uses, and take the first N
of the remainder - a deterministic prefix, so the n=100 corpus is nested inside n=500
inside n=1000 and the scaling points never reshuffle.

Captions are model-generated, not ground truth: :func:`build_coco_caption_manifest` runs
LLaVA's own greedy ``generate`` (the deployment path), so the model is loaded here exactly
as the other campaign scripts load it (backend ``tiny`` => the tiny CPU fixture,
``hf-llava`` => bf16 CUDA). ``--limit`` caps the run to a smoke.

Row shapes observed in the exclude manifests (results/validation_2026-10-01/raw/):
fit and held-out caption rows carry ``meta["image"]`` (``"COCO_val2014_000000012966.jpg"``)
and ``images`` (the full path); text (wikitext) rows carry ``images: []`` and no
``meta["image"]``, so they contribute nothing to the exclusion set. The exclusion key is
the image basename, not ``sample_id``: the held-out carries ten ids per image (one per
question), so ids would under-exclude.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402, I001  # the vlm_lens import must precede jlens: it installs the path
import vlm_lens  # noqa: E402, F401

from vlm_lens.data.captions import (  # noqa: E402
    build_coco_caption_manifest,
    list_coco_images,
)
from vlm_lens.data.manifest import manifest_meta, read_manifest  # noqa: E402
from vlm_lens.models.llava import LlavaLensModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir", required=True, help="directory of COCO val2014 *.jpg images"
    )
    parser.add_argument(
        "--n", type=int, default=1000, help="manifest sample count (one question per image)"
    )
    parser.add_argument("--out", default="manifest-scaling.jsonl", help="manifest path to write")
    parser.add_argument(
        "--exclude-manifests", default="",
        help="comma-separated manifests whose image filenames must not be reused "
        "(the 300-sample held-out, existing fit manifests)",
    )
    parser.add_argument(
        "--backend", choices=("hf-llava", "tiny"), default="hf-llava",
        help="model backend: 'hf-llava' (default, CUDA) or the tiny CPU smoke fixture",
    )
    parser.add_argument("--dtype", default="bfloat16", help="torch dtype for the model weights")
    parser.add_argument(
        "--limit", type=int, default=None, help="use only the first N images (smoke run; caps --n)"
    )
    parser.add_argument("--max-seq-len", type=int, default=1536, help="drop samples over length")
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> LlavaLensModel:
    """The builder model: the HF checkpoint (default, CUDA) or the tiny CPU fixture."""
    if args.backend == "tiny":
        from vlm_lens.models.tiny_llava import TinyLlavaConfig, build_tiny_llava

        hf_model, processor = build_tiny_llava(TinyLlavaConfig())
        return LlavaLensModel(hf_model, processor)
    return LlavaLensModel.from_pretrained(
        dtype=getattr(torch, args.dtype), device="cuda", local_files_only=True
    )


def collect_excluded_images(exclude_text: str) -> tuple[list[str], set[str]]:
    """Image basenames already used by the exclude manifests (must not be reused).

    The exclusion key is the image basename: ``meta["image"]`` when present (the caption
    corpora record it), else the basename of ``images[0]``; text-only samples
    (``images == []``, no ``meta["image"]``) contribute nothing. ``sample_id`` is not
    used - the held-out carries ten ids per image (one per question), so ids would
    under-exclude.
    """
    paths: list[str] = []
    names: set[str] = set()
    for entry in exclude_text.split(","):
        entry = entry.strip()
        if not entry:
            continue
        paths.append(entry)
        for sample in read_manifest(entry):
            image = sample.meta.get("image")
            if image:
                names.add(Path(image).name)
            elif sample.images:
                names.add(Path(sample.images[0]).name)
    return paths, names


def select_unused_images(images_dir: str, n_images: int, excluded: set[str]) -> list[Path]:
    """The first N images of the sorted listing that no exclude manifest uses.

    A deterministic prefix (sorted filenames, stable skip set), not the builder's seeded
    even slice: the n=100 corpus is a prefix of n=500, which is a prefix of n=1000, so
    the scaling points stay nested and the builder's ``select_images`` sampling never
    re-enters the split.
    """
    available = [path for path in list_coco_images(images_dir) if path.name not in excluded]
    if len(available) < n_images:
        raise SystemExit(
            f"need {n_images} unused images but only {len(available)} remain under "
            f"{images_dir} after excluding {len(excluded)} filename(s) that the "
            "exclude manifests already use"
        )
    return available[:n_images]


def main() -> int:
    args = parse_args()
    exclude_paths, excluded = collect_excluded_images(args.exclude_manifests)
    n_images = args.n if args.limit is None else min(args.n, args.limit)
    chosen = select_unused_images(args.images_dir, n_images, excluded)
    print(
        f"selected the first {len(chosen)} of the sorted listing ({n_images} requested); "
        f"excluded {len(excluded)} filename(s) from {len(exclude_paths)} manifest(s)"
    )

    model = load_model(args)
    print(
        f"model ready: backend={args.backend} layers={model.n_layers} "
        f"d_model={model.d_model} image_seq_length={model.image_seq_length}"
    )

    out_path = build_coco_caption_manifest(
        args.out,
        images_dir=args.images_dir,
        model=model,
        image_list=chosen,
        prompt_mode="prompt+caption",
        max_seq_len=args.max_seq_len,
    )
    meta = manifest_meta(out_path)
    print(
        f"wrote {out_path}: n_samples={meta.get('n_samples')} n_images={meta.get('n_images')}"
    )
    if meta.get("n_dropped_over_length"):
        print(f"dropped over length ({args.max_seq_len}): {meta['n_dropped_over_length']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
