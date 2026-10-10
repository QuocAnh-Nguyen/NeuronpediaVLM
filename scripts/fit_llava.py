#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Fit J lenses for LLaVA-1.5 (or the tiny CPU fixture) and write artifacts.

Corpus construction, fitting and artifact writing in one place; every step is a library
call from ``vlm_lens`` so the scripts stay thin. Examples:

    # local CPU dry run: random-weight model, synthetic images, no downloads
    python scripts/fit_llava.py --backend tiny --corpus dummy --n-samples 4 \\
        --out runs/dry-fit

    # H100 fit on LLaVA's own COCO val2014 captions
    python scripts/fit_llava.py --backend hf-llava --corpus coco \\
        --images-dir data/val2014 --n-images 100 --out runs/coco-100

    # fit a pre-built manifest in two shards, then merge
    python scripts/fit_llava.py --manifest manifests/coco.jsonl --shard 0/2 --out runs/s0
    python scripts/fit_llava.py --manifest manifests/coco.jsonl --shard 1/2 --out runs/s1
    python scripts/fit_llava.py --merge runs/s0 runs/s1 --out runs/coco-100
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vlm_lens.artifacts import (  # noqa: E402
    build_provenance,
    load_lens_set,
    merge_shards,
    save_lens_set,
)
from vlm_lens.data.captions import build_coco_caption_manifest  # noqa: E402
from vlm_lens.data.dummy import build_dummy_manifest  # noqa: E402
from vlm_lens.data.manifest import manifest_meta, read_manifest  # noqa: E402
from vlm_lens.data.pope import build_pope_manifest  # noqa: E402
from vlm_lens.data.text import build_text_manifest  # noqa: E402
from vlm_lens.fitting import (  # noqa: E402
    MM_SKIP_FIRST,
    TEXT_SKIP_FIRST,
    configure_tf32,
    convergence_summary,
    drop_stats,
    fit_masked,
)
from vlm_lens.models.llava import LlavaLensModel  # noqa: E402
from vlm_lens.models.tiny_llava import TinyLlavaConfig, build_tiny_llava  # noqa: E402

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
CORPORA = ("dummy", "coco", "pope", "text")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--merge", nargs="+", metavar="SHARD_DIR", help="merge fitted shards")
    parser.add_argument("--out", required=True, help="output directory (checkpoint + artifacts)")

    model = parser.add_argument_group("model")
    model.add_argument("--backend", choices=("hf-llava", "tiny"), default="hf-llava")
    model.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    model.add_argument("--dtype", choices=tuple(DTYPES), default=None, help="default: bf16 on CUDA, fp32 on CPU")
    model.add_argument("--device", default=None, help="default: cuda when available else cpu")
    model.add_argument("--attn-implementation", default=None)
    model.add_argument("--compile", action="store_true", help="torch.compile each residual block")
    model.add_argument("--local-files-only", action="store_true")
    model.add_argument("--tiny-image-size", type=int, default=336, help="tiny backend only")

    data = parser.add_argument_group("data")
    data.add_argument("--manifest", help="fit a pre-built manifest (mutually exclusive with --corpus)")
    data.add_argument("--corpus", choices=CORPORA, help="build a manifest before fitting")
    data.add_argument("--images-dir", help="image directory (coco / pope)")
    data.add_argument("--n-samples", type=int, default=4, help="dummy corpus size")
    data.add_argument("--n-images", type=int, default=100, help="coco corpus size")
    data.add_argument("--questions-per-image", type=int, default=1, help="coco")
    data.add_argument("--prompt-mode", choices=("prompt+caption", "prompt"), default="prompt+caption")
    data.add_argument("--max-new-tokens", type=int, default=64, help="caption generation budget")
    data.add_argument("--batch-size", type=int, default=4, help="caption generation batch")
    data.add_argument("--pope-jsonl", help="POPE jsonl path")
    data.add_argument("--pope-limit", type=int, default=None)
    data.add_argument("--no-answer", dest="include_answer", action="store_false", help="POPE prompts without the label")
    data.add_argument("--text-source", choices=("wikitext", "path"), default="wikitext")
    data.add_argument("--text-path", help="local text/jsonl file for --text-source path")
    data.add_argument("--n-prompts", type=int, default=100, help="text corpus size")
    data.add_argument("--min-chars", type=int, default=600, help="minimum prompt length")
    data.add_argument("--text-split", default="train", help="wikitext split (train fits, validation holds out)")
    data.add_argument("--seed", type=int, default=0)

    fit = parser.add_argument_group("fit")
    fit.add_argument("--layers", default="all", help="'all' or comma-separated layer indices")
    fit.add_argument("--target-layer", type=int, default=None, help="gradient target (default: final layer)")
    fit.add_argument("--dim-batch", type=int, default=8, help="output dims per backward pass")
    fit.add_argument("--max-seq-len", type=int, default=1536)
    fit.add_argument(
        "--skip-first",
        type=int,
        default=None,
        help=f"leading positions excluded; default {TEXT_SKIP_FIRST} for --corpus text, else {MM_SKIP_FIRST}",
    )
    fit.add_argument("--masks", default="text,image,all")
    fit.add_argument(
        "--target-mask",
        default="all",
        help="cotangent target set: 'all' (upstream) or 'text' (X1: no placeholder targets)",
    )
    fit.add_argument("--shard", default="0/1", metavar="I/N")
    fit.add_argument("--limit", type=int, default=None, help="use only the first N samples")
    fit.add_argument("--checkpoint-every", type=int, default=1, help="0 = only at the end")
    fit.add_argument("--no-resume", action="store_true")
    fit.add_argument("--log-every", type=int, default=1)
    fit.add_argument(
        "--probe-every",
        type=int,
        default=None,
        metavar="N",
        help="online probe: score the running lens every N used samples (needs --probe-manifest)",
    )
    fit.add_argument(
        "--probe-manifest",
        default=None,
        help="online probe: held-out manifest the running lens is scored on",
    )
    fit.add_argument(
        "--probe-n", type=int, default=16, help="online probe: first N probe samples to score"
    )
    fit.add_argument("--probe-tag", default="text", help="online probe: position tag to score")
    fit.add_argument(
        "--allow-tf32",
        action="store_true",
        help="enable TF32 matmul/cuDNN for fp32 fits (much faster on Hopper; changes numerics)",
    )

    out = parser.add_argument_group("output")
    out.add_argument("--save-dtype", choices=("float16", "float32"), default="float16")
    out.add_argument("--notes", default=None, help="free-text note stored in provenance.json")
    return parser.parse_args(argv)


def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device:
        return torch.device(args.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(args: argparse.Namespace, device: torch.device) -> torch.dtype:
    if args.dtype:
        return DTYPES[args.dtype]
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def build_model(args: argparse.Namespace) -> LlavaLensModel:
    if args.backend == "tiny":
        hf_model, processor = build_tiny_llava(TinyLlavaConfig(image_size=args.tiny_image_size))
        return LlavaLensModel(hf_model, processor)
    device = resolve_device(args)
    dtype = resolve_dtype(args, device)
    print(f"loading {args.model} (device={device}, dtype={dtype})")
    return LlavaLensModel.from_pretrained(
        args.model,
        dtype=dtype,
        device=device,
        attn_implementation=args.attn_implementation,
        compile=args.compile,
        local_files_only=args.local_files_only,
    )


def build_corpus(args: argparse.Namespace, model: LlavaLensModel, out_dir: Path) -> Path:
    if args.corpus == "dummy":
        image_size = args.tiny_image_size if args.backend == "tiny" else 336
        return build_dummy_manifest(
            out_dir / "dummy-corpus",
            n_samples=args.n_samples,
            image_size=image_size,
            seed=args.seed,
        )
    if args.images_dir is None:
        raise SystemExit(f"--corpus {args.corpus} needs --images-dir")
    if args.corpus == "coco":
        return build_coco_caption_manifest(
            out_dir / "manifest-coco.jsonl",
            images_dir=args.images_dir,
            model=model,
            n_images=args.n_images,
            questions_per_image=args.questions_per_image,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            seed=args.seed,
            prompt_mode=args.prompt_mode,
            max_seq_len=args.max_seq_len,
        )
    if args.corpus == "pope":
        if args.pope_jsonl is None:
            raise SystemExit("--corpus pope needs --pope-jsonl")
        return build_pope_manifest(
            out_dir / "manifest-pope.jsonl",
            pope_jsonl=args.pope_jsonl,
            images_dir=args.images_dir,
            include_answer=args.include_answer,
            limit=args.pope_limit,
        )
    if args.corpus == "text":
        return build_text_manifest(
            out_dir / "manifest-text.jsonl",
            n_prompts=args.n_prompts,
            source=args.text_source,
            path=args.text_path,
            min_chars=args.min_chars,
            split=args.text_split,
            max_tokens=args.max_seq_len,
            tokenizer=model.tokenizer,
        )
    raise SystemExit(f"unknown corpus {args.corpus!r}")


def parse_layers(resolved: str, n_layers: int) -> list[int] | None:
    if resolved.strip().lower() == "all":
        return None
    try:
        layers = [int(part) for part in resolved.split(",") if part.strip()]
    except ValueError as error:
        raise SystemExit(f"--layers must be 'all' or comma-separated ints: {error}") from error
    if not layers:
        raise SystemExit("--layers resolved to an empty list")
    out_of_range = [layer for layer in layers if layer >= n_layers]
    if out_of_range:
        raise SystemExit(f"--layers {out_of_range} out of range for {n_layers} layers")
    return layers


def parse_shard(value: str) -> tuple[int, int]:
    try:
        index_text, count_text = value.split("/")
        index, count = int(index_text), int(count_text)
    except ValueError as error:
        raise SystemExit(f"--shard must look like 0/2: {error}") from error
    if not 0 <= index < count:
        raise SystemExit(f"invalid shard {value!r}")
    return index, count


def run_merge(args: argparse.Namespace) -> int:
    shard_dirs = [Path(path) for path in args.merge or []]
    # Accept either the run directory or its artifacts/ subdirectory.
    shard_dirs = [
        path / "artifacts" if (path / "artifacts").is_dir() else path for path in shard_dirs
    ]
    missing = [str(path) for path in shard_dirs if not path.exists()]
    if missing:
        raise SystemExit(f"shard directories do not exist: {missing}")
    lenses = merge_shards(shard_dirs)
    _, provenance = load_lens_set(shard_dirs[0])
    provenance = {
        **provenance,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "merged_from": [str(path) for path in shard_dirs],
        "n_prompts": {mask: int(lens.n_prompts) for mask, lens in lenses.items()},
    }
    written = save_lens_set(Path(args.out) / "artifacts", lenses, provenance=provenance)
    print(f"merged {len(shard_dirs)} shards -> {args.out}")
    for mask, path in sorted(written.items()):
        print(f"  lens-{mask}.pt  n_prompts={lenses[mask].n_prompts}  {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge:
        return run_merge(args)
    if args.manifest and args.corpus:
        raise SystemExit("pass either --manifest or --corpus, not both")
    if not args.manifest and not args.corpus:
        raise SystemExit("pass --manifest or --corpus")

    started = time.perf_counter()
    model = build_model(args)
    precision = configure_tf32(enabled=args.allow_tf32)
    print(
        f"model ready: n_layers={model.n_layers} d_model={model.d_model} "
        f"image_seq_length={model.image_seq_length} tf32={precision['allow_tf32']}"
    )

    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        manifest_path = build_corpus(args, model, out_dir)
        print(f"built manifest: {manifest_path}")

    samples = read_manifest(manifest_path)
    corpus_meta = manifest_meta(manifest_path)
    skip_first = args.skip_first
    if skip_first is None:
        skip_first = TEXT_SKIP_FIRST if args.corpus == "text" else MM_SKIP_FIRST
    masks = [part.strip() for part in args.masks.split(",") if part.strip()]
    shard = parse_shard(args.shard)
    source_layers = parse_layers(args.layers, model.n_layers)

    print(
        f"fitting {len(samples)} samples: backend={args.backend} layers={source_layers or 'all below target'} "
        f"target={args.target_layer if args.target_layer is not None else model.n_layers - 1} "
        f"dim_batch={args.dim_batch} skip_first={skip_first} masks={masks} "
        f"target_mask={args.target_mask} shard={shard}"
    )
    result = fit_masked(
        model,
        samples,
        source_layers=source_layers,
        target_layer=args.target_layer,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        skip_first=skip_first,
        masks=masks,
        target_mask=args.target_mask,
        checkpoint_path=out_dir / "checkpoint.pt",
        checkpoint_every=args.checkpoint_every or None,
        resume=not args.no_resume,
        limit=args.limit,
        shard=shard,
        log_every=args.log_every,
        probe_every=args.probe_every,
        probe_manifest=args.probe_manifest,
        probe_n=args.probe_n,
        probe_tag=args.probe_tag,
    )

    provenance = build_provenance(
        model=model,
        tasks=[args.corpus or "manifest"],
        masks=sorted(result.lenses),
        n_prompts=result.n_prompts,
        fit_config=result.config,
        corpus=corpus_meta,
        manifest_path=manifest_path,
        notes=args.notes,
        extra={
            "backend": args.backend,
            "model_id": args.model if args.backend == "hf-llava" else "tiny-llava",
            "allow_tf32": precision["allow_tf32"],
            "wall_seconds": round(time.perf_counter() - started, 1),
            **drop_stats(int(result.config["n_samples"]), result.skipped),
            "prompt_template_sha256": corpus_meta.get("prompt_template_sha256"),
        },
    )

    written = save_lens_set(
        out_dir / "artifacts",
        result.lenses,
        provenance=provenance,
        dtype=DTYPES[args.save_dtype],
    )

    print(f"\nfitted {len(result.lenses)} lens(es) from {len(samples)} samples in "
          f"{time.perf_counter() - started:.1f}s; skipped={len(result.skipped)}")
    for mask, lens in sorted(result.lenses.items()):
        print(f"  {mask:<5} n_prompts={lens.n_prompts:<5} layers={lens.source_layers} -> {written[mask]}")
    if result.skipped:
        for entry in result.skipped[:5]:
            print(f"  skipped: {entry}")
    summary = convergence_summary(result.history) if result.history else {}
    if summary:
        print("convergence (max relative change over the last 10 samples):")
        for mask, value in sorted(summary.items()):
            print(f"  {mask:<5} {value:.2e}")
    print(f"provenance: {out_dir / 'artifacts' / 'provenance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
