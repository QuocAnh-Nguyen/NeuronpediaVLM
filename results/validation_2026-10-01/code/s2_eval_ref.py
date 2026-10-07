#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Step 3 evaluation: held-out fidelity for the caption lenses, plus the bias/transfer grid.

Three analyses in one process (one model load):

* **main** — every tag (``text``/``image``/``all``/``image-q0..q3``) on the held-out caption
  manifest, reported twice: with the scorer's placeholder exclusion (V5: the ``image`` tag
  collapses to one position per sample) and with ``include_placeholders=True`` (whole-block
  model matching, descriptive only per V4). Mask composition is recorded per tag.
* **halves** — A4/X8 instruction-pool split: lens A (fitted on question half A) and lens B
  scored on held-out questions of both halves, giving same-half vs cross-half cells.
* **transfer** — A4/E6 cross-corpus: the caption lens on held-out WikiText and the text
  lens on held-out captions.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from vlm_lens._batch import as_batch  # noqa: E402
from vlm_lens.artifacts import load_lens_set  # noqa: E402
from vlm_lens.data.manifest import read_manifest  # noqa: E402
from vlm_lens.evaluate import format_scores, score_lens  # noqa: E402
from vlm_lens.models.llava import LlavaLensModel  # noqa: E402
from vlm_lens.positions import build_position_masks  # noqa: E402

TAGS = ("text", "image", "all", "image-q0", "image-q1", "image-q2", "image-q3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-lens-dir", required=True, help="merged 100-sample caption lens")
    parser.add_argument("--heldout-manifest", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--half-lens-a", help="lens fitted on question half A (unmerged)")
    parser.add_argument("--half-lens-b", help="lens fitted on question half B (unmerged)")
    parser.add_argument("--split-json", help="corpus-split.json (provides half_a_questions)")
    parser.add_argument("--text-lens-dir", help="S1 text lens, for the transfer rows")
    parser.add_argument("--text-heldout", help="held-out WikiText manifest")
    parser.add_argument("--mask", default="text")
    parser.add_argument("--tags", default=",".join(TAGS))
    parser.add_argument("--skip-first", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--composition-samples", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="score only the first N samples (smoke)")
    parser.add_argument(
        "--backend", choices=("hf-llava", "tiny"), default="hf-llava",
        help="model backend: 'hf-llava' (default, CUDA) or the tiny CPU smoke fixture",
    )
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> LlavaLensModel:
    """The scoring model: the HF checkpoint (default, CUDA) or the tiny CPU fixture."""
    if args.backend == "tiny":
        from vlm_lens.models.tiny_llava import TinyLlavaConfig, build_tiny_llava

        hf_model, processor = build_tiny_llava(TinyLlavaConfig())
        return LlavaLensModel(hf_model, processor)
    return LlavaLensModel.from_pretrained(
        dtype=torch.bfloat16, device="cuda", local_files_only=True
    )


def mask_composition(model, samples, tags, skip_first: int, max_seq_len: int) -> dict[str, dict]:
    """Average mask counts per tag over a few samples (V3: always report composition)."""
    totals: dict[str, list[int]] = defaultdict(list)
    seq_lengths: list[int] = []
    for sample in samples:
        batch = as_batch(model, sample, max_seq_len)
        masks = build_position_masks(
            batch.input_ids, model.image_token_id, skip_first=skip_first, masks=tags
        )
        seq_lengths.append(batch.seq_len)
        for tag, mask in masks.items():
            totals[tag].append(int(mask.sum()))
    return {
        tag: {
            "mean_positions": round(sum(values) / len(values), 1),
            "min": min(values),
            "max": max(values),
        }
        for tag, values in totals.items()
    } | {"seq_len": {"mean": round(sum(seq_lengths) / len(seq_lengths), 1)}}


def score_table(model, lens_dir: str, samples, *, mask: str, tags, skip_first: int, max_seq_len: int):
    lenses, provenance = load_lens_set(lens_dir)
    lens = lenses[mask]
    rows = {
        "default": score_lens(
            model, lens, samples, tags=tags, skip_first=skip_first, max_seq_len=max_seq_len
        ),
        "include_placeholders": score_lens(
            model,
            lens,
            samples,
            tags=tags,
            skip_first=skip_first,
            max_seq_len=max_seq_len,
            include_placeholders=True,
        ),
    }
    return lens, provenance, rows


def main() -> int:
    args = parse_args()
    tags = tuple(part.strip() for part in args.tags.split(",") if part.strip())
    heldout = read_manifest(args.heldout_manifest)
    if args.limit:
        heldout = heldout[: args.limit]
    model = load_model(args)
    report: dict[str, object] = {
        "heldout_manifest": args.heldout_manifest,
        "n_heldout_samples": len(heldout),
        "skip_first": args.skip_first,
        "tags": list(tags),
        "mask_composition": mask_composition(
            model, heldout[: args.composition_samples], tags, args.skip_first, args.max_seq_len
        ),
    }

    print("== main: merged caption lens, held-out captions ==")
    lens, provenance, rows = score_table(
        model, args.main_lens_dir, heldout, mask=args.mask, tags=tags,
        skip_first=args.skip_first, max_seq_len=args.max_seq_len,
    )
    report["main"] = {
        "lens_dir": args.main_lens_dir,
        "n_prompts": int(lens.n_prompts),
        "layers": sorted(lens.source_layers),
        "rows": {mode: [row.to_json() for row in score_rows] for mode, score_rows in rows.items()},
    }
    print(format_scores(rows["default"]))
    print("\n-- with include_placeholders (whole-block model matching) --")
    print(format_scores(rows["include_placeholders"]))

    if args.half_lens_a and args.half_lens_b:
        half_a = {
            question.strip()
            for question in json.loads(Path(args.split_json).read_text(encoding="utf-8"))[
                "half_a_questions"
            ]
            if question.strip()
        }
        if not half_a:
            raise SystemExit("--split-json with half_a_questions is required with --half-lens-a/-b")
        subsets = {
            "half_a": [sample for sample in heldout if sample.meta.get("question") in half_a],
            "half_b": [sample for sample in heldout if sample.meta.get("question") not in half_a],
        }
        halves: dict[str, object] = {
            "half_a_questions": sorted(half_a),
            "n_samples": {name: len(subset) for name, subset in subsets.items()},
        }
        for label, lens_dir in (("lens_a", args.half_lens_a), ("lens_b", args.half_lens_b)):
            for subset_name, subset in subsets.items():
                half_lens, _, half_rows = score_table(
                    model, lens_dir, subset, mask=args.mask, tags=("text",),
                    skip_first=args.skip_first, max_seq_len=args.max_seq_len,
                )
                key = f"{label}_on_{subset_name}"
                halves[key] = {
                    "n_prompts": int(half_lens.n_prompts),
                    "rows": [row.to_json() for row in half_rows["default"]],
                }
        report["halves"] = halves
        print("\n== A4/X8 instruction halves (text tag, held-out captions) ==")
        for key, entry in halves.items():
            if not isinstance(entry, dict) or "rows" not in entry:
                continue
            near = sorted(entry["rows"], key=lambda row: row["layer"])[-4:]
            print(
                f"{key:<18} n_prompts={entry['n_prompts']:<4} "
                f"top rank={[round(r['mean_rank_true'], 1) for r in near][-1]} "
                f"kl={[round(r['mean_kl'], 3) for r in near][-1]}"
            )

    if args.text_lens_dir and args.text_heldout:
        text_heldout = read_manifest(args.text_heldout)
        transfer: dict[str, object] = {"n_text_heldout": len(text_heldout)}
        caption_lens, _, caption_rows = score_table(
            model, args.main_lens_dir, text_heldout, mask=args.mask, tags=("text",),
            skip_first=16, max_seq_len=args.max_seq_len,
        )
        transfer["caption_lens_on_wikitext"] = {
            "skip_first": 16,
            "rows": [row.to_json() for row in caption_rows["default"]],
        }
        text_lens, _, text_rows = score_table(
            model, args.text_lens_dir, heldout, mask="all", tags=("text",),
            skip_first=1, max_seq_len=args.max_seq_len,
        )
        transfer["text_lens_on_captions"] = {
            "skip_first": 1,
            "rows": [row.to_json() for row in text_rows["default"]],
        }
        report["transfer"] = transfer
        print("\n== A4/E6 cross-corpus transfer ==")
        for key in ("caption_lens_on_wikitext", "text_lens_on_captions"):
            entry = transfer[key]
            last = entry["rows"][-1]
            print(
                f"{key:<26} L{last['layer']} rank={last['mean_rank_true']:.1f} "
                f"(model {last['model_mean_rank_true']:.1f}) kl={last['mean_kl']:.3f}"
            )

    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
