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

Runtime note (D34). The reference implementation of this script is preserved verbatim in
``s2_eval_ref.py``. It produced identical logits (via ``lens_readout``) but computed every
metric on the CPU from the fp32-CPU logits and ran each phase twice (default +
include_placeholders): measured ~23 s/sample, ~41 cores busy, GPU ~22 % utilization, and the
first table still unprinted after 115 min (projected ~15.5 h total on the shared Brev box).
This file keeps the logits identical and changes only how the metrics are reduced:

* metrics (rank, top-1, KL) are computed with GPU reductions instead of CPU reductions;
* one ``lens_readout`` per sample is shared by both modes (the modes differ only in which
  *targets* are counted, never in the readout);
* the ``include_placeholders`` pass is skipped where the reference discarded it
  (halves/transfer store only the default rows).

Integer metrics (``n``, true-token rank, top-1 agreement) are exact reproductions; the
float means (KL) differ only in fp32 reduction order (``s2_eval_ab.py`` prints the max
deltas on real samples).
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
from vlm_lens.evaluate import LensScore, format_scores  # noqa: E402
from vlm_lens.models.llava import LlavaLensModel  # noqa: E402
from vlm_lens.positions import build_position_masks  # noqa: E402
from vlm_lens.readout import lens_readout  # noqa: E402

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


def _rank_of_row(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """1-based rank of each target id in its logit row (ties count as better ranks).

    Same definition as ``vlm_lens.evaluate._rank_of_row``; element-wise comparisons give
    identical integers on CPU and GPU for identical logits.
    """
    target_logit = logits.gather(1, targets[:, None]).squeeze(1)
    return (logits > target_logit[:, None]).sum(dim=1) + 1


def _empty_row() -> dict[str, float]:
    return {"n": 0.0, "rank": 0.0, "model_rank": 0.0, "agree": 0.0, "kl": 0.0}


def score_table_fast(
    model,
    lens_dir: str,
    samples,
    *,
    mask: str,
    tags,
    skip_first: int,
    max_seq_len: int,
    modes=("default",),
    use_jacobian: bool = True,
    chunk_size: int = 256,
) -> tuple[object, dict, dict[str, list[LensScore]]]:
    """GPU-metric scorer with the reference's exact position/target semantics.

    One ``lens_readout`` per sample covers every tag (union positions, as in
    ``evaluate._iter_scored_batches``); both requested modes derive their metrics from it.
    Per (layer, tag) the same per-chunk float32 sums are accumulated in the same order as
    the reference; only the reduction device differs.
    """
    lenses, provenance = load_lens_set(lens_dir)
    lens = lenses[mask]
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and model.unembed_weight().is_cuda) else "cpu"
    )
    final_layer = model.n_layers - 1
    score_layers = sorted(set(lens.source_layers) | {final_layer})
    stats = {
        mode: {layer: {tag: _empty_row() for tag in tags} for layer in score_layers}
        for mode in modes
    }

    for sample in samples:
        batch = as_batch(model, sample, max_seq_len)
        masks = build_position_masks(
            batch.input_ids, model.image_token_id, skip_first=skip_first, masks=tags
        )
        active = {tag: m for tag, m in masks.items() if bool(m.any())}
        if not active:
            continue
        index_list = sorted(
            {int(p) for m in active.values() for p in m.nonzero(as_tuple=True)[0]}
        )
        readout = lens_readout(
            model,
            lens,
            batch,
            layers=[layer for layer in score_layers if not (use_jacobian and layer == final_layer)],
            positions=index_list,
            use_jacobian=use_jacobian,
            max_seq_len=max_seq_len,
        )
        row_of = {position: row for row, position in enumerate(readout.positions)}
        input_ids = readout.input_ids
        seq_len = input_ids.numel()
        model_logits_cpu = readout.model_logits

        for mode in modes:
            include_placeholders = mode == "include_placeholders"
            for tag, m in active.items():
                positions = [
                    int(p) for p in m.nonzero(as_tuple=True)[0] if int(p) + 1 < seq_len
                ]
                if not include_placeholders:
                    positions = [
                        p for p in positions if int(input_ids[p + 1]) != model.image_token_id
                    ]
                if not positions:
                    continue
                rows = torch.tensor([row_of[p] for p in positions], dtype=torch.long)
                targets = input_ids[torch.tensor([p + 1 for p in positions], dtype=torch.long)]
                for start in range(0, rows.numel(), chunk_size):
                    chunk = rows[start : start + chunk_size]
                    chunk_targets = targets[start : start + chunk_size].to(device)
                    model_chunk = model_logits_cpu[chunk].to(device)
                    model_rank = _rank_of_row(model_chunk, chunk_targets).float().sum()
                    model_top1 = model_chunk.argmax(dim=1)
                    model_log_probs = torch.log_softmax(model_chunk, dim=-1)
                    n = int(chunk_targets.numel())
                    for layer in score_layers:
                        if use_jacobian and layer == final_layer:
                            lens_chunk = model_chunk
                        else:
                            lens_chunk = readout.lens_logits[layer][chunk].to(device)
                        accumulator = stats[mode][layer][tag]
                        accumulator["n"] += float(n)
                        accumulator["rank"] += float(
                            _rank_of_row(lens_chunk, chunk_targets).float().sum()
                        )
                        accumulator["model_rank"] += float(model_rank)
                        accumulator["agree"] += float(
                            (lens_chunk.argmax(dim=1) == model_top1).float().sum()
                        )
                        lens_log_probs = torch.log_softmax(lens_chunk, dim=-1)
                        accumulator["kl"] += float(
                            torch.nn.functional.kl_div(
                                lens_log_probs,
                                model_log_probs,
                                reduction="none",
                                log_target=True,
                            )
                            .sum(dim=-1)
                            .sum()
                        )

    out: dict[str, list[LensScore]] = {}
    for mode in modes:
        rows_out: list[LensScore] = []
        for layer in score_layers:
            for tag in tags:
                accumulator = stats[mode][layer][tag]
                n = int(accumulator["n"])
                if n == 0:
                    continue
                rows_out.append(
                    LensScore(
                        layer=layer,
                        tag=tag,
                        n=n,
                        mean_rank_true=accumulator["rank"] / n,
                        model_mean_rank_true=accumulator["model_rank"] / n,
                        top1_agreement=accumulator["agree"] / n,
                        mean_kl=accumulator["kl"] / n,
                    )
                )
        out[mode] = rows_out
    return lens, provenance, out


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
    lens, provenance, rows = score_table_fast(
        model, args.main_lens_dir, heldout, mask=args.mask, tags=tags,
        skip_first=args.skip_first, max_seq_len=args.max_seq_len,
        modes=("default", "include_placeholders"),
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
                half_lens, _, half_rows = score_table_fast(
                    model, lens_dir, subset, mask=args.mask, tags=("text",),
                    skip_first=args.skip_first, max_seq_len=args.max_seq_len,
                    modes=("default",),
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
        caption_lens, _, caption_rows = score_table_fast(
            model, args.main_lens_dir, text_heldout, mask=args.mask, tags=("text",),
            skip_first=16, max_seq_len=args.max_seq_len, modes=("default",),
        )
        transfer["caption_lens_on_wikitext"] = {
            "skip_first": 16,
            "rows": [row.to_json() for row in caption_rows["default"]],
        }
        text_lens, _, text_rows = score_table_fast(
            model, args.text_lens_dir, heldout, mask="all", tags=("text",),
            skip_first=1, max_seq_len=args.max_seq_len, modes=("default",),
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
