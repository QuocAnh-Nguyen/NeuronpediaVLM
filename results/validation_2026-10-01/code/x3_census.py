#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""X3: residual-norm census by modality/position at layers 0/16/31 over COCO samples.

Reports, per position group, the residual norm distribution (for the alpha scaling in
X7/X9) and the dominant massive-activation dimensions, and applies the register X3 rule:
if the leading positions look BOS-like (same order of magnitude, same dominant
dimensions), raise ``skip_first`` to the smallest value in {8,16,32} that removes them;
otherwise keep 1.

Groups are derived from the real ``input_ids`` (never hardcoded): ``bos``, ``text_pre``
(the "USER: <image>..." tokens before the block), ``patch_1_16`` (first 16 patch
positions), ``patch_17_end``, ``text_post`` (caption tokens), plus ``pos_1_16`` (the
positions the paper's text control drops).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402, I001  # the vlm_lens import must precede jlens: it installs the path
import vlm_lens  # noqa: E402, F401
from jlens.hooks import ActivationRecorder  # noqa: E402

from vlm_lens._batch import as_batch  # noqa: E402
from vlm_lens.data.manifest import read_manifest  # noqa: E402
from vlm_lens.models.llava import LlavaLensModel  # noqa: E402

LAYERS = (0, 16, 24, 31)  # 24 = X7/X9's upper edit layer: its row gives alpha measured units
SKIP_CANDIDATES = (8, 16, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True, help="JSON output path")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--top-dims", type=int, default=5)
    return parser.parse_args()


def position_groups(input_ids: torch.Tensor, image_token_id: int) -> dict[str, list[int]]:
    """Position groups for one sample, derived from its token ids."""
    ids = input_ids.reshape(-1)
    image_positions = [int(p) for p in (ids == image_token_id).nonzero(as_tuple=True)[0]]
    first_patch = image_positions[0]
    last_patch = image_positions[-1]
    text_positions = [p for p in range(ids.numel()) if p not in set(image_positions)]
    pre = [p for p in text_positions if p < first_patch]
    post = [p for p in text_positions if p > last_patch]
    return {
        "bos": [0],
        "pos_1_16": list(range(1, min(17, ids.numel()))),
        "text_pre": [p for p in pre if p > 0],
        "patch_1_16": image_positions[:16],
        "patch_17_end": image_positions[16:],
        "text_post": post,
    }


def main() -> int:
    args = parse_args()
    samples = read_manifest(args.manifest)[: args.n_samples]
    model = LlavaLensModel.from_pretrained(dtype=torch.bfloat16, device="cuda", local_files_only=True)
    print(f"model ready: layers={model.n_layers} image_seq_length={model.image_seq_length}")

    group_norms: dict[int, dict[str, list[float]]] = {layer: {} for layer in LAYERS}
    group_max: dict[int, dict[str, float]] = {layer: {} for layer in LAYERS}
    top_dim_counts: dict[int, dict[str, Counter]] = {
        layer: {} for layer in LAYERS
    }
    per_sample: list[dict[str, object]] = []

    with torch.no_grad():
        for index, sample in enumerate(samples):
            batch = as_batch(model, sample, args.max_seq_len)
            groups = position_groups(batch.input_ids, model.image_token_id)
            with ActivationRecorder(model.layers, at=list(LAYERS)) as recorder:
                model.forward_mm(batch)
            record: dict[str, object] = {
                "sample_id": sample.sample_id,
                "seq_len": batch.seq_len,
                "n_image_tokens": batch.n_image_tokens,
            }
            for layer in LAYERS:
                activation = recorder.activations[layer].float().reshape(-1, model.d_model)
                norms = activation.norm(dim=-1)
                record[f"layer{layer}"] = {}
                for name, positions in groups.items():
                    if not positions:
                        continue
                    index_tensor = torch.tensor(positions, dtype=torch.long, device=activation.device)
                    values = norms[index_tensor]
                    group_norms[layer].setdefault(name, []).append(float(values.mean()))
                    group_max[layer][name] = max(
                        group_max[layer].get(name, 0.0), float(values.max())
                    )
                    counter = top_dim_counts[layer].setdefault(name, Counter())
                    peak = int(values.argmax())
                    row = activation[index_tensor[peak]]
                    for dim in torch.topk(row.abs(), k=args.top_dims).indices.tolist():
                        counter[int(dim)] += 1
                    record[f"layer{layer}"][name] = {  # type: ignore[index]
                        "mean_norm": round(float(values.mean()), 3),
                        "max_norm": round(float(values.max()), 3),
                        "peak_top_dims": [
                            [int(dim), round(float(row[dim]), 2)]
                            for dim in torch.topk(row.abs(), k=args.top_dims).indices.tolist()
                        ],
                    }
            per_sample.append(record)
            if (index + 1) % 10 == 0:
                print(f"  {index + 1}/{len(samples)} samples", flush=True)

    summary: dict[str, object] = {"n_samples": len(samples), "layers": {}}
    for layer in LAYERS:
        layer_summary: dict[str, object] = {}
        for name in group_norms[layer]:
            means = group_norms[layer][name]
            layer_summary[name] = {
                "mean_norm": round(sum(means) / len(means), 3),
                "max_norm_across_samples": round(group_max[layer][name], 3),
                "top_dims": [
                    [dim, count] for dim, count in top_dim_counts[layer][name].most_common(args.top_dims)
                ],
            }
        summary["layers"][layer] = layer_summary

    # X3 decision support: is the leading block BOS-like?
    verdict: dict[str, object] = {}
    for layer in LAYERS:
        stats = summary["layers"][layer]  # type: ignore[index]
        bos = stats.get("bos", {}).get("mean_norm")
        lead = stats.get("pos_1_16", {}).get("mean_norm")
        patch = stats.get("patch_1_16", {}).get("mean_norm")
        bos_dims = [dim for dim, _ in stats.get("bos", {}).get("top_dims", [])]
        lead_dims = [dim for dim, _ in stats.get("pos_1_16", {}).get("top_dims", [])]
        overlap = len(set(bos_dims) & set(lead_dims)) / max(1, len(set(bos_dims) | set(lead_dims)))
        verdict[layer] = {
            "ratio_pos_1_16_over_bos": round(lead / bos, 3) if bos and lead else None,
            "ratio_patch_1_16_over_bos": round(patch / bos, 3) if bos and patch else None,
            "dominant_dim_jaccard_vs_bos": round(overlap, 3),
        }
    sink_like = all(
        (entry["ratio_pos_1_16_over_bos"] or 0) <= 10.0 for entry in verdict.values()
    ) and max(entry["dominant_dim_jaccard_vs_bos"] for entry in verdict.values()) >= 0.5
    summary["verdict"] = {
        "layers": verdict,
        "pos_1_16_sink_like": sink_like,
        "recommended_skip_first": (
            min(value for value in SKIP_CANDIDATES if value >= 16) if sink_like else 1
        ),
        "rule": "register X3: sink-like leading positions -> smallest skip_first in {8,16,32} covering them",
    }
    summary["per_sample"] = per_sample
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nlayer  group         mean_norm  max_norm  top_dims")
    for layer in LAYERS:
        for name, stats in summary["layers"][layer].items():  # type: ignore[index]
            print(
                f"{layer:<6} {name:<13} {stats['mean_norm']:>9.2f} {stats['max_norm_across_samples']:>9.2f}  "
                + ", ".join(f"d{dim}:{count}" for dim, count in stats["top_dims"])
            )
    print(f"\nverdict: {json.dumps(summary['verdict'])}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
