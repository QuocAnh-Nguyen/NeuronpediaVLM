#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Step 4: X7 (edit-conditioning) and X9 (swap/edit sanity check on real generations).

X7 — for each candidate token pair, measure how well-conditioned the two J-lens vector
directions are at each layer: ``cond([v_s, v_t])`` (sigma1/sigma2 of the 2-column basis the
``swap`` edit projects onto), their cosine, and the norm ratio. Pairs above the conditioning
threshold are excluded from the swap experiment and recorded.

X9 — on a handful of real held-out captions, generate greedily with no edit (baseline) and
with each edit active at the chosen layer(s): ``add`` toward the target, ``ablate`` the
source, ``swap`` source->target. Report the generated text, whether it differs from
baseline, and the first differing token.

Add strengths are expressed in residual units: ``alpha = k * ||h_layer|| / ||v_t||`` so the
injected vector has norm ``k * ||h_layer||``; norms come from the X3 census JSON when
available (``--norms-json``), else ``--fallback-norm`` is used.
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

import vlm_lens  # noqa: E402, F401  # installs the vendored jlens path: must precede jlens
from vlm_lens.artifacts import load_lens_set  # noqa: E402
from vlm_lens.data.manifest import read_manifest  # noqa: E402
from vlm_lens.interventions import ResidualEdit, generate_with_edits, lens_vectors  # noqa: E402
from vlm_lens.models.llava import LlavaLensModel  # noqa: E402

# Token strings keep the caption's leading space: the lens vector is for the token the
# model would actually emit at that position.
PAIRS = (
    (" dog", " cat"),
    (" car", " bus"),
    (" man", " woman"),
    (" horse", " cow"),
    (" pizza", " sandwich"),
    (" boat", " airplane"),
    (" chair", " couch"),
    (" tree", " plant"),
    (" bicycle", " motorcycle"),
    (" television", " laptop"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lens-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--layers", default="8,16,24", help="layers for the conditioning census")
    parser.add_argument("--edit-layers", default="16,24")
    parser.add_argument("--cond-max", type=float, default=1e3, help="skip swaps above this cond")
    parser.add_argument("--norms-json", help="X3 census output (mean residual norms per layer)")
    parser.add_argument("--norm-group", default="text_post", help="X3 group used for ||h_layer||")
    parser.add_argument("--fallback-norm", type=float, default=150.0)
    parser.add_argument("--alphas-add", default="0.05,0.1,0.2", help="fractions of ||h_layer||")
    parser.add_argument("--alphas-ablate", default="0.5,1.0")
    parser.add_argument("--alphas-swap", default="0.5,1.0")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--json", required=True)
    return parser.parse_args()


def conditioning(model, lens, pairs, layers) -> dict[str, dict]:
    tokens = [token for pair in pairs for token in pair]
    out: dict[str, dict] = {}
    for layer in layers:
        rows = lens_vectors(model, lens, tokens, layers=[layer])[layer]  # [len(tokens), d]
        per_pair: dict[str, dict] = {}
        for index, (source, target) in enumerate(pairs):
            v_s, v_t = rows[2 * index], rows[2 * index + 1]
            basis = torch.stack([v_s, v_t], dim=1)  # [d, 2]
            singular = torch.linalg.svdvals(basis)
            cond = float(singular[0] / singular[1].clamp_min(1e-12))
            cosine = float(
                torch.nn.functional.cosine_similarity(v_s, v_t, dim=0)
            )
            per_pair[f"{source.strip()}->{target.strip()}"] = {
                "cond": cond,
                "cosine": cosine,
                "norm_source": float(v_s.norm()),
                "norm_target": float(v_t.norm()),
                "degenerate": cond > 1e3,
            }
        out[str(layer)] = per_pair
    return out


def load_norms(path: str | None, group: str = "text_post") -> dict[int, float]:
    """Mean residual norms per layer from the X3 census.

    The census stores ``layers[layer][group]["mean_norm"]`` for groups ``bos``,
    ``pos_1_16``, ``text_pre``, ``patch_1_16``, ``patch_17_end`` and ``text_post``. Edits act
    on text positions during generation, so ``text_post`` is the default scale (falling back
    to any available group if the requested one is missing).
    """
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    norms: dict[int, float] = {}
    for layer, groups in data.get("layers", {}).items():
        if not isinstance(groups, dict):
            continue
        entry = groups.get(group)
        if not isinstance(entry, dict):
            entry = next(
                (
                    value
                    for value in groups.values()
                    if isinstance(value, dict) and "mean_norm" in value
                ),
                None,
            )
        if isinstance(entry, dict) and "mean_norm" in entry:
            norms[int(layer)] = float(entry["mean_norm"])
    return norms


def main() -> int:
    args = parse_args()
    layers = [int(part) for part in args.layers.split(",")]
    edit_layers = [int(part) for part in args.edit_layers.split(",")]
    alphas_add = [float(part) for part in args.alphas_add.split(",")]
    alphas_ablate = [float(part) for part in args.alphas_ablate.split(",")]
    alphas_swap = [float(part) for part in args.alphas_swap.split(",")]

    lenses, provenance = load_lens_set(args.lens_dir)
    lens = lenses["text"]
    model = LlavaLensModel.from_pretrained(
        dtype=torch.bfloat16, device="cuda", local_files_only=True
    )
    norms = load_norms(args.norms_json, args.norm_group)
    samples = read_manifest(args.manifest)[: args.n_samples]

    report: dict[str, object] = {
        "lens_dir": args.lens_dir,
        "n_prompts": int(lens.n_prompts),
        "n_samples": len(samples),
        "edit_layers": edit_layers,
        "residual_norms": {str(layer): norms.get(layer, None) for layer in edit_layers},
        "fallback_norm": args.fallback_norm,
    }

    condition = conditioning(model, lens, PAIRS, layers)
    report["x7_conditioning"] = condition
    print("== X7 conditioning (cond=sigma1/sigma2 of [v_s, v_t]) ==")
    for layer, per_pair in condition.items():
        flagged = [name for name, entry in per_pair.items() if entry["degenerate"]]
        print(f"L{layer}: n_pairs={len(per_pair)} degenerate(cond>1e3)={flagged}")

    # ---- X9: generations -------------------------------------------------------------
    usable = [
        (source, target)
        for source, target in PAIRS
        if all(
            condition[str(layer)][f"{source.strip()}->{target.strip()}"]["cond"] <= args.cond_max
            for layer in edit_layers
        )
    ]
    report["x9_usable_pairs"] = [f"{s.strip()}->{t.strip()}" for s, t in usable]

    baselines: dict[str, dict] = {}
    for sample in samples:
        out = generate_with_edits(model, None, sample, max_new_tokens=args.max_new_tokens)
        baselines[sample.name] = {"text": out["text"], "token_ids": out["token_ids"]}
    report["x9_baselines"] = baselines

    results: list[dict] = []
    for sample in samples:
        baseline = baselines[sample.name]["token_ids"]
        for layer in edit_layers:
            norm = norms.get(layer, args.fallback_norm)
            for source, target in usable:
                vectors = lens_vectors(model, lens, [source, target], layers=[layer])[layer]
                v_t = vectors[1]
                specs: list[ResidualEdit] = []
                for alpha in alphas_add:
                    specs.append(
                        ResidualEdit(
                            layer=layer, mode="add", token=target,
                            alpha=float(alpha * norm / v_t.norm().clamp_min(1e-8)),
                        )
                    )
                for alpha in alphas_ablate:
                    specs.append(
                        ResidualEdit(layer=layer, mode="ablate", token=source, alpha=float(alpha))
                    )
                for alpha in alphas_swap:
                    specs.append(
                        ResidualEdit(
                            layer=layer, mode="swap", token=target, source_token=source,
                            alpha=float(alpha),
                        )
                    )
                for spec in specs:
                    out = generate_with_edits(
                        model, lens, sample, edits=[spec], max_new_tokens=args.max_new_tokens
                    )
                    tokens = out["token_ids"]
                    first_diff = next(
                        (
                            index
                            for index, (a, b) in enumerate(zip(baseline, tokens, strict=False))
                            if a != b
                        ),
                        None,
                    )
                    results.append(
                        {
                            "sample": sample.name,
                            "question": sample.meta.get("question"),
                            "baseline": baselines[sample.name]["text"],
                            "pair": f"{source.strip()}->{target.strip()}",
                            "edit": spec.to_json(),
                            "text": out["text"],
                            "changed": tokens != baseline,
                            "first_diff_token": first_diff,
                            "n_edit_forwards": out["n_edit_forwards"],
                        }
                    )
        print(f"  {sample.name}: baseline={baselines[sample.name]['text'][:60]!r}")

    report["x9_results"] = results
    changed = sum(1 for row in results if row["changed"])
    report["x9_summary"] = {
        "n_generations": len(results),
        "n_changed": changed,
        "change_rate": changed / max(len(results), 1),
    }
    print(f"\n== X9 summary: {changed}/{len(results)} generations differ from baseline ==")

    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
