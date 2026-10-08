# SPDX-License-Identifier: Apache-2.0
"""X9b: alpha-sweep re-run of X9's edits with a wider strength grid (D36/D39 follow-up).

D36/D39: at X9's surgical add strengths (0.05-0.2 of ``||h_layer||``) 85 % of greedy
generations were unchanged, and the 7 concept-directed cases that did appear required
alpha ~= 5-14x the residual norm. This script re-runs the same ``add``/``ablate``/``swap``
edits on real held-out captions with a wider grid — add strengths as multiples ``k`` of
the residual norm (``alpha = k * ||h_layer|| / ||v_t||``, so the injected vector has norm
``k * ||h_layer||``; the grid spans the surgical 1-3x band through the 5-14x band where
the directed cases appeared) and wider ablate/swap alphas — to find the minimal effective
strength: the smallest strength whose change rate reaches the bar (0.2).

Per sample x layer x pair x mode x strength: build the :class:`ResidualEdit` (add: the
``k`` scaling above with the same ``clamp_min`` guard; ablate/swap: alpha as given),
generate greedily with ``generate_with_edits``, and record ``changed`` (token ids differ
from the unedited baseline), ``first_diff_token``, and the D39 directionality check — the
pair's target token appears in the edited text but not the baseline (add/swap), or the
source token disappears (ablate/swap) — via word-boundary regex on lowercased text. For
``swap`` either sign counts; both sub-checks are recorded per row.
"""

from __future__ import annotations

import argparse
import json
import re
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
# model would actually emit at that position; the directionality regex strips the space
# and matches on lowercased text.
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

#: The D39 minimal-effectiveness bar: a strength counts as effective once at least this
#: fraction of greedy generations differs from the unedited baseline.
MIN_CHANGE_RATE = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lens-dir", required=True)
    parser.add_argument("--manifest", required=True, help="held-out caption manifest")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument(
        "--norms-json", required=True, help="X3 census output (mean residual norms per layer)"
    )
    parser.add_argument("--json", required=True)
    parser.add_argument("--layers", default="16,24", help="edit layers")
    parser.add_argument(
        "--add-ks", default="1,2,4,8,16,32", help="add strengths as multiples k of ||h_layer||"
    )
    parser.add_argument("--ablate-alphas", default="0.5,1,2,4")
    parser.add_argument("--swap-alphas", default="0.5,1,2,4")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument(
        "--backend", choices=("hf-llava", "tiny"), default="hf-llava",
        help="model backend: 'hf-llava' (default, CUDA) or the tiny CPU smoke fixture",
    )
    parser.add_argument("--norm-group", default="text_post", help="X3 group used for ||h_layer||")
    parser.add_argument("--fallback-norm", type=float, default=150.0)
    return parser.parse_args()


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


def _word_present(word: str, text: str) -> bool:
    """Word-boundary match on lowercased text (tokens carry a leading space that is stripped)."""
    return re.search(rf"\b{re.escape(word.strip())}\b", text.lower()) is not None


def _directed(mode: str, source: str, target: str, baseline: str, text: str) -> dict[str, bool]:
    """D39 directionality check: a concept-specific sign of success, not just any change.

    ``add``: the target token appears in the edited text but not the baseline;
    ``ablate``: the source token appears in the baseline but not the edited text;
    ``swap``: either sign counts. All matches are word-boundary regexes on lowercased text.
    """
    target_gained = _word_present(target, text) and not _word_present(target, baseline)
    source_lost = _word_present(source, baseline) and not _word_present(source, text)
    if mode == "add":
        directed = target_gained
    elif mode == "ablate":
        directed = source_lost
    else:  # swap
        directed = target_gained or source_lost
    return {"directed": directed, "target_gained": target_gained, "source_lost": source_lost}


def main() -> int:
    args = parse_args()
    layers = [int(part) for part in args.layers.split(",")]
    add_ks = [float(part) for part in args.add_ks.split(",")]
    ablate_alphas = [float(part) for part in args.ablate_alphas.split(",")]
    swap_alphas = [float(part) for part in args.swap_alphas.split(",")]

    lenses, _ = load_lens_set(args.lens_dir)
    lens = lenses["text"]
    if args.backend == "tiny":
        from vlm_lens.models.tiny_llava import TinyLlavaConfig, build_tiny_llava

        hf_model, processor = build_tiny_llava(TinyLlavaConfig())
        model = LlavaLensModel(hf_model, processor)
    else:
        model = LlavaLensModel.from_pretrained(
            dtype=torch.bfloat16, device="cuda", local_files_only=True
        )
    norms = load_norms(args.norms_json, args.norm_group)
    samples = read_manifest(args.manifest)[: args.n_samples]

    report: dict[str, object] = {
        "experiment": "x9b_alpha_sweep",
        "lens_dir": args.lens_dir,
        "backend": args.backend,
        "n_prompts": int(lens.n_prompts),
        "n_samples": len(samples),
        "n_pairs": len(PAIRS),
        "layers": layers,
        "add_ks": add_ks,
        "ablate_alphas": ablate_alphas,
        "swap_alphas": swap_alphas,
        "max_new_tokens": args.max_new_tokens,
        "residual_norms": {str(layer): norms.get(layer, None) for layer in layers},
        "fallback_norm": args.fallback_norm,
        "min_change_rate": MIN_CHANGE_RATE,
    }

    # ---- baselines -------------------------------------------------------------------
    baselines: dict[str, dict] = {}
    for sample in samples:
        out = generate_with_edits(model, None, sample, max_new_tokens=args.max_new_tokens)
        baselines[sample.sample_id] = {"text": out["text"], "token_ids": out["token_ids"]}
    report["baselines"] = baselines

    # ---- the sweep -------------------------------------------------------------------
    # ||v_t|| per (layer, pair): the add alpha scales the injected vector to
    # k * ||h_layer||, and ||v_t|| varies per pair and layer but not per sample, so the
    # lens vectors are resolved once here rather than inside the sample loop.
    v_norms: dict[str, dict[str, float]] = {}
    for layer in layers:
        tokens = [token for pair in PAIRS for token in pair]
        rows = lens_vectors(model, lens, tokens, layers=[layer])[layer]
        for index, (source, target) in enumerate(PAIRS):
            v_norms.setdefault(str(layer), {})[f"{source.strip()}->{target.strip()}"] = float(
                rows[2 * index + 1].norm()
            )

    results: list[dict] = []
    for sample in samples:
        baseline_text = baselines[sample.sample_id]["text"]
        baseline = baselines[sample.sample_id]["token_ids"]
        for layer in layers:
            norm = norms.get(layer, args.fallback_norm)
            for source, target in PAIRS:
                v_norm = v_norms[str(layer)][f"{source.strip()}->{target.strip()}"]
                specs: list[tuple[float, ResidualEdit]] = []
                for k in add_ks:
                    specs.append(
                        (
                            k,
                            ResidualEdit(
                                layer=layer,
                                mode="add",
                                token=target,
                                alpha=float(k * norm / max(v_norm, 1e-8)),
                            ),
                        )
                    )
                for alpha in ablate_alphas:
                    specs.append(
                        (
                            alpha,
                            ResidualEdit(
                                layer=layer, mode="ablate", token=source, alpha=float(alpha)
                            ),
                        )
                    )
                for alpha in swap_alphas:
                    specs.append(
                        (
                            alpha,
                            ResidualEdit(
                                layer=layer,
                                mode="swap",
                                token=target,
                                source_token=source,
                                alpha=float(alpha),
                            ),
                        )
                    )
                for strength, spec in specs:
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
                            "sample": sample.sample_id,
                            "question": sample.meta.get("question"),
                            "baseline": baseline_text,
                            "pair": f"{source.strip()}->{target.strip()}",
                            "mode": spec.mode,
                            "layer": layer,
                            "strength": strength,
                            "edit": spec.to_json(),
                            "text": out["text"],
                            "changed": tokens != baseline,
                            "first_diff_token": first_diff,
                            "n_edit_forwards": out["n_edit_forwards"],
                            **_directed(spec.mode, source, target, baseline_text, out["text"]),
                        }
                    )
        print(f"  {sample.sample_id}: baseline={baseline_text[:60]!r}")

    report["results"] = results

    # ---- summary per (mode, layer, strength) --------------------------------------------
    cells: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    for row in results:
        cell = (
            cells.setdefault(row["mode"], {})
            .setdefault(str(row["layer"]), {})
            .setdefault(
                f"{row['strength']:g}",
                {"n": 0, "n_changed": 0, "n_directed": 0, "first_diff": {}},
            )
        )
        cell["n"] += 1
        cell["n_changed"] += int(row["changed"])
        cell["n_directed"] += int(row["directed"])
        if row["first_diff_token"] is not None:
            position = str(row["first_diff_token"])
            cell["first_diff"][position] = cell["first_diff"].get(position, 0) + 1

    summary: dict[str, object] = {}
    for mode, layer_cells in cells.items():
        summary[mode] = {
            layer: {
                strength: {
                    "n": cell["n"],
                    "n_changed": cell["n_changed"],
                    "change_rate": cell["n_changed"] / cell["n"],
                    "n_directed": cell["n_directed"],
                    "directed_rate": cell["n_directed"] / cell["n"],
                    "first_diff_distribution": dict(
                        sorted(cell["first_diff"].items(), key=lambda item: int(item[0]))
                    ),
                }
                for strength, cell in strength_cells.items()
            }
            for layer, strength_cells in layer_cells.items()
        }
    report["summary"] = summary

    minimal_effective: dict[str, object] = {}
    for mode, layer_cells in summary.items():
        minimal_effective[mode] = {}
        for layer, strength_cells in layer_cells.items():
            qualifying = sorted(
                (float(strength), cell["change_rate"])
                for strength, cell in strength_cells.items()
                if cell["change_rate"] >= MIN_CHANGE_RATE
            )
            minimal_effective[mode][layer] = qualifying[0][0] if qualifying else None
    report["minimal_effective"] = minimal_effective

    print("\n== x9b sweep summary (n / change_rate / directed_rate) ==")
    print(
        f"{'mode':<8}{'layer':>6}{'strength':>10}{'n':>7}{'change':>9}{'directed':>10}"
        "  first-diff positions (top)"
    )
    for mode in sorted(summary):
        for layer in sorted(summary[mode], key=int):
            for strength in sorted(summary[mode][layer], key=float):
                cell = summary[mode][layer][strength]
                top = " ".join(
                    f"{pos}:{count}"
                    for pos, count in sorted(
                        cell["first_diff_distribution"].items(),
                        key=lambda item: (-item[1], int(item[0])),
                    )[:4]
                )
                print(
                    f"{mode:<8}{layer:>6}{strength:>10}{cell['n']:>7}"
                    f"{cell['change_rate']:>9.3f}{cell['directed_rate']:>10.3f}  {top}"
                )

    print(
        f"\n== minimal effective strength (smallest strength with change_rate >= {MIN_CHANGE_RATE:g}) =="
    )
    for mode in sorted(minimal_effective):
        for layer in sorted(minimal_effective[mode], key=int):
            value = minimal_effective[mode][layer]
            if value is None:
                print(f"{mode:<8}L{layer}: none (bar not reached)")
            else:
                rate = summary[mode][layer][f"{value:g}"]["change_rate"]
                print(f"{mode:<8}L{layer}: {value:g} (change_rate={rate:.3f})")

    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.json} ({len(results)} generations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
