#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""X20: the workspace paper's methodological lens evals on the LLaVA-7B J-lens.

Replicates the six lens-quality evaluations of "Verbalizable Representations Form a
Global Workspace in Language Models" (Gurnee et al. 2026, transformer-circuits.pub/
2026/workspace, section methods-comparison) using the paper's own prompt data, vendored
at ``third_party/jacobian-lens/data/evaluations/`` (the upstream repo ships no scoring
code for these files, only the README conventions quoted below — the scorer here
implements those conventions verbatim).

Protocol (verbatim from the vendored ``data/evaluations/README.md``):

* **Lens readout** — at each (layer, token position) the lens returns a ranked list of
  vocabulary tokens (softmax(W_U @ norm(J_l @ h)); ranking after softmax equals ranking
  of the logits, so ranks are computed on the logits).
* **Readout position** — a single position per item, held fixed across all layers:
  - association, typo: the final prompt token (the closing period / the last tokenizer
    fragment of the misspelling);
  - multihop, multilingual, order-ops: the token immediately preceding ``target`` —
    since ``target`` is the expected continuation and is not part of the prompt, that is
    the last prompt token (``target`` itself is never scored);
  - poetry: the last newline token (end of line 1 of the couplet). Tokens are inspected
    with ``tokenizer.decode([id])``; if the tokenizer emits no standalone newline token
    the position falls back to the last prompt token (counted per suite).
* **Metric** — ``pass@k`` = mean over items of the fraction of ``intermediates`` whose
  min-over-layers lens rank <= k. Reported per suite per lens variant (min over the
  variant's scored layers), plus the per-layer curves (same fraction computed per layer)
  reindexed to 0-100 percent of depth. Ranks are 1-based with ties counting as the
  better rank (``s2_eval._rank_of_row`` semantics).
* **Order-ops expansion** — "each intermediate is a key expanded to a synonym set
  (numbers -> digit and word forms; operations -> symbol and word forms); rank is the
  min over single-token synonyms at each layer". The expansion table lives in
  ``ORDER_OPS_SYNONYMS``/``number_word`` below (the vendored data holds the keys only);
  synonyms that do not tokenize to a single token (leading-space form) are dropped at
  scoring time, and if none survives the key's word form first-subword is used as a
  fallback (counted in ``n_synonym_fallbacks``).
* **Intermediate token ids** — the llava-hf tokenizer emits a leading space as its own
  whitespace piece (decodes to ``''``), so a word's candidate set is judged on the
  non-whitespace pieces of its ``' ' + word`` form; the canonical id is the first
  non-whitespace piece (the piece the model must emit first to start the word, matching
  the next-token readout). Words whose form has extra subwords still score on that
  first piece and are counted in ``n_multi_token_intermediates`` (order-ops, per the
  README, only lets *single-token* synonyms into the min and falls back to the word
  form when none survives — both counted).
* **Workspace band** — the paper's 0-100 percent-of-depth reindex places the workspace
  band at ~[38, 92] percent, i.e. layers [12, 29] of a 32-layer model; the digest
  reports the band midpoint (layer 20) against the final layer.

Lens variants (the paper compares the J-lens against a logit lens):

* ``j-lens`` — the fitted ``lens-<mask>.pt`` Jacobian transport, raw (no bias). Layers
  outside the fitted set have no transport; the final layer is the model's own logits
  (the ``s2_eval`` convention).
* ``logit-lens`` — identity transport plus the calibrated bias payload
  (``bias-<mask>.pt``: per-dim mean/variance bias, optional scale/temp/logit_bias), i.e.
  ``unembed(s_l * (h_l + b_l)) / T_l + logit_bias`` at payload layers — our calibrated
  logit-lens baseline. The final layer is never corrected.
* ``identity-uncal`` — identity transport with no corrections, the vanilla logit lens
  ``unembed(norm(h_l))``.

Both identity variants read every layer 0..n_layers-1; the J-lens reads its fitted
layers plus the final layer. The final-layer logits are identical for every variant
(no corrections there) and are computed once.

Tiny backend: the tiny CPU fixture has a 128-token hash vocab, so the eval words are
resolved through the tiny tokenizer's own word-hash (every word is exactly one token)
and the lens/bias payloads are synthesized in-process when ``--lens-dir``/``--bias-dir``
are not given: fitted layers 1..n-2 (skip_first=1, exclude_last=True), Jacobians
``I + 0.05 * randn`` and biases ``0.05 * randn`` from a fixed seed, so the J-lens and
the calibrated identity readouts genuinely differ. Passing ``--lens-dir``/``--bias-dir``
with ``--backend tiny`` loads those files instead (they must match the tiny geometry).

The paper's claim under test: the J-lens beats the logit lens at early/mid depth, so
the digest prints the pass@10 gap (j-lens - logit-lens) at the workspace midpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402, I001  # the vlm_lens import must precede jlens: it installs the path
import vlm_lens  # noqa: E402, F401

from jlens.hooks import ActivationRecorder  # noqa: E402
from jlens.lens import JacobianLens  # noqa: E402

from vlm_lens.artifacts import load_bias, load_lens_set  # noqa: E402
from vlm_lens.models.llava import LlavaLensModel, MultimodalBatch  # noqa: E402

EVALS_DIR = REPO / "third_party" / "jacobian-lens" / "data" / "evaluations"
DEFAULT_LENS_DIR = "/data/anhnq/vlm-lens-out/validation/s2-merged/artifacts"
DEFAULT_BIAS_DIR = "/data/anhnq/vlm-lens-out/validation/step5e"

#: All six vendored suites, in README order.
ALL_SUITES = ("association", "multihop", "multilingual", "order-ops", "poetry", "typo")
ALL_VARIANTS = ("j-lens", "logit-lens", "identity-uncal")
DEFAULT_VARIANTS = ("j-lens", "logit-lens")

#: The paper's workspace band as a fraction of depth (0-100 reindex): ~[38, 92] percent
#: of depth -> layers [12, 29] for a 32-layer model.
BAND_LO_FRAC = 0.38
BAND_HI_FRAC = 0.92

PAPER_CITATION = (
    "Gurnee et al. 2026, 'Verbalizable Representations Form a Global Workspace in "
    "Language Models', transformer-circuits.pub/2026/workspace, section "
    "methods-comparison; prompts vendored from third_party/jacobian-lens/data/evaluations"
)

VARIANT_DESCRIPTIONS = {
    "j-lens": "fitted lens-<mask>.pt Jacobian transport, raw (no bias); "
    "unfitted layers have no readout, final layer = model logits",
    "logit-lens": "identity transport + bias-<mask>.pt calibrated payload "
    "(bias/scale/temp/logit_bias); the calibrated logit-lens baseline",
    "identity-uncal": "identity transport, no corrections (vanilla logit lens "
    "unembed(norm(h_l)))",
}

#: Order-ops synonym expansion: "numbers -> digit and word forms; operations -> symbol
#: and word forms" (data/evaluations/README.md). Single-token filter happens at scoring.
ORDER_OPS_SYNONYMS: dict[str, tuple[str, ...]] = {
    "addition": ("+", "addition", "plus", "add"),
    "subtraction": ("-", "subtraction", "minus", "subtract"),
    "multiplication": ("*", "multiplication", "times", "multiplied", "×"),
    "division": ("/", "division", "divided", "÷"),
    "mod": ("%", "mod", "modulo", "remainder"),
    "squared": ("^", "squared", "square", "²"),
}
_ONES = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def number_word(value: int) -> str:
    """English word form of a small non-negative integer (hyphenated 21-99)."""
    if value < len(_ONES):
        return _ONES[value]
    if value < 100:
        tens, ones = divmod(value, 10)
        return _TENS[tens] if ones == 0 else f"{_TENS[tens]}-{_ONES[ones]}"
    return str(value)


def expand_order_ops(key: str) -> tuple[str, ...]:
    """One order-ops intermediate key -> its synonym set (README convention)."""
    if key in ORDER_OPS_SYNONYMS:
        return ORDER_OPS_SYNONYMS[key]
    try:
        value = int(key)
    except ValueError:
        return (key,)
    forms = [key, number_word(value)]
    return tuple(dict.fromkeys(forms))  # dedupe, keep order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--backend", choices=("hf-llava", "tiny"), default="hf-llava",
        help="model backend: 'hf-llava' (default, CUDA) or the tiny CPU smoke fixture",
    )
    parser.add_argument("--device", default="", help="override the backend's device (e.g. cpu)")
    parser.add_argument("--lens-dir", default=DEFAULT_LENS_DIR, help="merged lens dir with lens-<mask>.pt")
    parser.add_argument("--bias-dir", default=DEFAULT_BIAS_DIR, help="dir holding bias-<mask>.pt (step5e payload)")
    parser.add_argument("--mask", default="text", help="which lens-<mask>.pt / bias-<mask>.pt to use")
    parser.add_argument(
        "--variants", default=",".join(DEFAULT_VARIANTS),
        help="comma list from: " + ", ".join(ALL_VARIANTS),
    )
    parser.add_argument("--suites", default=",".join(ALL_SUITES), help="comma list from: " + ", ".join(ALL_SUITES))
    parser.add_argument("--ks", default="1,5,10", help="comma list of k values for pass@k")
    parser.add_argument("--max-items", type=int, default=100, help="cap per suite (bounded runtime; 0 = no cap)")
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--evals-dir", default=str(EVALS_DIR), help="vendored evaluations directory")
    parser.add_argument("--json", required=True)
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> LlavaLensModel:
    """The eval model: the HF checkpoint (default, CUDA) or the tiny CPU fixture."""
    if args.backend == "tiny":
        from vlm_lens.models.tiny_llava import TinyLlavaConfig, build_tiny_llava

        hf_model, processor = build_tiny_llava(TinyLlavaConfig())
        return LlavaLensModel(hf_model, processor)
    device = args.device or "cuda"
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    return LlavaLensModel.from_pretrained(
        dtype=dtype, device=device, local_files_only=True
    )


def workspace_band(n_layers: int) -> tuple[int, int, int]:
    """(band_lo, band_hi, band_mid) from the paper's 0-100 reindex fractions."""
    lo = max(0, min(n_layers - 1, int(BAND_LO_FRAC * n_layers)))
    hi = max(lo, min(n_layers - 1, int(BAND_HI_FRAC * n_layers)))
    return lo, hi, (lo + hi) // 2


def synth_tiny_lens(model: LlavaLensModel) -> JacobianLens:
    """Identity + small seeded noise over the tiny fixture's fitted layers.

    Mirrors the production fit geometry (skip_first=1, exclude_last=True), so the
    J-lens variant reads fitted layers 1..n-2 plus the raw final layer exactly like
    the real lens does.
    """
    fitted = list(range(1, model.n_layers - 1))
    generator = torch.Generator().manual_seed(0)
    return JacobianLens(
        jacobians={
            layer: torch.eye(model.d_model) + 0.05 * torch.randn(model.d_model, model.d_model, generator=generator)
            for layer in fitted
        },
        n_prompts=16,
        d_model=model.d_model,
    )


def synth_tiny_bias(model: LlavaLensModel) -> dict[str, dict[str, Any]]:
    """Seeded per-layer bias payload (same layers as ``synth_tiny_lens``)."""
    generator = torch.Generator().manual_seed(1)
    return {
        str(layer): {"bias": 0.05 * torch.randn(model.d_model, generator=generator), "scale": 1.0}
        for layer in range(1, model.n_layers - 1)
    }


def load_calibrated_params(
    bias_dir: str, mask: str
) -> tuple[dict[int, torch.Tensor], dict[int, float], dict[int, float], dict[int, torch.Tensor], dict[str, Any]]:
    """The bias-<mask>.pt payload as lens_readout's four dicts, plus the file's meta."""
    payload, meta = load_bias(Path(bias_dir) / f"bias-{mask}.pt")
    bias = {int(layer): entry["bias"] for layer, entry in payload.items()}
    scale = {int(layer): float(entry["scale"]) for layer, entry in payload.items()}
    temp = {
        int(layer): float(entry["temp"])
        for layer, entry in payload.items()
        if "temp" in entry
    }
    logit_bias = {
        int(layer): entry["logit_bias"]
        for layer, entry in payload.items()
        if "logit_bias" in entry
    }
    return bias, scale, temp, logit_bias, meta


def variant_coverage(name: str, lens: JacobianLens, n_layers: int) -> list[int]:
    """Layers a variant reads out (see module docstring)."""
    if name == "j-lens":
        return sorted(set(lens.source_layers) | {n_layers - 1})
    return list(range(n_layers))


def _word_pieces(model: LlavaLensModel, word: str) -> list[int]:
    """Non-whitespace piece ids of a word's leading-space continuation form.

    The llava-hf tokenizer emits the leading space as its own whitespace piece
    (``' Brazil' -> [29871, 'Brazil-piece']``, the space piece decodes to ``''``), so
    whitespace-only pieces are dropped before judging; the tiny tokenizer hashes the
    whole word to one id.
    """
    tokenizer = model.tokenizer
    if hasattr(tokenizer, "encode_text"):  # TinyLlavaProcessor: one id per word
        return [int(tokenizer.encode_text(word)[1])]
    pieces = tokenizer(" " + word, add_special_tokens=False, return_tensors=None)["input_ids"]
    return [int(piece) for piece in pieces if tokenizer.decode([piece]).strip()]


def canonical_token_id(model: LlavaLensModel, word: str) -> tuple[int | None, int]:
    """(canonical id, n_word_pieces) for one intermediate word.

    The canonical id is the first non-whitespace piece of ``' ' + word`` — the piece
    the model must emit first to start the word, matching the next-token readout.
    ``(None, 0)`` for a word with no word pieces.
    """
    pieces = _word_pieces(model, word)
    return (pieces[0] if pieces else None), len(pieces)


def intermediate_ids(
    model: LlavaLensModel, suite: str, intermediates: list[str]
) -> tuple[list[list[int]], int, int]:
    """Per-intermediate candidate token ids for one item.

    Order-ops expands each key to its synonym set and, per the README, only
    single-token synonyms enter the min over candidates; a key with no single-token
    synonym falls back to its word-form canonical piece. Every other suite scores the
    word's canonical piece (multi-piece words still score, counted). Returns
    (candidates, n_multi_token, n_fallbacks).
    """
    candidates: list[list[int]] = []
    n_multi = 0
    n_fallback = 0
    for key in intermediates:
        if suite == "order-ops":
            singles = [canonical for form in expand_order_ops(key)
                       if len(pieces := _word_pieces(model, form)) == 1
                       for canonical in pieces]
            if singles:
                candidates.append(singles)
                continue
            canonical, _ = canonical_token_id(model, key)
            n_fallback += 1
            candidates.append([canonical] if canonical is not None else [])
        else:
            canonical, n_pieces = canonical_token_id(model, key)
            candidates.append([canonical] if canonical is not None else [])
            if n_pieces != 1:
                n_multi += 1
    return candidates, n_multi, n_fallback


def _per_intermediate_min_over_layers(record: dict[int, list[int]], layers: list[int]) -> list[int]:
    """Per intermediate: the min non-negative rank across the variant's layers (-1 if none)."""
    n_intermediates = len(next(iter(record.values())))
    return [
        min((record[layer][index] for layer in layers if record[layer][index] >= 0), default=-1)
        for index in range(n_intermediates)
    ]


def last_newline_position(model: LlavaLensModel, input_ids: torch.Tensor) -> int:
    """Index of the last token whose decoded form contains a newline, else -1."""
    ids = input_ids.tolist()
    for index in range(len(ids) - 1, -1, -1):
        if "\n" in model.tokenizer.decode([ids[index]]):
            return index
    return -1


def resolve_position(suite: str, batch: MultimodalBatch, model: LlavaLensModel) -> tuple[int, str]:
    """The suite's single readout position (see module docstring)."""
    if suite == "poetry":
        position = last_newline_position(model, batch.input_ids[0])
        if position >= 0:
            return position, "last-newline"
        return batch.seq_len - 1, "last-token(fallback:no-newline-token)"
    return batch.seq_len - 1, "last-token"


def rank_of_id(logits: torch.Tensor, token_id: int) -> int:
    """1-based rank of ``token_id`` in one logit row (ties count as better ranks)."""
    return int((logits > logits[token_id]).sum().item()) + 1


@torch.no_grad()
def item_ranks(
    model: LlavaLensModel,
    variants: dict[str, dict[str, Any]],
    coverage: dict[str, list[int]],
    batch: MultimodalBatch,
    position: int,
    candidates: list[list[int]],
    final_layer: int,
) -> dict[str, dict[int, list[int]]]:
    """Per variant, per layer: the 1-based rank of each intermediate's best candidate.

    One forward pass feeds every variant: residuals are recorded at the union of the
    variants' layers, transported per variant (J_l @ h for the fitted J-lens layers,
    identity otherwise), corrected per the calibrated payload, unembedded, and ranked
    against each intermediate's candidate set (min over candidates, ties as better
    rank). The final-layer row is shared by all variants (no corrections apply there).
    """
    wanted = sorted({layer for layers in coverage.values() for layer in layers})
    with ActivationRecorder(model.layers, at=wanted) as recorder:
        model.forward_mm(batch)

    final_row = model.unembed(recorder.activations[final_layer][0, position].float()).float().cpu()
    out: dict[str, dict[int, list[int]]] = {}
    for name, spec in variants.items():
        per_layer: dict[int, list[int]] = {}
        for layer in coverage[name]:
            if layer == final_layer:
                row = final_row
            else:
                residual = recorder.activations[layer][0, position].float()
                if name == "j-lens" and layer in spec["lens"].jacobians:
                    residual = spec["lens"].transport(residual, layer)
                if name == "logit-lens":
                    if layer in spec["bias"]:
                        residual = residual + spec["bias"][layer].to(residual.device)
                    if layer in spec["scale"]:
                        residual = residual * spec["scale"][layer]
                row = model.unembed(residual).float().cpu()
                if name == "logit-lens":
                    if layer in spec["temp"]:
                        row = row / spec["temp"][layer]
                    if layer in spec["logit_bias"]:
                        row = row + spec["logit_bias"][layer].to(row.device)
            per_layer[layer] = [
                min(rank_of_id(row, token_id) for token_id in group) if group else -1
                for group in candidates
            ]
        out[name] = per_layer
    return out


def aggregate_suite(
    suite: str,
    suite_items: list[dict[str, Any]],
    ks: tuple[int, ...],
    coverage: dict[str, list[int]],
    n_layers: int,
    band: tuple[int, int, int],
) -> dict[str, Any]:
    """Pass@k curves, headline min-over-layers pass@k, and the digest from item ranks."""
    band_lo, band_hi, band_mid = band
    final_layer = n_layers - 1
    per_layer: dict[str, list[dict[str, Any]]] = {}
    headline: dict[str, dict[str, float]] = {}
    digest: dict[str, dict[str, Any]] = {}
    for name in coverage:
        item_records = [item["ranks"][name] for item in suite_items]
        layers = coverage[name]

        curves: list[dict[str, Any]] = []
        for layer in layers:
            per_item = [record[layer] for record in item_records]
            flat = [rank for ranks in per_item for rank in ranks if rank >= 0]
            entry: dict[str, Any] = {
                "layer": layer,
                "pct_depth": round(100.0 * layer / (n_layers - 1), 2),
                "n_intermediates": len(flat),
            }
            for k in ks:
                entry[f"pass@{k}"] = round(
                    sum(
                        sum(1 for rank in ranks if 0 <= rank <= k) / max(1, len(ranks))
                        for ranks in per_item
                    ) / len(per_item),
                    4,
                )
            entry["mean_rank"] = round(sum(flat) / len(flat), 2) if flat else None
            curves.append(entry)
        per_layer[name] = curves

        # README headline: pass@k = mean over items of the fraction of intermediates
        # whose min-over-layers lens rank <= k. The min is per intermediate across the
        # variant's scored layers; a candidate-less intermediate (rank -1) never
        # passes but still counts in the fraction's denominator.
        headline[name] = {}
        for k in ks:
            headline[name][f"pass@{k}_min_over_layers"] = round(
                sum(
                    sum(1 for rank in mins if 0 <= rank <= k) / max(1, len(mins))
                    for mins in (
                        _per_intermediate_min_over_layers(record, layers)
                        for record in item_records
                    )
                ) / len(item_records),
                4,
            )

        by_layer = {entry["layer"]: entry for entry in curves}
        mid = by_layer.get(band_mid) or by_layer[max(by_layer, key=lambda layer: abs(layer - band_mid))]
        final = by_layer[final_layer]
        best = max(curves, key=lambda entry: (entry[f"pass@{ks[-1]}"], -entry["layer"]))
        digest[name] = {
            "best_layer": best["layer"],
            f"best_pass@{ks[-1]}": best[f"pass@{ks[-1]}"],
            "workspace_mid_layer": mid["layer"],
            f"pass@{ks[-1]}_at_workspace_mid": mid[f"pass@{ks[-1]}"],
            "mean_rank_at_workspace_mid": mid["mean_rank"],
            "final_layer": final["layer"],
            f"pass@{ks[-1]}_at_final_layer": final[f"pass@{ks[-1]}"],
            "mean_rank_at_final_layer": final["mean_rank"],
        }
    return {"per_layer": per_layer, "headline": headline, "digest": digest}


def score_suite(
    suite: str,
    items: list[dict[str, Any]],
    model: LlavaLensModel,
    variants: dict[str, dict[str, Any]],
    coverage: dict[str, list[int]],
    ks: tuple[int, ...],
    band: tuple[int, int, int],
    max_seq_len: int,
) -> dict[str, Any]:
    """Run one vendored suite: position rule, ranks per (variant, layer), aggregates."""
    n_layers = model.n_layers
    scored: list[dict[str, Any]] = []
    n_skipped = 0
    n_multi_total = 0
    n_fallback_total = 0
    rules: dict[str, int] = {}
    for item in items:
        try:
            batch = model.encode_mm(item["prompt"], None, max_length=max_seq_len)
        except ValueError:
            n_skipped += 1
            continue
        position, rule = resolve_position(suite, batch, model)
        rules[rule] = rules.get(rule, 0) + 1
        candidates, n_multi, n_fallback = intermediate_ids(model, suite, item["intermediates"])
        n_multi_total += n_multi
        n_fallback_total += n_fallback
        ranks = item_ranks(
            model,
            variants,
            coverage,
            batch,
            position,
            candidates,
            n_layers - 1,
        )
        scored.append(
            {
                "name": item.get("name"),
                "position": position,
                "position_rule": rule,
                "n_intermediates": len(item["intermediates"]),
                "ranks": ranks,
            }
        )

    aggregates = aggregate_suite(suite, scored, ks, coverage, n_layers, band)
    return {
        "n_items_loaded": len(items),
        "n_items_scored": len(scored),
        "n_items_skipped_too_long": n_skipped,
        "position_rules": rules,
        "n_multi_token_intermediates": n_multi_total,
        "n_synonym_fallbacks": n_fallback_total,
        "n_intermediates_total": sum(entry["n_intermediates"] for entry in scored),
        **aggregates,
    }


def print_digest(suite: str, result: dict[str, Any], ks: tuple[int, ...]) -> None:
    """One digest table per suite + the paper-claim gap line."""
    k = ks[-1]
    print(f"\n== {suite}: n={result['n_items_scored']}/{result['n_items_loaded']}"
          f" rules={result['position_rules']}")
    header = (
        f"{'variant':<16}{'best':>8}{f'p@{k}@best':>12}{f'p@{k}@mid':>12}"
        f"{f'rank@mid':>12}{f'p@{k}@final':>13}{f'rank@final':>13}"
    )
    print(header)
    for name, cell in result["digest"].items():
        print(
            f"{name:<16}"
            f"{'L' + str(cell['best_layer']):>8}"
            f"{cell[f'best_pass@{k}']:>12.4f}"
            f"{cell[f'pass@{k}_at_workspace_mid']:>12.4f}"
            f"{str(cell['mean_rank_at_workspace_mid']):>12}"
            f"{cell[f'pass@{k}_at_final_layer']:>13.4f}"
            f"{str(cell['mean_rank_at_final_layer']):>13}"
        )
    if "j-lens" in result["digest"] and "logit-lens" in result["digest"]:
        gap = (
            result["digest"]["j-lens"][f"pass@{k}_at_workspace_mid"]
            - result["digest"]["logit-lens"][f"pass@{k}_at_workspace_mid"]
        )
        verdict = "J-lens wins at mid-depth" if gap > 0 else "logit lens wins at mid-depth"
        print(f"   paper claim (J-lens > logit lens at workspace mid): gap = {gap:+.4f} -> {verdict}")


def main() -> int:
    args = parse_args()
    ks = tuple(int(part) for part in args.ks.split(",") if part.strip())
    suites = tuple(part.strip() for part in args.suites.split(",") if part.strip())
    unknown_suites = [suite for suite in suites if suite not in ALL_SUITES]
    if unknown_suites:
        raise SystemExit(f"unknown suites {unknown_suites}; choose from {ALL_SUITES}")
    variant_names = tuple(part.strip() for part in args.variants.split(",") if part.strip())
    unknown_variants = [name for name in variant_names if name not in ALL_VARIANTS]
    if unknown_variants:
        raise SystemExit(f"unknown variants {unknown_variants}; choose from {ALL_VARIANTS}")

    started = time.time()
    model = load_model(args)
    n_layers = model.n_layers
    band = workspace_band(n_layers)
    final_layer = n_layers - 1
    print(
        f"model ready: layers={n_layers} d_model={model.d_model} backend={args.backend}; "
        f"workspace band layers [{band[0]}, {band[1]}] (mid L{band[2]}), "
        f"band frac [{BAND_LO_FRAC}, {BAND_HI_FRAC}]"
    )

    # Tiny synthesizes in-process unless an explicit --lens-dir was given (the default
    # dir exists on the GPU box but holds a 4096-wide lens the tiny model cannot use).
    if args.backend == "tiny" and (
        args.lens_dir == DEFAULT_LENS_DIR or not Path(args.lens_dir).exists()
    ):
        lens = synth_tiny_lens(model)
        lens_source = "synthesized (tiny fixture, I + 0.05*randn, seed 0)"
        payload = synth_tiny_bias(model)
        bias_source = "synthesized (tiny fixture, 0.05*randn, seed 1)"
    else:
        lenses, _ = load_lens_set(args.lens_dir)
        if args.mask not in lenses:
            raise SystemExit(
                f"mask {args.mask!r} not in {args.lens_dir} (found {sorted(lenses)})"
            )
        lens = lenses[args.mask]
        lens_source = f"{args.lens_dir} (n_prompts={lens.n_prompts})"
        bias, scale, temp, logit_bias, bias_meta = load_calibrated_params(args.bias_dir, args.mask)
        payload = {
            str(layer): {
                key: value
                for key, value in (
                    ("bias", bias.get(layer)),
                    ("scale", scale.get(layer)),
                    ("temp", temp.get(layer)),
                    ("logit_bias", logit_bias.get(layer)),
                )
                if value is not None
            }
            for layer in bias
        }
        bias_source = f"{args.bias_dir}/bias-{args.mask}.pt (meta keys: {sorted(map(str, bias_meta))})"
    print(f"lens: {lens_source}; bias: {bias_source}")

    specs: dict[str, dict[str, Any]] = {
        "j-lens": {"lens": lens, "bias": {}, "scale": {}, "temp": {}, "logit_bias": {}},
        "logit-lens": {
            "lens": None,
            "bias": {int(layer): entry["bias"] for layer, entry in payload.items() if "bias" in entry},
            "scale": {int(layer): float(entry["scale"]) for layer, entry in payload.items() if "scale" in entry},
            "temp": {int(layer): float(entry["temp"]) for layer, entry in payload.items() if "temp" in entry},
            "logit_bias": {
                int(layer): entry["logit_bias"] for layer, entry in payload.items() if "logit_bias" in entry
            },
        },
        "identity-uncal": {"lens": None, "bias": {}, "scale": {}, "temp": {}, "logit_bias": {}},
    }
    variants = {name: specs[name] for name in variant_names}
    coverage = {name: variant_coverage(name, lens, n_layers) for name in variant_names}
    for name in variant_names:
        fitted = sorted(set(coverage[name]) - {final_layer})
        print(
            f"variant {name}: {len(coverage[name])} layers"
            f" (fitted-readout L{fitted[0]}..L{fitted[-1]} + final L{final_layer})"
            if name == "j-lens"
            else f"variant {name}: {len(coverage[name])} layers (all)"
        )

    evals_dir = Path(args.evals_dir)
    suite_results: dict[str, Any] = {}
    for suite in suites:
        path = evals_dir / f"lens-eval-{suite}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data["items"][: args.max_items] if args.max_items else data["items"]
        result = score_suite(
            suite, items, model, variants, coverage, ks, band, args.max_seq_len
        )
        suite_results[suite] = result
        print_digest(suite, result, ks)

    report = {
        "experiment": "X20_lens_evals",
        "paper": PAPER_CITATION,
        "backend": args.backend,
        "lens_dir": args.lens_dir,
        "lens_source": lens_source,
        "bias_dir": args.bias_dir,
        "bias_source": bias_source,
        "mask": args.mask,
        "max_items": args.max_items,
        "max_seq_len": args.max_seq_len,
        "ks": list(ks),
        "variants": {name: VARIANT_DESCRIPTIONS[name] for name in variant_names},
        "model": {
            "n_layers": n_layers,
            "d_model": model.d_model,
            "workspace_band": {"layers": [band[0], band[1]], "mid": band[2],
                               "depth_fractions": [BAND_LO_FRAC, BAND_HI_FRAC]},
            "final_layer": final_layer,
        },
        "lens": {
            "n_prompts": lens.n_prompts,
            "source_layers": sorted(lens.source_layers),
            "d_model": lens.d_model,
        },
        "suites": suite_results,
        "runtime_seconds": round(time.time() - started, 1),
    }
    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
