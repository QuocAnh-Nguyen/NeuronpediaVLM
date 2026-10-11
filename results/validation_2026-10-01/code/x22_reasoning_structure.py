#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""X22: the reasoning + structure replication arms (Gurnee et al. 2026 protocols).

Three arms, selected with ``--arm``; every arm runs on the calibrated checkpoint
(``--backend hf-llava``, CUDA) and on the tiny CPU fixture (``--backend tiny``,
random weights + an identity J-lens) so the full code path is exercisable without
a GPU or any download. All readouts use the RAW J-lens (``unembed(J_l h)``, no
bias/scale/temp corrections) — the paper's lens; see RAW_LENS_NOTE.

``--arm multihop`` — internal reasoning in two hops.
  (a) Probe: for the 93 vendored ``lens-eval-multihop`` items (and the 90
      ``probe-swap`` items), record the J-lens rank of the unspoken intermediate
      at the final prompt position, per fitted layer: best rank + best layer
      per item, pass@k over items (the paper's lens-quality metric).
  (b) Swap: the paper's lens-coordinate swap of the intermediate's direction
      for an alternative's (``probe-swap.json`` supplies ``swap_to`` /
      ``swap_answer``) — ``V = [v_s v_t]``, ``c = pinv(V) h``,
      ``h' = h + V (flip(c) - c)`` — applied at EVERY position during prefill
      AND decode, at every workspace-band layer. Success: the greedy answer
      becomes the target's answer (string match).
  (c) Depth control: the same swap restricted to the EARLY vs LATE half of the
      band, for (i) intermediate swaps and (ii) answer-token swaps; plus an
      onset sweep (shallowest effective swap layer per condition). The paper
      reports intermediate swaps taking effect ~17% of depth earlier than
      answer swaps; the digest reports the half-table and the median effective
      depth fraction per condition with their gap.

``--arm flexgen`` — flexible generalization over the vendored 192-trial set.
  16 function templates x 4 arguments x 3 same-category swap targets = 192
  directed swaps (count asserted), each run at alpha=1 and alpha=2: the
  argument's J-lens vector is swapped for the target's across the band at all
  positions; success = the function's output for the TARGET (top-1 string
  match). Each (template, argument) cell also gets a clean pass whose
  "workspace loading" is the mean cosine(h, v_arg) over the argument span and
  the final readout position, band-averaged; the digest correlates per-argument
  loading vs swap success (Pearson + Spearman).

``--arm structure`` — per-layer structure statistics over ~200 text prompts.
  Per lens layer: (a) top-k next-token accuracy of the J-lens readout vs the
  corpus's actual next token (k = 1, 5, 25) with the model's own top-1 as a
  reference; (b) excess kurtosis of the lens readout logit distribution across
  positions (per-position max logit, plus the pooled all-logit value);
  (c) top-1 lens-token autocorrelation at lags 1..8 against a
  position-shuffled null; (d) effective dimensionality — the fraction of
  d_model PCA directions capturing 90% of the variance of the J-lens vector
  matrix ``W_U J_l`` (exact via ``W_U^T W_U`` + centered covariance; optional
  row-sampled mode). The digest marks each metric's onset/offset layers by
  half-rise threshold. Corpus priority: ``--text-file`` > WikiText (if the
  ``datasets`` cache is available) > repo manifests > synthetic sentences.

Outputs one JSON report (``--json``) with per-arm rows, digest text and a
summary, following the x16 report layout.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
for _path in (str(REPO / "src"), str(CODE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import vlm_lens  # noqa: E402, F401  # installs the vendored jlens path: must precede jlens
from jlens.hooks import ActivationRecorder  # noqa: E402
from jlens.lens import JacobianLens  # noqa: E402
from vlm_lens._batch import as_batch  # noqa: E402
from vlm_lens.artifacts import load_lens_set  # noqa: E402
from vlm_lens.data.text import load_text_file_prompts, load_wikitext_prompts  # noqa: E402
from vlm_lens.interventions import (  # noqa: E402
    ResidualEdit,  # noqa: E402
    ResidualEditor,  # noqa: E402
    lens_vectors,  # noqa: E402
)
from vlm_lens.models.llava import LlavaLensModel, MultimodalBatch  # noqa: E402

# --------------------------------------------------------------- vendored data
VENDORED_DATA = REPO / "third_party" / "jacobian-lens" / "data"
DEFAULT_MULTIHOP_EVAL = VENDORED_DATA / "evaluations" / "lens-eval-multihop.json"
DEFAULT_PROBE_SWAP = VENDORED_DATA / "experiments" / "probe-swap.json"
DEFAULT_FLEXGEN = VENDORED_DATA / "experiments" / "flexible-generalization.json"

DEFAULT_LENS_DIR = "/data/anhnq/vlm-lens-out/validation/s2-merged/artifacts"
DEFAULT_MASK = "text"
#: The paper's workspace band, in layers of a 32-layer LLM (see context/README:
#: "experiments report over this band").
DEFAULT_BAND = (12, 29)
DEFAULT_MAX_SEQ_LEN = 1536

RAW_LENS_NOTE = (
    "all lens readouts are the RAW J-lens: unembed(J_l @ h) with the model's own "
    "final-norm + lm_head unembedding and no bias/scale/temp/logit_bias corrections "
    "(the paper uses the raw J-lens; the calibrated variant is a separate x14/x16 path)"
)
TOKEN_FORM_NOTE = (
    "token vectors use the form the model would emit at the readout position: "
    "' ' + word when the preceding prompt character is not whitespace, else the bare "
    "word; the probe rank takes the MIN over both forms (the order-ops synonym "
    "convention), the swap vectors use the single emitted form"
)
DEPTH_NOTE = (
    "depth control: the band splits into EARLY = band[:len//2] and LATE = "
    "band[len//2:]; the onset sweep walks levels shallow->deep (stride "
    "--onset-stride, deepest band layer always included) and an item's effective "
    "layer is the SHALLOWEST level whose swap flips the answer (early-stop); depth "
    "fractions are (layer - band_lo) / (band_hi - band_lo), and the paper's claim is "
    "that intermediate swaps take effect ~0.17 of depth earlier than answer swaps "
    "(i.e. a negative intermediate-minus-answer gap)"
)
SCORING_NOTE = (
    "answer scoring: normalize (lowercase, keep alphanumerics) both the greedy "
    "generation and the expected answer; success = the normalized generation "
    "STARTS WITH the normalized answer (covers multi-token answers like "
    "'North America' whose stored answer is 'North')"
)
LOADING_NOTE = (
    "workspace loading: mean over band layers and over (argument-span positions + "
    "final prompt position) of cosine(residual_h, v_arg) in the clean pass; "
    "v_arg = (W_U J_l)[arg] is the same row the swap edits along"
)
FLEXGEN_COUNT_NOTE = (
    "trial count: 16 function templates x 4 arguments x 3 same-category targets = "
    "192 directed swaps per alpha; alpha in {1, 2} doubles the swap generations"
)
CORPUS_NOTE = (
    "corpus priority: --text-file > repo manifests (runs/**/manifest*.jsonl text "
    "fields, '<image>' stripped) > WikiText-103 via datasets (if importable) > "
    "seeded synthetic sentences (function-word bank); each prompt is truncated to "
    "--positions-per-prompt + 1 tokens and position t's ground truth is token t+1"
)
ONSET_NOTE = (
    "onset modes: 'single' edits ONLY the level's layer per level (shallowest "
    "flipping layer = effective layer); 'cumulative' edits levels[i:] (the band "
    "from the level to the band end). The EARLY/LATE half table is the primary "
    "control; the sweep is the median-effective-layer statistic"
)


# ---------------------------------------------------------------------- helpers
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=("multihop", "flexgen", "structure"))
    parser.add_argument("--json", required=True, help="output JSON report path")
    parser.add_argument("--backend", choices=("hf-llava", "tiny"), default="hf-llava",
                        help="'hf-llava' (checkpoint, CUDA) or the tiny CPU smoke fixture")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16",
                        help="hf-llava checkpoint dtype (tiny is always fp32)")
    parser.add_argument("--lens-dir", default=DEFAULT_LENS_DIR,
                        help="J source lens dir; its fitted layers define the band and "
                             "the vectors (ignored for --backend tiny, which builds an "
                             "identity lens over all layers)")
    parser.add_argument("--mask", default=DEFAULT_MASK)
    parser.add_argument("--band", default=f"{DEFAULT_BAND[0]},{DEFAULT_BAND[1]}",
                        help="workspace band '<lo>,<hi>' in layer indices; layers outside "
                             "the lens's fitted set are dropped, and an empty "
                             "intersection falls back to the fitted middle half "
                             "(tiny smoke)")
    parser.add_argument("--n-items", type=int, default=0,
                        help="cap on items/cells for multihop and flexgen (0 = all)")
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--seed", type=int, default=0)
    # multihop / swap arms
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="swap strength for the multihop arms (flexgen sweeps "
                             "--flexgen-alphas instead)")
    parser.add_argument("--multihop-eval", default=str(DEFAULT_MULTIHOP_EVAL))
    parser.add_argument("--probe-swap", default=str(DEFAULT_PROBE_SWAP))
    parser.add_argument("--onset-sweep", dest="onset_sweep", action="store_true",
                        default=True, help="run the effective-onset sweep (default on)")
    parser.add_argument("--no-onset-sweep", dest="onset_sweep", action="store_false")
    parser.add_argument("--onset-mode", choices=("single", "cumulative"), default="single")
    parser.add_argument("--onset-stride", type=int, default=3,
                        help="level stride through the band for the onset sweep")
    # flexgen
    parser.add_argument("--flexgen-dataset", default=str(DEFAULT_FLEXGEN))
    parser.add_argument("--flexgen-alphas", default="1,2")
    # structure
    parser.add_argument("--n-prompts", type=int, default=200)
    parser.add_argument("--positions-per-prompt", type=int, default=64)
    parser.add_argument("--text-file", default=None,
                        help=".txt (blank-line blocks) or .jsonl ('text' field) corpus")
    parser.add_argument("--min-chars", type=int, default=200,
                        help="minimum characters per corpus block")
    parser.add_argument("--pca-sample-rows", type=int, default=0,
                        help="0 = exact PCA via W_U^T W_U; >0 = sample that many "
                             "vocab rows of W_U J_l before the SVD (cheap CPU mode)")
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> LlavaLensModel:
    """The arm model: the HF checkpoint (default, CUDA/bf16) or the tiny CPU fixture."""
    if args.backend == "tiny":
        from vlm_lens.models.tiny_llava import TinyLlavaConfig, build_tiny_llava

        hf_model, processor = build_tiny_llava(TinyLlavaConfig())
        return LlavaLensModel(hf_model, processor)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    return LlavaLensModel.from_pretrained(
        dtype=dtype, device=device, local_files_only=True
    )


def build_lens(args: argparse.Namespace, model: LlavaLensModel) -> tuple[JacobianLens, dict[str, Any]]:
    """The J-lens whose vectors drive every readout and edit.

    ``--backend tiny`` has no fitted artifact: an identity lens over ALL layers
    (x16's no-lens fallback), so ``v_t = W_U[t]`` and the whole swap/probe
    machinery runs end to end on random weights.
    """
    if args.backend == "tiny":
        lens = JacobianLens(
            jacobians={
                layer: torch.eye(model.d_model, dtype=torch.float32)
                for layer in range(model.n_layers)
            },
            n_prompts=0,
            d_model=model.d_model,
        )
        return lens, {"loaded": False, "source": "identity-over-all-layers",
                      "dir": None, "mask": args.mask}
    lenses, provenance = load_lens_set(args.lens_dir)
    if args.mask not in lenses:
        raise ValueError(f"mask {args.mask!r} not in {args.lens_dir} (found {sorted(lenses)})")
    lens = lenses[args.mask]
    if not lens.source_layers:
        raise ValueError(f"lens {args.mask!r} in {args.lens_dir} has no fitted layers")
    fitted = [layer for layer in lens.source_layers if 0 <= layer < model.n_layers]
    meta = {
        "loaded": True,
        "source": "fitted",
        "dir": str(args.lens_dir),
        "mask": args.mask,
        "n_prompts": lens.n_prompts,
        "fitted_layers": fitted,
        "dropped_out_of_range": [l for l in lens.source_layers
                                 if not 0 <= l < model.n_layers],
        "provenance_present": bool(provenance),
    }
    return lens, meta


def resolve_band(lens: JacobianLens, model: LlavaLensModel, lo: int, hi: int) -> tuple[list[int], str]:
    """The workspace band: fitted layers in [lo, hi]; middle-half fallback for smokes."""
    fitted = [layer for layer in lens.source_layers if 0 <= layer < model.n_layers]
    band = [layer for layer in fitted if lo <= layer <= hi]
    if band:
        return band, f"fitted ∩ [{lo},{hi}]"
    quarter = max(1, len(fitted) // 4)
    band = fitted[quarter: len(fitted) - quarter] or fitted
    return band, f"fallback-middle-half of fitted {fitted} (band [{lo},{hi}] empty)"


def _flat_ids(ids: Any) -> list[int]:
    """Tokenizer output (tensor or nested lists) as one flat id list."""
    if isinstance(ids, torch.Tensor):
        return [int(i) for i in ids.flatten().tolist()]
    flat: list[int] = []
    stack: list[Any] = list(ids)
    while stack:
        item = stack.pop(0)
        if isinstance(item, (list, tuple)):
            stack = list(item) + stack
        else:
            flat.append(int(item))
    return flat


def _content_ids(model: LlavaLensModel, text: str) -> list[int]:
    """Content ids of ``text``: flat, minus any bos the tokenizer force-prepends
    (the tiny fixture's ``encode_text`` adds bos even with
    ``add_special_tokens=False``; a real tokenizer's content ids never start
    with the bos id, so dropping a leading bos is safe for both)."""
    ids = _flat_ids(model.tokenizer(text, add_special_tokens=False)["input_ids"])
    bos = getattr(model.tokenizer, "bos_token_id", None)
    if bos is not None and ids and ids[0] == int(bos):
        ids = ids[1:]
    return ids


def token_id_of(model: LlavaLensModel, text: str) -> int:
    """Last subword id of ``text`` (the token that would be emitted there)."""
    ids = _content_ids(model, text)
    if not ids:
        raise ValueError(f"tokenizer produced no ids for {text!r}")
    return ids[-1]


def form_ids(model: LlavaLensModel, word: str) -> list[int]:
    """Last-subword ids of both emitted forms (' word' preferred, then 'word')."""
    out: list[int] = []
    for form in (f" {word}", word):
        tid = token_id_of(model, form)
        out.append(tid)
    return out


def emitted_form(prompt: str, word: str) -> str:
    """The token string the model would emit for ``word`` after ``prompt``."""
    return word if (prompt.endswith(" ") or not prompt) else f" {word}"


def rank_of_id(logits: torch.Tensor, token_id: int) -> int:
    """1-based rank of ``token_id`` in the logit row (ties count as better ranks)."""
    return int((logits > logits[token_id]).sum().item()) + 1


def norm_answer(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def answer_match(generated: str, answer: str) -> bool:
    """SCORING_NOTE: normalized generation starts with the normalized answer."""
    want = norm_answer(answer)
    return bool(want) and norm_answer(generated).startswith(want)


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None
    a = torch.tensor(x, dtype=torch.float64)
    b = torch.tensor(y, dtype=torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom <= 0:
        return None
    return float((a * b).sum() / denom)


def _ranks(values: list[float]) -> torch.Tensor:
    """1-based ranks, ties averaged."""
    t = torch.tensor(values, dtype=torch.float64)
    order = torch.argsort(t)
    ranks = torch.empty_like(t)
    ranks[order] = torch.arange(1, len(t) + 1, dtype=torch.float64)
    # average ties
    for value in torch.unique(t[t.argsort()]):
        mask = t == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None
    return pearson(_ranks(x).tolist(), _ranks(y).tolist())


def excess_kurtosis(values: torch.Tensor) -> float:
    """Population excess kurtosis m4/m2^2 - 3 of a tensor of samples."""
    x = values.detach().float().flatten()
    if x.numel() < 3:
        return float("nan")
    centered = x - x.mean()
    m2 = (centered**2).mean()
    if m2 <= 0:
        return float("nan")
    m4 = (centered**4).mean()
    return float(m4 / m2**2 - 3.0)


# ------------------------------------------------------------- forwards/edits
@torch.no_grad()
def recorded_forward(
    model: LlavaLensModel, batch: MultimodalBatch, layers: list[int]
) -> dict[int, torch.Tensor]:
    """One no-grad forward: block-output residuals ``[seq, d]`` fp32 CPU per layer."""
    recorder = ActivationRecorder(model.layers, at=layers)
    with recorder:
        model.hf_model(**batch.hf_kwargs(), use_cache=False)
    return {
        layer: recorder.activations[layer][0].detach().float().cpu() for layer in layers
    }


@torch.no_grad()
def lens_last_logits(
    model: LlavaLensModel,
    lens: JacobianLens,
    residuals: dict[int, torch.Tensor],
    layer: int,
) -> torch.Tensor:
    """Raw J-lens logits at the LAST position of a recorded residual ``[seq, d]``."""
    hidden = residuals[layer][-1]
    return model.unembed(lens.transport(hidden, layer)).float().cpu()


class CachedVectorEditor(ResidualEditor):
    """:class:`ResidualEditor` with the swap vectors resolved once up front.

    The hook math is the parent's verbatim (``apply_edit`` on the block output,
    ``positions="all"`` covering prefill AND every decode forward). Only the
    vector resolution differs: the multihop/flexgen arms reuse identical
    ``(layer, token-pair)`` vectors across many trials, and re-running
    ``lens_vectors`` per edit would re-copy the ``[vocab, d_model]`` unembedding
    for every band layer of every trial.
    """

    def __init__(
        self,
        model: LlavaLensModel,
        edits: list[ResidualEdit],
        vectors_by_layer: dict[int, dict[str, torch.Tensor]],
    ) -> None:
        self._model = model
        self._grouped: dict[int, list[tuple[ResidualEdit, dict[str, torch.Tensor]]]] = {}
        for edit in edits:
            layer_vectors = vectors_by_layer[edit.layer]
            payload = {
                role: layer_vectors[role]
                for role in ("source", "target")
                if role in layer_vectors
            }
            self._grouped.setdefault(edit.layer, []).append((edit, payload))
        self._handles: list[Any] = []
        self.forward_calls = 0


@torch.no_grad()
def generate_swapped(
    model: LlavaLensModel,
    sample: str | MultimodalBatch,
    edits: list[ResidualEdit],
    vectors_by_layer: dict[int, dict[str, torch.Tensor]],
    *,
    max_new_tokens: int,
    max_seq_len: int,
) -> dict[str, Any]:
    """Greedy generation under swap edits (empty ``edits`` = baseline)."""
    batch = sample if isinstance(sample, MultimodalBatch) else as_batch(model, sample, max_seq_len)
    with CachedVectorEditor(model, edits, vectors_by_layer):
        generated = model.hf_model.generate(
            **batch.hf_kwargs(),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    new_tokens = generated[0, batch.seq_len:].detach().cpu()
    return {
        "text": model._decode(new_tokens),
        "token_ids": [int(t) for t in new_tokens],
        "first_token": model._decode(new_tokens[:1]),
    }


def swap_edits(
    layers: list[int], source_token: str, target_token: str, alpha: float
) -> list[ResidualEdit]:
    """One ``swap`` spec per band layer, at ALL positions (prefill + decode)."""
    return [
        ResidualEdit(layer=layer, mode="swap", alpha=alpha,
                     token=target_token, source_token=source_token, positions="all")
        for layer in layers
    ]


def probe_intermediates(
    model: LlavaLensModel,
    lens: JacobianLens,
    entries: list[dict[str, Any]],
    *,
    max_seq_len: int,
    progress_label: str,
) -> list[dict[str, Any]]:
    """(a) the unspoken intermediate's J-lens rank at the final prompt position.

    Per entry: min-over-forms rank per fitted layer; ``best_rank``/``best_layer``
    over all fitted layers. Entries carry ``intermediates`` (a list; the vendored
    multihop eval allows several, probe-swap has one).
    """
    layers = [l for l in lens.source_layers if 0 <= l < model.n_layers]
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        batch = as_batch(model, entry["prompt"], max_seq_len)
        residuals = recorded_forward(model, batch, layers)
        candidate_ids = [tid for word in entry["intermediates"]
                         for tid in form_ids(model, word)]
        per_layer: dict[str, int] = {}
        for layer in layers:
            logits = lens_last_logits(model, lens, residuals, layer)
            per_layer[str(layer)] = min(rank_of_id(logits, tid) for tid in candidate_ids)
        best_layer = min(per_layer, key=lambda key: per_layer[key])
        rows.append({
            "name": entry["name"],
            "intermediates": entry["intermediates"],
            "best_rank": per_layer[best_layer],
            "best_layer": int(best_layer),
            "per_layer_rank": per_layer,
        })
        if (index + 1) % 10 == 0 or index + 1 == len(entries):
            print(f"  [{progress_label} probe {index + 1}/{len(entries)}] "
                  f"{entry['name']}: best rank {rows[-1]['best_rank']} "
                  f"@ layer {best_layer}")
    return rows


# ------------------------------------------------------------------ multihop
def run_multihop(args: argparse.Namespace, model: LlavaLensModel, lens: JacobianLens,
                 band: list[int]) -> dict[str, Any]:
    started = time.time()
    eval_items = json.loads(Path(args.multihop_eval).read_text(encoding="utf-8"))["items"]
    swap_items = json.loads(Path(args.probe_swap).read_text(encoding="utf-8"))["items"]
    if args.n_items:
        eval_items, swap_items = eval_items[:args.n_items], swap_items[:args.n_items]
    eval_entries = [{"name": item["name"], "prompt": item["prompt"],
                     "intermediates": list(item["intermediates"])} for item in eval_items]
    swap_entries = [{"name": item["name"], "prompt": item["prompt"],
                     "intermediates": [item["intermediate"]]} for item in swap_items]

    print(f"-- (a) intermediate probe: {len(eval_entries)} lens-eval-multihop + "
          f"{len(swap_entries)} probe-swap items --")
    eval_probe = probe_intermediates(model, lens, eval_entries,
                                     max_seq_len=args.max_seq_len, progress_label="eval")
    swap_probe = probe_intermediates(model, lens, swap_entries,
                                     max_seq_len=args.max_seq_len, progress_label="swap")

    # Swap vectors: one lens_vectors call over the whole band (single W_U copy).
    for item in swap_items:
        item["_src"] = emitted_form(item["prompt"], item["intermediate"])
        item["_tgt"] = emitted_form(item["prompt"], item["swap_to"])
        item["_ans_src"] = emitted_form(item["prompt"], item["answer"])
        item["_ans_tgt"] = emitted_form(item["prompt"], item["swap_answer"])
    token_list = sorted({tok for item in swap_items
                         for tok in (item["_src"], item["_tgt"],
                                     item["_ans_src"], item["_ans_tgt"])})
    token_index = {tok: i for i, tok in enumerate(token_list)}
    vectors = lens_vectors(model, lens, token_list, layers=band)

    def pair_vectors(source: str, target: str) -> dict[int, dict[str, torch.Tensor]]:
        return {
            layer: {"source": vectors[layer][token_index[source]],
                    "target": vectors[layer][token_index[target]]}
            for layer in band
        }

    halves = half_split(band)
    conditions = (
        ("intermediate", "_src", "_tgt", "swap_answer"),
        ("answer", "_ans_src", "_ans_tgt", "swap_answer"),
    )
    print(f"-- (b) baseline + swaps: {len(swap_items)} items, alpha={args.alpha}, "
          f"band={band[0]}..{band[-1]} ({len(band)} layers) --")
    swap_rows: list[dict[str, Any]] = []
    for index, item in enumerate(swap_items):
        base = generate_swapped(model, item["prompt"], [], {},
                                max_new_tokens=args.max_new_tokens,
                                max_seq_len=args.max_seq_len)
        baseline_correct = answer_match(base["text"], item["answer"])
        arms: dict[str, dict[str, Any]] = {}
        for cond, source_key, target_key, expect_key in conditions:
            vectors_by_layer = pair_vectors(item[source_key], item[target_key])
            for half_name, half_layers in halves.items():
                out = generate_swapped(
                    model, item["prompt"], swap_edits(half_layers, item[source_key],
                                                      item[target_key], args.alpha),
                    vectors_by_layer,
                    max_new_tokens=args.max_new_tokens,
                    max_seq_len=args.max_seq_len,
                )
                arms[f"{cond}_{half_name}"] = {
                    "text": out["text"], "first_token": out["first_token"],
                    "success": answer_match(out["text"], item[expect_key]),
                    "n_layers": len(half_layers),
                }
        swap_rows.append({
            "name": item["name"], "category": item["category"],
            "prompt": item["prompt"],
            "intermediate": item["intermediate"], "answer": item["answer"],
            "swap_to": item["swap_to"], "swap_answer": item["swap_answer"],
            "baseline_text": base["text"],
            "baseline_correct": baseline_correct,
            "arms": arms,
        })
        if (index + 1) % 10 == 0 or index + 1 == len(swap_items):
            print(f"  [swap {index + 1}/{len(swap_items)}] {item['name']}: "
                  f"baseline={base['text']!r} correct={baseline_correct}")

    onset_rows: dict[str, list[dict[str, Any]]] = {}
    if args.onset_sweep:
        levels = band[:: max(1, args.onset_stride)]
        if levels[-1] != band[-1]:
            levels.append(band[-1])
        print(f"-- (c) onset sweep ({args.onset_mode}, stride {args.onset_stride}, "
              f"levels {levels}) --")
        for cond, source_key, target_key, expect_key in conditions:
            cond_rows: list[dict[str, Any]] = []
            for index, item in enumerate(swap_items):
                if not swap_rows[index]["baseline_correct"]:
                    continue
                vectors_by_layer = pair_vectors(item[source_key], item[target_key])
                level_results: dict[str, bool] = {}
                effective: int | None = None
                for level_index, level in enumerate(levels):
                    arm_layers = (
                        levels[level_index:] if args.onset_mode == "cumulative" else [level]
                    )
                    out = generate_swapped(
                        model, item["prompt"],
                        swap_edits(arm_layers, item[source_key], item[target_key],
                                   args.alpha),
                        vectors_by_layer,
                        max_new_tokens=args.max_new_tokens,
                        max_seq_len=args.max_seq_len,
                    )
                    success = answer_match(out["text"], item[expect_key])
                    level_results[str(level)] = success
                    if success and effective is None:
                        effective = level
                        break  # shallowest effective level found
                cond_rows.append({
                    "name": item["name"],
                    "levels": level_results,
                    "effective_layer": effective,
                    "effective_frac": (
                        None if effective is None
                        else (effective - band[0]) / (band[-1] - band[0])
                    ),
                })
            onset_rows[cond] = cond_rows
            n_effective = sum(1 for row in cond_rows if row["effective_layer"] is not None)
            print(f"  [{cond}] effective on {n_effective}/{len(cond_rows)} "
                  f"baseline-correct items")

    probe_stats = {
        "lens-eval-multihop": _passk(eval_probe, band),
        "probe-swap": _passk(swap_probe, band),
    }
    arm_stats = {arm: _arm_success(swap_rows, arm)
                 for arm in ("intermediate_early", "intermediate_late",
                             "answer_early", "answer_late")}
    onset_stats = {cond: _median_frac(onset_rows.get(cond, []))
                   for cond in ("intermediate", "answer")}
    results = {
        "alpha": args.alpha,
        "band": band,
        "early_half": halves["early"],
        "late_half": halves["late"],
        "probe_stats": probe_stats,
        "arm_stats": arm_stats,
        "onset_stats": onset_stats,
        "probe_eval": eval_probe,
        "probe_swap": swap_probe,
        "swap_rows": swap_rows,
        "onset_rows": onset_rows,
    }
    digest = digest_multihop(results)
    print("\n" + digest)
    return {
        "results": results,
        "digest": digest,
        "summary": _multihop_summary(probe_stats, arm_stats, onset_stats),
        "runtime_s": round(time.time() - started, 1),
    }


def half_split(band: list[int]) -> dict[str, list[int]]:
    """EARLY = first half of the band, LATE = the rest (LATE keeps the odd layer)."""
    cut = len(band) // 2
    return {"early": band[:cut], "late": band[cut:]}


def _passk(probe_rows: list[dict[str, Any]], band: list[int]) -> dict[str, Any]:
    """Paper's pass@k: fraction of items whose min-over-layers intermediate rank <= k."""
    best = [row["best_rank"] for row in probe_rows]
    band_best = []
    for row in probe_rows:
        in_band = [rank for layer, rank in row["per_layer_rank"].items()
                   if int(layer) in set(band)]
        band_best.append(min(in_band) if in_band else None)
    return {
        "n_items": len(probe_rows),
        "pass@1": sum(1 for r in best if r <= 1) / len(best) if best else None,
        "pass@3": sum(1 for r in best if r <= 3) / len(best) if best else None,
        "pass@5": sum(1 for r in best if r <= 5) / len(best) if best else None,
        "mean_best_rank": sum(best) / len(best) if best else None,
        "median_best_rank": median([float(r) for r in best]),
        "median_best_layer": median([float(row["best_layer"]) for row in probe_rows]),
        "band_pass@1": (
            sum(1 for r in band_best if r is not None and r <= 1) / len(band_best)
            if band_best else None
        ),
    }


def _arm_success(swap_rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    total = len(swap_rows)
    gated = [row for row in swap_rows if row["baseline_correct"]]
    hit = sum(1 for row in swap_rows if row["arms"][arm]["success"])
    hit_gated = sum(1 for row in gated if row["arms"][arm]["success"])
    return {
        "n_all": total,
        "success_all": hit / total if total else None,
        "n_baseline_correct": len(gated),
        "success_baseline_correct": hit_gated / len(gated) if gated else None,
    }


def _median_frac(onset_rows: list[dict[str, Any]]) -> dict[str, Any]:
    fracs = [row["effective_frac"] for row in onset_rows if row["effective_frac"] is not None]
    return {
        "n_effective": len(fracs),
        "n_items": len(onset_rows),
        "median_effective_frac": median(fracs),
    }


def _multihop_summary(probe_stats: dict[str, dict[str, Any]],
                      arm_stats: dict[str, dict[str, Any]],
                      onset_stats: dict[str, dict[str, Any]]) -> str:
    eval_stats = probe_stats["lens-eval-multihop"]
    intermediate = onset_stats["intermediate"]["median_effective_frac"]
    answer = onset_stats["answer"]["median_effective_frac"]
    gap = intermediate - answer if (intermediate is not None and answer is not None) else None
    return (
        f"multihop: probe pass@1={_cell(eval_stats['pass@1'])} "
        f"(n={eval_stats['n_items']}); intermediate swap early/late="
        f"{_cell(arm_stats['intermediate_early']['success_baseline_correct'])}/"
        f"{_cell(arm_stats['intermediate_late']['success_baseline_correct'])}; "
        f"answer swap early/late="
        f"{_cell(arm_stats['answer_early']['success_baseline_correct'])}/"
        f"{_cell(arm_stats['answer_late']['success_baseline_correct'])}; "
        f"effective-onset gap={_cell(gap)} (paper: ~-0.17)"
    )


def digest_multihop(results: dict[str, Any]) -> str:
    band = results["band"]
    lines = [
        "== X22 multihop digest ==",
        f"band {band[0]}..{band[-1]} ({len(band)} layers) | early "
        f"{results['early_half']} | late {results['late_half']} | "
        f"alpha={results['alpha']}",
        "",
        "-- (a) intermediate probe (raw J-lens, final prompt position) --",
    ]
    for label, stats in results["probe_stats"].items():
        lines.append(
            f"  {label:<18} n={stats['n_items']:>3}  "
            f"pass@1={_cell(stats['pass@1'])}  pass@3={_cell(stats['pass@3'])}  "
            f"pass@5={_cell(stats['pass@5'])}  "
            f"band-pass@1={_cell(stats['band_pass@1'])}  "
            f"mean_best_rank={_cell(stats['mean_best_rank'], 6, 1)}  "
            f"median_best_layer={_cell(stats['median_best_layer'], 4, 0)}"
        )
    lines += ["", "-- (b) swap success (answer becomes the target's answer) --"]
    for arm, stats in results["arm_stats"].items():
        lines.append(
            f"  {arm:<20} success={_cell(stats['success_baseline_correct'])} "
            f"(on baseline-correct n={stats['n_baseline_correct']})  "
            f"all={_cell(stats['success_all'])}"
        )
    lines += ["", "-- (c) depth control (effective onset, band-relative fraction) --"]
    for cond, stats in results["onset_stats"].items():
        lines.append(
            f"  {cond:<13} effective {stats['n_effective']}/{stats['n_items']}  "
            f"median_frac={_cell(stats['median_effective_frac'])}"
        )
    intermediate = results["onset_stats"]["intermediate"]["median_effective_frac"]
    answer = results["onset_stats"]["answer"]["median_effective_frac"]
    if intermediate is not None and answer is not None:
        lines.append(
            f"  precedence gap (intermediate - answer) = {intermediate - answer:+.3f} "
            f"of band depth (paper: intermediate ~0.17 EARLIER, i.e. gap < 0)"
        )
    return "\n".join(lines)


def _cell(value: float | None, width: int = 6, precision: int = 3) -> str:
    return f"{value:>{width}.{precision}f}" if value is not None else f"{'-':>{width}}"


# ------------------------------------------------------------------- flexgen
def find_arg_positions(
    model: LlavaLensModel, prompt_ids: list[int], arg: str
) -> tuple[list[int], str | None]:
    """Token positions of the argument span in the prompt (space form first)."""
    for form in (f" {arg}", arg):
        ids = _content_ids(model, form)
        if not ids:
            continue
        for start in range(len(prompt_ids) - len(ids) + 1):
            if prompt_ids[start:start + len(ids)] == ids:
                return list(range(start, start + len(ids))), form
    return [], None


def run_flexgen(args: argparse.Namespace, model: LlavaLensModel, lens: JacobianLens,
                band: list[int]) -> dict[str, Any]:
    started = time.time()
    data = json.loads(Path(args.flexgen_dataset).read_text(encoding="utf-8"))
    categories = data["categories"]
    n_funcs = sum(len(cat["funcs"]) for cat in categories)
    n_trials = sum(len(cat["funcs"]) * len(cat["args"]) * (len(cat["args"]) - 1)
                   for cat in categories)
    expected = 16 * 4 * 3
    count_note = (
        f"{n_funcs} funcs x 4 args x 3 targets = {n_trials} directed swaps per alpha "
        f"(paper count: {expected})"
    )
    print(f"-- flexgen: {count_note} --")

    def slot_form(head: str, arg: str) -> str:
        """The emitted form of the arg at this template's slot (TOKEN_FORM_NOTE)."""
        return arg if head.endswith((" ", '"', "'")) else f" {arg}"

    form_list = sorted({slot_form(func["template"].split("{arg}", 1)[0], arg)
                        for cat in categories for func in cat["funcs"]
                        for arg in cat["args"]})
    form_index = {form: i for i, form in enumerate(form_list)}
    vectors = lens_vectors(model, lens, form_list, layers=band)
    alphas = [float(part) for part in args.flexgen_alphas.split(",") if part.strip()]

    cells: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    cell_total = sum(len(cat["funcs"]) * len(cat["args"]) for cat in categories)
    capped_note = f" (capped to --n-items={args.n_items})" if args.n_items else ""
    done = 0
    print(f"-- baseline + loading clean passes over {cell_total} cells{capped_note}, "
          f"then {n_trials} swaps x alphas {alphas} --")
    for cat in categories:
        for func in cat["funcs"]:
            for arg in cat["args"]:
                if args.n_items and done >= args.n_items:
                    break
                done += 1
                prompt = func["template"].replace("{arg}", arg)
                head = func["template"].split("{arg}", 1)[0]
                form = slot_form(head, arg)
                batch = as_batch(model, prompt, args.max_seq_len)
                prompt_ids = batch.input_ids[0].tolist()
                positions, matched_form = find_arg_positions(model, prompt_ids, arg)
                scored_positions = positions + [len(prompt_ids) - 1]
                residuals = recorded_forward(model, batch, band)
                arg_row = form_index[form]
                per_layer_cos: dict[str, float] = {}
                cosines: list[torch.Tensor] = []
                for layer in band:
                    hidden = residuals[layer][scored_positions]
                    v = vectors[layer][arg_row].to(hidden.device, hidden.dtype).expand_as(hidden)
                    cos = F.cosine_similarity(hidden, v, dim=-1)
                    per_layer_cos[str(layer)] = float(cos.mean())
                    cosines.append(cos)
                loading = float(torch.cat(cosines).mean())

                base = generate_swapped(model, batch, [], {},
                                        max_new_tokens=args.max_new_tokens,
                                        max_seq_len=args.max_seq_len)
                baseline_correct = answer_match(base["text"], func["answers"][arg])
                per_target: dict[str, dict[str, Any]] = {}
                for target in cat["args"]:
                    if target == arg:
                        continue
                    tgt_row = form_index[slot_form(head, target)]
                    vectors_by_layer = {
                        layer: {"source": vectors[layer][arg_row],
                                "target": vectors[layer][tgt_row]}
                        for layer in band
                    }
                    for alpha in alphas:
                        out = generate_swapped(
                            model, batch,
                            swap_edits(band, form, slot_form(head, target), alpha),
                            vectors_by_layer,
                            max_new_tokens=args.max_new_tokens,
                            max_seq_len=args.max_seq_len,
                        )
                        success = answer_match(out["text"], func["answers"][target])
                        per_target.setdefault(target, {})[str(alpha)] = success
                        trial_rows.append({
                            "category": cat["name"], "func": func["name"],
                            "arg": arg, "target": target, "alpha": alpha,
                            "baseline_correct": baseline_correct,
                            "generated": out["text"], "success": success,
                        })
                cells.append({
                    "category": cat["name"], "func": func["name"], "arg": arg,
                    "prompt": prompt,
                    "arg_positions": positions, "arg_form_matched": matched_form,
                    "loading": loading, "loading_per_layer": per_layer_cos,
                    "baseline_text": base["text"],
                    "baseline_correct": baseline_correct,
                    "swap_success": per_target,
                })
                if done % 8 == 0 or done == cell_total:
                    print(f"  [cell {done}/{cell_total}] {cat['name']}/{func['name']}"
                          f"/{arg}: loading={loading:.3f} baseline_correct="
                          f"{baseline_correct}")

    # per-argument aggregates + the loading-predicts-success correlation
    arg_stats: list[dict[str, Any]] = []
    for cat in categories:
        for arg in cat["args"]:
            arg_cells = [cell for cell in cells
                         if cell["category"] == cat["name"] and cell["arg"] == arg]
            if not arg_cells:  # --n-items caps the cell set; aggregate only present args
                continue
            per_alpha_success: dict[str, float | None] = {}
            for alpha in alphas:
                key = str(alpha)
                gated_hits = gated_n = 0
                for cell in arg_cells:
                    for per_alpha in cell["swap_success"].values():
                        if key not in per_alpha:
                            continue
                        gated_n += 1
                        gated_hits += int(per_alpha[key] and cell["baseline_correct"])
                per_alpha_success[key] = gated_hits / gated_n if gated_n else None
            arg_stats.append({
                "category": cat["name"], "arg": arg,
                "loading": sum(cell["loading"] for cell in arg_cells) / len(arg_cells),
                "success": per_alpha_success,
                "n_trials": sum(len(s) for cell in arg_cells
                                for s in cell["swap_success"].values()),
            })

    primary = str(alphas[0])
    xs = [entry["loading"] for entry in arg_stats]
    ys = [entry["success"][primary] for entry in arg_stats]
    paired = [(x, y) for x, y in zip(xs, ys) if y is not None]
    correlation = {
        "alpha": alphas[0],
        "n_points": len(paired),
        "pearson": pearson([p[0] for p in paired], [p[1] for p in paired]) if paired else None,
        "spearman": spearman([p[0] for p in paired], [p[1] for p in paired]) if paired else None,
        "per_category": {},
    }
    for cat in categories:
        sub = [entry for entry in arg_stats if entry["category"] == cat["name"]]
        sx = [entry["loading"] for entry in sub]
        sy = [entry["success"][primary] for entry in sub]
        keep = [(x, y) for x, y in zip(sx, sy) if y is not None]
        correlation["per_category"][cat["name"]] = {
            "n_points": len(keep),
            "pearson": pearson([p[0] for p in keep], [p[1] for p in keep]) if len(keep) >= 3 else None,
        } if keep else {"n_points": 0, "pearson": None}

    results = {
        "n_funcs": n_funcs,
        "n_trials_per_alpha": n_trials,
        "expected_trials_per_alpha": expected,
        "count_note": count_note,
        "alphas": alphas,
        "cells": cells,
        "trial_rows": trial_rows,
        "arg_stats": arg_stats,
        "correlation": correlation,
        "band": band,
    }
    digest = digest_flexgen(results)
    print("\n" + digest)
    summary = (
        f"flexgen: {n_trials} directed swaps per alpha; success(a=1)="
        f"{_cell(arm_rate(results, alphas[0], gated=True))} "
        f"(a={alphas[-1]})={_cell(arm_rate(results, alphas[-1], gated=True))}; "
        f"loading-success pearson={_cell(correlation['pearson'])} "
        f"spearman={_cell(correlation['spearman'])}"
    )
    return {
        "results": results,
        "digest": digest,
        "summary": summary,
        "runtime_s": round(time.time() - started, 1),
    }


def arm_rate(results: dict[str, Any], alpha: float, *, gated: bool) -> float | None:
    """Swap success at one alpha over all trial rows (optionally baseline-gated)."""
    key = str(alpha)
    hits = n = 0
    for cell in results["cells"]:
        if gated and not cell["baseline_correct"]:
            continue
        for per_alpha in cell["swap_success"].values():
            if key in per_alpha:
                n += 1
                hits += int(per_alpha[key])
    return hits / n if n else None


def digest_flexgen(results: dict[str, Any]) -> str:
    alphas = results["alphas"]
    lines = [
        "== X22 flexgen digest ==",
        f"band {results['band'][0]}..{results['band'][-1]} "
        f"({len(results['band'])} layers) | {results['count_note']}",
        "",
        "-- swap success (top-1 = the target's answer), by alpha --",
    ]
    for alpha in alphas:
        n_gated = sum(1 for cell in results["cells"] if cell["baseline_correct"])
        lines.append(
            f"  alpha={alpha}: success={_cell(arm_rate(results, alpha, gated=True))} "
            f"(baseline-correct cells n={n_gated}/{len(results['cells'])})  "
            f"all={_cell(arm_rate(results, alpha, gated=False))}"
        )
    corr = results["correlation"]
    lines += [
        "",
        f"-- workspace loading vs swap success (alpha={corr['alpha']}, "
        f"per-argument, n={corr['n_points']}) --",
        f"  pearson={_cell(corr['pearson'])}  spearman={_cell(corr['spearman'])}",
    ]
    for name, stats in corr["per_category"].items():
        lines.append(f"    {name:<9} pearson={_cell(stats['pearson'])} "
                     f"(n={stats['n_points']})")
    lines += ["", f"-- per-argument loading / success (alpha={alphas[0]}) --"]
    for entry in results["arg_stats"]:
        lines.append(
            f"  {entry['category']:<9} {entry['arg']:<9} "
            f"loading={entry['loading']:.3f}  "
            f"success={_cell(entry['success'][str(alphas[0])])} "
            f"(n={entry['n_trials']})"
        )
    return "\n".join(lines)


# ----------------------------------------------------------------- structure
_FUNCTION_WORDS = (
    "the of and in is was that a to it for on with as at by from or an be this which "
    "one had not but what all were when we there can been more if will would about "
    "their other new some could time these two may then do first any my now such "
    "like our over man me even most made after also did many before must through "
    "years where much your way well down should because each just those people how "
    "too little state good very make world still own see men work long get here "
    "between both life being under never day same another know while last might us "
    "great old year off come since against go came right used take three"
).split()

_CONTENT_WORDS = (
    "river mountain city window garden letter engine forest harbour melody candle "
    "harvest traveller classroom furnace column orchard blanket measure mirror "
    "passage pyramid quarter railway shadow temple valley voyage winter attorney "
    "bicycle cabinet diamond evening freedom gravity horizon island journal "
    "kingdom liberty machine network officer planet quarry region silence telescope "
    "umbrella village whistle yellow zebra animal bridge castle desert evening "
    "factory glacier history industry jungle kitchen lantern market north ocean "
    "palace question reason summer theatre university victory water west"
).split()


def synthetic_prompts(n: int, *, seed: int, words_per_prompt: int) -> list[str]:
    """Seeded pretraining-like sentences (CORPUS_NOTE fallback)."""
    rng = random.Random(seed)
    bank = _FUNCTION_WORDS + _CONTENT_WORDS
    prompts: list[str] = []
    for _ in range(n):
        words: list[str] = []
        while len(words) < words_per_prompt:
            length = rng.randint(8, 16)
            sentence = [rng.choice(bank) for _ in range(length)]
            sentence[0] = sentence[0].capitalize()
            words.extend(sentence)
            words.append(".")
        text = " ".join(words[:-1]) + "."
        prompts.append(text)
    return prompts


def load_corpus(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    """CORPUS_NOTE priority: text-file > wikitext > repo manifests > synthetic."""
    n = args.n_prompts

    def clean(prompts: list[str]) -> list[str]:
        return [p.replace("<image>", " ").strip()
                for p in prompts if p.replace("<image>", " ").strip()]

    if args.text_file:
        return clean(load_text_file_prompts(args.text_file, n,
                                            min_chars=args.min_chars)), {
            "source": "text-file", "path": args.text_file,
        }
    note = ""
    try:
        prompts = load_wikitext_prompts(n, min_chars=max(args.min_chars, 600))
        return prompts, {"source": "wikitext-103"}
    except Exception as exc:  # noqa: BLE001 - datasets not installed / no cache
        note = f"wikitext unavailable ({type(exc).__name__}: {exc})"
    for manifest in sorted(REPO.glob("runs/**/manifest*.jsonl")):
        try:
            prompts = clean(load_text_file_prompts(manifest, n,
                                                   min_chars=args.min_chars))
        except Exception:  # noqa: BLE001 - try the next corpus source
            continue
        if len(prompts) >= min(n, 4):  # a real corpus, not a 3-sample dry run
            return prompts[:n], {"source": "repo-manifest", "path": str(manifest)}
    words_per_prompt = max(24, int(args.positions_per_prompt * 1.3))
    return synthetic_prompts(n, seed=args.seed, words_per_prompt=words_per_prompt), {
        "source": "synthetic", "note": note, "words_per_prompt": words_per_prompt,
    }


@torch.no_grad()
def run_structure(args: argparse.Namespace, model: LlavaLensModel, lens: JacobianLens,
                  band: list[int]) -> dict[str, Any]:
    started = time.time()
    prompts, corpus_meta = load_corpus(args)
    readout_layers = sorted({l for l in lens.source_layers if 0 <= l < model.n_layers}
                            | {model.n_layers - 1})
    print(f"-- structure: {len(prompts)} prompts ({corpus_meta['source']}), "
          f"readout layers {readout_layers[0]}..{readout_layers[-1]}, "
          f"positions<={args.positions_per_prompt} --")

    ks = (1, 5, 25)
    max_k = max(ks)
    top1_by_layer: dict[int, list[list[int]]] = {layer: [] for layer in readout_layers}
    per_layer_stats: dict[int, dict[str, list[float]]] = {
        layer: {"acc1": [], "acc5": [], "acc25": [],
                "kurt_maxlogit": [], "kurt_all": []}
        for layer in readout_layers
    }
    model_top1_hits = 0
    model_positions = 0

    for index, prompt in enumerate(prompts):
        batch = as_batch(model, prompt, args.max_seq_len)
        ids = batch.input_ids[0].tolist()
        n_pos = min(batch.seq_len - 1, args.positions_per_prompt)
        if n_pos < 1:
            continue
        truth = torch.tensor(ids[1:n_pos + 1], dtype=torch.long)
        residuals = recorded_forward(model, batch, readout_layers)
        for layer in readout_layers:
            hidden = residuals[layer][:n_pos]
            logits = model.unembed(lens.transport(hidden, layer)).float().cpu()
            top = torch.topk(logits, k=min(max_k, logits.shape[-1]), dim=-1).indices
            hits = [(truth.unsqueeze(-1) == top[:, :k]).any(-1).float().mean().item()
                    for k in ks]
            stats = per_layer_stats[layer]
            stats["acc1"].append(hits[0])
            stats["acc5"].append(hits[1])
            stats["acc25"].append(hits[2])
            stats["kurt_maxlogit"].append(excess_kurtosis(logits.max(dim=-1).values))
            stats["kurt_all"].append(excess_kurtosis(logits))
            top1_by_layer[layer].append(top[:, 0].tolist())
            if layer == model.n_layers - 1:
                model_top1_hits += int((truth.unsqueeze(-1) == top[:, :1]).any(-1).sum())
                model_positions += n_pos
        if (index + 1) % 20 == 0 or index + 1 == len(prompts):
            print(f"  [forward {index + 1}/{len(prompts)}]")

    model_top1_acc = model_top1_hits / model_positions if model_positions else None

    # (c) top-1 autocorrelation vs a position-shuffled null (per layer, lags 1..8)
    generator = torch.Generator().manual_seed(args.seed)
    n_perms = 5
    autocorr: dict[int, dict[str, Any]] = {}
    for layer in readout_layers:
        observed = {delta: [] for delta in range(1, 9)}
        null = {delta: [] for delta in range(1, 9)}
        for seq in top1_by_layer[layer]:
            for delta in range(1, 9):
                rate = _match_rate(seq, delta)
                if rate is not None:
                    observed[delta].append(rate)
            for _ in range(n_perms):
                perm = torch.randperm(len(seq), generator=generator).tolist()
                shuffled = [seq[i] for i in perm]
                for delta in range(1, 9):
                    rate = _match_rate(shuffled, delta)
                    if rate is not None:
                        null[delta].append(rate)
        autocorr[layer] = {
            "observed": {str(d): (sum(v) / len(v) if v else None)
                         for d, v in observed.items()},
            "null": {str(d): (sum(v) / len(v) if v else None)
                     for d, v in null.items()},
        }

    # (d) effective dimensionality of the J-lens vector matrix W_U J_l
    effdim = _effective_dimensionality(args, model, lens, readout_layers)

    layer_rows: dict[str, dict[str, Any]] = {}
    for layer in readout_layers:
        stats = per_layer_stats[layer]
        n = len(stats["acc1"])
        mean = lambda key: (sum(stats[key]) / n if n else None)  # noqa: E731
        row: dict[str, Any] = {
            "n_positions": n,
            "top1_acc": mean("acc1"), "top5_acc": mean("acc5"),
            "top25_acc": mean("acc25"),
            "kurtosis_maxlogit": mean("kurt_maxlogit"),
            "kurtosis_all": mean("kurt_all"),
            "autocorr": autocorr[layer],
        }
        if layer in effdim:
            row["effdim90"] = effdim[layer]["centered"]
            row["effdim90_uncentered"] = effdim[layer]["uncentered"]
        layer_rows[str(layer)] = row

    in_band = [int(l) for l in layer_rows if int(l) in set(band)]
    digest = digest_structure(layer_rows, band, in_band, model_top1_acc, corpus_meta)
    print("\n" + digest)
    results = {
        "layer_rows": layer_rows,
        "model_top1_acc": model_top1_acc,
        "n_prompts": len(prompts),
        "positions_per_prompt": args.positions_per_prompt,
        "band": band,
        "readout_layers": readout_layers,
        "corpus": corpus_meta,
        "pca_sample_rows": args.pca_sample_rows,
    }
    best_top1_layer = max(layer_rows, key=lambda key: layer_rows[key]["top1_acc"] or -1.0)
    summary = (
        f"structure: corpus={corpus_meta['source']} n_prompts={len(prompts)}; "
        f"model top-1 vs text={_cell(model_top1_acc)}; best lens top-1 layer="
        f"{best_top1_layer} ({_cell(layer_rows[best_top1_layer]['top1_acc'])}); "
        f"onset/offset per metric in the digest"
    )
    return {
        "results": results,
        "digest": digest,
        "summary": summary,
        "runtime_s": round(time.time() - started, 1),
    }


def _match_rate(seq: list[int], delta: int) -> float | None:
    if len(seq) <= delta:
        return None
    a, b = seq[:-delta], seq[delta:]
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def _effective_dimensionality(
    args: argparse.Namespace, model: LlavaLensModel, lens: JacobianLens,
    layers: list[int],
) -> dict[int, dict[str, float]]:
    """Fraction of d_model PCA dims capturing 90% of the variance of ``W_U J_l``.

    Exact mode: with ``M = W_U J_l`` ([vocab, d]), the centered covariance is
    ``M_c^T M_c = J^T (W_U^T W_U) J - V * mu mu^T`` with ``mu = mean_row(M) =
    (W_U.mean(0)) J`` — never materializing ``M``. Sampled mode SVDs
    ``--pca-sample-rows`` random rows instead (cheap CPU sanity mode).
    """
    device = model.unembed_weight().device
    weight = model.unembed_weight().detach().float()
    d_model = weight.shape[1]
    out: dict[int, dict[str, float]] = {}
    gram = None
    mean_row = None
    if args.pca_sample_rows <= 0:
        gram = weight.T @ weight  # [d, d], shared across layers
        mean_row = weight.mean(dim=0)  # [d]
    generator = torch.Generator().manual_seed(args.seed)
    for layer in layers:
        J = lens.jacobians[layer].to(device).float()
        if args.pca_sample_rows <= 0:
            mtm = J.T @ gram @ J
            mu = mean_row @ J
            centered = mtm - weight.shape[0] * torch.outer(mu, mu)
            uncentered = mtm
            evals_c = torch.linalg.eigvalsh(0.5 * (centered + centered.T))
            evals_u = torch.linalg.eigvalsh(0.5 * (uncentered + uncentered.T))
        else:
            rows = torch.randperm(weight.shape[0], generator=generator)[: min(args.pca_sample_rows, weight.shape[0])]
            m = weight[rows] @ J  # [rows, d]
            m = m - m.mean(dim=0, keepdim=True)
            evals_c = torch.linalg.svdvals(m) ** 2
            evals_u = evals_c  # sampled mode reports the centered spectrum only
        out[layer] = {
            "centered": _dims_for_variance(evals_c, d_model),
            "uncentered": _dims_for_variance(evals_u, d_model),
        }
    return out


def _dims_for_variance(eigvals: torch.Tensor, d_model: int, target: float = 0.9) -> float:
    e, _ = torch.sort(eigvals, descending=True)
    e = e.clamp_min(0)
    total = e.sum()
    if total <= 0:
        return float("nan")
    cumulative = torch.cumsum(e / total, dim=0)
    needed = int((cumulative < target).sum().item()) + 1
    return needed / d_model


def _half_rise(values: dict[int, float]) -> dict[str, float | None]:
    """Onset/offset by half-rise: first/last layer with value >= min + 0.5*range."""
    if not values:
        return {"onset": None, "offset": None, "argmax": None}
    xs = sorted(values)
    ys = [values[x] for x in xs]
    lo, hi = min(ys), max(ys)
    threshold = lo + 0.5 * (hi - lo)
    above = [x for x, y in zip(xs, ys) if y >= threshold]
    argmax = xs[ys.index(max(ys))]
    return {
        "onset": above[0] if above else None,
        "offset": above[-1] if above else None,
        "argmax": argmax,
        "min": lo, "max": hi,
    }


def digest_structure(layer_rows: dict[str, dict[str, Any]], band: list[int],
                     in_band: list[int], model_top1_acc: float | None,
                     corpus_meta: dict[str, Any]) -> str:
    lines = [
        "== X22 structure digest ==",
        f"corpus: {corpus_meta['source']}"
        + (f" ({corpus_meta.get('path', corpus_meta.get('note', ''))})" if len(corpus_meta) > 1 else "")
        + f" | band {band[0]}..{band[-1]} | model top-1 vs text = {_cell(model_top1_acc)}",
        "layer |  top1   top5  top25 | kurtMax  kurtAll |  ac1obs  ac1null     d1 | effdim90",
    ]
    series: dict[str, dict[int, float]] = {
        "top1": {}, "kurt": {}, "autocorr_delta": {}, "effdim": {},
    }
    for layer in sorted(layer_rows, key=int):
        row = layer_rows[layer]
        ac = row["autocorr"]["observed"].get("1")
        null = row["autocorr"]["null"].get("1")
        d1 = (ac - null) if (ac is not None and null is not None) else None
        eff = row.get("effdim90")
        lines.append(
            f"{layer:>5} | {_cell(row['top1_acc'], 7, 3)} {_cell(row['top5_acc'], 7, 3)} "
            f"{_cell(row['top25_acc'], 7, 3)} | {_cell(row['kurtosis_maxlogit'], 8, 2)} "
            f"{_cell(row['kurtosis_all'], 8, 2)} | {_cell(ac, 8, 3)} {_cell(null, 8, 3)} "
            f"{_cell(d1, 7, 3)} | {_cell(eff, 8, 3)}"
        )
        series["top1"][int(layer)] = row["top1_acc"]
        series["kurt"][int(layer)] = row["kurtosis_maxlogit"]
        series["autocorr_delta"][int(layer)] = d1 if d1 is not None else float("nan")
        if eff is not None and not math.isnan(eff):
            series["effdim"][int(layer)] = eff
    lines += ["", "-- half-rise onset/offset per metric (in-band layers only) --"]
    in_band_rows = {
        metric: {layer: value for layer, value in values.items() if layer in set(in_band)}
        for metric, values in series.items()
    }
    for metric, values in in_band_rows.items():
        values = {k: v for k, v in values.items() if not math.isnan(v)}
        marks = _half_rise(values)
        lines.append(
            f"  {metric:<16} onset={marks['onset']} offset={marks['offset']} "
            f"argmax={marks['argmax']} (min={marks.get('min'):.3f} max={marks.get('max'):.3f})"
            if marks["onset"] is not None else f"  {metric:<16} (no in-band values)"
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------- main
def main() -> int:
    args = parse_args()
    if args.arm in ("multihop", "flexgen") and args.n_items < 0:
        raise ValueError("--n-items must be >= 0 (0 = all)")
    if args.arm == "structure" and args.n_prompts < 1:
        raise ValueError("--n-prompts must be >= 1")
    if args.arm == "structure" and args.positions_per_prompt < 2:
        raise ValueError("--positions-per-prompt must be >= 2")

    model = load_model(args)
    lens, lens_meta = build_lens(args, model)
    band_lo, band_hi = (int(part) for part in args.band.split(","))
    band, band_origin = resolve_band(lens, model, band_lo, band_hi)
    device = model.unembed_weight().device
    print(f"== X22 arm={args.arm} backend={args.backend} device={device} "
          f"layers={model.n_layers} d_model={model.d_model} ==")
    print(f"lens: {lens_meta.get('source')} (loaded={lens_meta.get('loaded')}); "
          f"band={band} [{band_origin}]")
    print(RAW_LENS_NOTE)

    if args.arm == "multihop":
        payload = run_multihop(args, model, lens, band)
        notes = {"raw_lens": RAW_LENS_NOTE, "token_form": TOKEN_FORM_NOTE,
                 "scoring": SCORING_NOTE, "depth": DEPTH_NOTE, "onset": ONSET_NOTE}
    elif args.arm == "flexgen":
        payload = run_flexgen(args, model, lens, band)
        notes = {"raw_lens": RAW_LENS_NOTE, "token_form": TOKEN_FORM_NOTE,
                 "scoring": SCORING_NOTE, "loading": LOADING_NOTE,
                 "count": FLEXGEN_COUNT_NOTE}
    else:
        payload = run_structure(args, model, lens, band)
        notes = {"raw_lens": RAW_LENS_NOTE, "corpus": CORPUS_NOTE}

    report = {
        "script": Path(__file__).name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "arm": args.arm,
        "backend": args.backend,
        "device": str(device),
        "n_layers": model.n_layers,
        "d_model": model.d_model,
        "lens": lens_meta,
        "band": band,
        "band_origin": band_origin,
        "config": {
            key: value for key, value in vars(args).items()
        },
        "notes": notes,
        "results": payload["results"],
        "digest": payload["digest"],
        "summary": payload.get("summary", payload["digest"]),
        "runtime_s": payload["runtime_s"],
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out} (runtime {payload['runtime_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
