#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""X21: the workspace paper's three protocols — verbal report, injected-thought
introspection, directed modulation — replicated on the LLaVA J-lens stack.

Paper: Gurnee et al. 2026, "workspace" (transformer-circuits.pub/2026/workspace); the
items/templates come from the vendored ``third_party/jacobian-lens/data/experiments/``
JSONs. The model is ``LlavaLensModel`` on text-only prompts (the LLM is a 32-layer
Vicuna fine-tune; workspace band = layers ~[12, 29], 0-indexed). Three arms:

``--arm report`` (paper verbal-report, parts a+b)
    Prompt ``Think of a {category}, and name it in a single word:`` (14 categories x
    10 candidates from ``verbal-report.json``). (a) CORRELATION: one greedy trace
    generation per category; its prefill record (``step_records[0]``, the COLON
    position that predicts generated token 0) is lens-read-out at every workspace-band
    layer and Spearman-correlated with the model's own next-token logit ranking of the
    10 candidates (mean of the 14 per-category correlations, plus the pooled one).
    (b) SWAP: the model's spontaneously chosen item (its first generated token) has
    its lens coordinates swapped with a same-category target not in the model's top-10
    outputs — ``V = [v_s v_t]`` (rows of ``W_U @ J_l``), ``c = pinv(V) h``, patched
    ``h' = h + V (sigma(c) - c)`` — applied at ALL token positions of ALL band layers
    during generation. Success = the answer flips to the target (first generated
    word); also recorded: the target reaching top-5 of the next-token distribution.
    (c) the J-space vs non-J-space decomposition arm (k=16) is SKIPPED: the vendored
    jlens exposes no sparse nonnegative gradient-pursuit decomposition (recorded as a
    deviation in the JSON).

``--arm introspect`` (paper verbal-introspection)
    The vendored intro prompt ("a thought may have been implanted...") + the vendored
    ``word`` prefill. ``v_t`` (the injected concept's J-lens vector) is injected
    (``add``) at EVERY token of the user turn(s), at every band layer, during the
    prefill; strengths sweep fractions of the span's mean residual norm (the x7 add
    convention ``alpha = k * ||h_l|| / ||v_t||``). Rate = trials whose completion
    mentions the concept (word-boundary match). Position control: the same injection
    over the intermediate assistant turn instead. A no-injection baseline completion
    is run once (the intro prompt is concept-independent).

``--arm modulation`` (paper directed-modulation)
    Three task families while copying a fixed unrelated sentence (the vendored carrier
    "The old painting hung crookedly on the wall."): (a) focus on a category instance
    (``topic_categories``), (b) mentally evaluate an arithmetic expression
    (``math_problems``), (c) silently count characters per line
    (``selectivity-linecount.json`` passages, greedily wrapped at the vendored width).
    Conditions: positive instruction (vendored ``focus`` phrasing) / negative
    instruction (vendored ``dismissal`` "ignore" phrasing) / no-instruction baseline.
    Measure: the target concept (category name+members / expression's answer / the
    first line's character count) in the J-lens top-10 at any (position, layer) during
    the copied region — rate per condition — and, for line-count, the absolute error
    of the top-1 numeric lens readout.

Readout: the raw J-lens readout ``unembed(J_l h)`` (the paper's), or the repo's
calibrated chain (``bias`` inside the fitted branch, then ``scale``/``temp``/
``logit_bias`` from ``bias-<mask>.pt``) with ``--calibrated``. Lens vectors are ALWAYS
the raw rows of ``W_U @ J_l`` (the paper's ``v_t``; the convention pinned by
``test_lens_vectors_are_readout_gradients``).

Backends: ``--backend hf-llava`` (default, CUDA, fitted lens from ``--lens-dir``) or
``--backend tiny`` (the tiny random-weight CPU fixture; the lens falls back to an
in-script identity over the model's own layers so the smoke runs without /data).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
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
from vlm_lens.interventions import ResidualEdit, apply_edit  # noqa: E402
from vlm_lens.models.llava import LlavaLensModel, MultimodalBatch  # noqa: E402

DEFAULT_LENS_DIR = "/data/anhnq/vlm-lens-out/validation/s2-merged/artifacts"
DEFAULT_BIAS_DIR = "/data/anhnq/vlm-lens-out/validation/step5e"
DEFAULT_EXPERIMENTS_DIR = REPO / "third_party" / "jacobian-lens" / "data" / "experiments"

WORKSPACE_LO = 12
WORKSPACE_HI = 29

#: The paper's verbal-report template, wrapped in the repo's chat convention.
REPORT_TEMPLATE = "Think of a {category}, and name it in a single word:"
CHAT_USER = "USER:"
CHAT_ASSISTANT = "\nASSISTANT:"

#: Modulation prompt bodies. The instruction slot carries the vendored phrasing
#: (focus/dismissal) with ``{x}`` filled, or "" for the no-instruction baseline.
MOD_BODY = 'Copy this sentence exactly:\n"{carrier}"\n{instruction}'
LC_BODY = (
    "Here is a passage:\n\n{passage}\n\n"
    "Copy the passage exactly, maintaining the same line-wrapping pattern. {instruction}"
)
COUNT_FOCUS = "While you copy, count the number of characters in each line."
COUNT_IGNORE = "While you copy, ignore the number of characters in each line."

COLON_NOTE = (
    "the chat-wrapped prompt ends 'ASSISTANT:', so the colon token immediately before "
    "the answer is the LAST PROMPT position; with KV caching its prefill record is "
    "step_records[0] (x14's handoff convention) and generated.scores[0] is the model's "
    "next-token distribution there"
)
CANDIDATE_NOTE = (
    "candidates are the vendored per-category lists, truncated to --n-candidates (the "
    "paper's 10); a candidate's readout token is its leading-space form's LAST token id "
    "(x13's 'token strings keep the caption's leading space' + the edit path's "
    "_token_id convention), the token that would be emitted at that position"
)
RANK_RULE = "1-based rank, ties count as better ranks (s2_eval._rank_of_row semantics)"
SWAP_NOTE = (
    "the paper's coordinate swap: V=[v_s v_t] with v = rows of W_U @ J_l (the raw "
    "J-lens vectors), c = pinv(V) @ h, h' = h + V @ (sigma(c) - c), sigma swapping the "
    "two coordinates; applied at ALL token positions of ALL workspace-band layers "
    "simultaneously (prefill rows AND every decode step) by reusing "
    "interventions.apply_edit's 'swap' mode per band layer"
)
SOURCE_NOTE = (
    "the source s is the model's spontaneously chosen item = its FIRST GENERATED TOKEN "
    "(the answer word's first emitted subword); its id cannot round-trip through the "
    "tokenizer as a string, so vectors are resolved by token id in this script "
    "(vectors_for_ids = interventions.lens_vectors' exact math keyed by id)"
)
TARGET_NOTE = (
    "targets are same-category candidates whose surface is not the model's answer and "
    "whose token is not among the model's top-10 next-token outputs at the colon "
    "(top-10 ids or their decoded surfaces), first --n-swaps in vendored order"
)
SUCCESS_NOTE = (
    "success = the first generated word equals the target (case-insensitive; multiword "
    "targets count when the completion starts with them); target_in_top5 = the target's "
    "first subword id ranks <= 5 (RANK_RULE) in the swapped run's step-0 distribution"
)
DECOMPOSITION_NOTE = (
    "protocol part (c), the J-space vs non-J-space decomposition arm (k=16), is SKIPPED: "
    "the vendored jlens exposes no sparse nonnegative gradient-pursuit decomposition "
    "(jlens/lens.py provides transport/apply only) - recorded as a deviation"
)
TEMPLATE_NOTE = (
    "introspection/modulation prompts wrap the vendored contents in the repo's chat "
    "convention (USER:/ASSISTANT:, x13's PROMPT_TEMPLATE family); assistant turns are "
    "separated by the eos token (the literal '</s>' piece becomes "
    "tokenizer.eos_token_id when the tokenizer has one); vendored content strings are "
    "used verbatim"
)
STRENGTH_NOTE = (
    "injection strength k scales the raw vector to a fraction of the edited span's mean "
    "residual norm per layer: alpha_l = k * mean(||h_l|| over the span) / ||v_t,l|| "
    "(the x7 add convention, so 'add' alphas are comparable across layers/concepts)"
)
SPAN_NOTE = (
    "the intro prompt is assembled from the vendored turns; user-turn spans (both user "
    "messages) are the injection positions, the intermediate assistant turn (' "
    "Understood.') is the position control; segments are tokenized per piece and "
    "concatenated so spans are exact by construction (BOS prepended once, matching the "
    "processor path's add_bos_token)"
)
MATCH_NOTE = (
    "concept-reported = word-boundary regex match of the concept's surface form in the "
    "generated completion (the fixed prefill is part of the prompt, not the response)"
)
NUMERIC_NOTE = (
    "numeric readout: the top-1 token among single-token leading-space forms of "
    "0..--numeric-max (first value wins on id collisions); error = |value - first "
    "line's actual character count|"
)
WRAP_NOTE = (
    "line-count passages are re-wrapped greedily at the vendored width (greedy word "
    "wrap, single spaces); the numeric ground truth is the FIRST line's actual length"
)
BAND_NOTE = (
    "workspace band = layers [layer-lo, layer-hi] (0-indexed) intersected with the "
    "lens's fitted layers; if the intersection is empty (tiny fixture) ALL fitted "
    "layers are used and noted"
)

ARM_DEFAULT_NEW_TOKENS = {"report": 6, "introspect": 8, "modulation": 64}


# ------------------------------------------------------------------------ args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", required=True, choices=("report", "introspect", "modulation"),
        help="which paper protocol to run",
    )
    parser.add_argument(
        "--backend", choices=("hf-llava", "tiny"), default="hf-llava",
        help="model backend: 'hf-llava' (default, CUDA) or the tiny CPU smoke fixture",
    )
    parser.add_argument(
        "--lens-dir", default=DEFAULT_LENS_DIR,
        help="fitted J-lens dir (lens-<mask>.pt); tiny builds an in-script identity "
             "lens over the model's own layers instead",
    )
    parser.add_argument("--mask", default="text", help="which lens-<mask>.pt / bias-<mask>.pt to use")
    parser.add_argument(
        "--bias-dir", default=DEFAULT_BIAS_DIR,
        help="directory holding bias-<mask>.pt, used only with --calibrated",
    )
    parser.add_argument(
        "--calibrated", action="store_true",
        help="apply the repo's calibrated readout chain (bias/scale/temp/logit_bias) on "
             "top of the J-lens transport; default is the paper's raw J-lens readout",
    )
    parser.add_argument(
        "--experiments-dir", default=str(DEFAULT_EXPERIMENTS_DIR),
        help="vendored paper experiment JSONs",
    )
    parser.add_argument(
        "--layers", default=None,
        help="comma-separated band layers override (default: the workspace band)",
    )
    parser.add_argument("--layer-lo", type=int, default=WORKSPACE_LO, help="workspace band low (0-indexed)")
    parser.add_argument("--layer-hi", type=int, default=WORKSPACE_HI, help="workspace band high (inclusive)")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="generation cap (per-arm default: "
                             + ", ".join(f"{arm}={n}" for arm, n in ARM_DEFAULT_NEW_TOKENS.items()) + ")")
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--json", required=True)
    # report arm
    parser.add_argument("--n-candidates", type=int, default=10,
                        help="verbal-report candidates per category (the paper's 10)")
    parser.add_argument("--n-swaps", type=int, default=3,
                        help="swap targets per category (0 disables the swap arm)")
    parser.add_argument("--swap-alpha", type=float, default=1.0,
                        help="scale on the swap delta (1.0 = the paper's exact swap)")
    # introspect arm
    parser.add_argument("--n-concepts", type=int, default=20,
                        help="injected concepts (first N of the vendored list)")
    parser.add_argument("--strengths", default="0.1,0.25,0.5,1.0",
                        help="comma-separated injection strengths (fractions of the span "
                             "residual norm, see STRENGTH_NOTE)")
    parser.add_argument("--prefill-key", default="word", choices=("word", "default"),
                        help="which vendored prefill to append after the final ASSISTANT:")
    # modulation arm
    parser.add_argument("--n-items-per-family", type=int, default=5,
                        help="items per task family (concept/math/line-count)")
    parser.add_argument("--phrasing-index", type=int, default=0,
                        help="which vendored phrasing of each group to use (0 = first)")
    parser.add_argument("--numeric-max", type=int, default=99,
                        help="largest candidate value for the numeric readout")
    return parser.parse_args()


# -------------------------------------------------------------------- plumbing


def load_model(args: argparse.Namespace) -> LlavaLensModel:
    """The protocol model: the HF checkpoint (default, CUDA) or the tiny CPU fixture."""
    if args.backend == "tiny":
        from vlm_lens.models.tiny_llava import TinyLlavaConfig, build_tiny_llava

        hf_model, processor = build_tiny_llava(TinyLlavaConfig())
        return LlavaLensModel(hf_model, processor)
    return LlavaLensModel.from_pretrained(
        dtype=torch.bfloat16, device="cuda", local_files_only=True
    )


def load_lens(args: argparse.Namespace, model: LlavaLensModel) -> tuple[JacobianLens, dict[str, Any]]:
    """The fitted J-lens on the model's device; the tiny fixture gets an identity lens.

    The paper's protocol reads out (and swaps coordinates in) the *J-lens* basis, so
    the fitted ``J_l`` matrices from ``--lens-dir`` are used directly - unlike x14/x15,
    whose calibrated probe deliberately replaces the transport with the identity. With
    no fitted lens available (tiny smoke), the identity transport over the model's own
    layers keeps every code path live; lens vectors then reduce to the unembed rows.
    Matrices are re-homed once on the model device so the per-step readouts do not pay
    a host->device copy per call (x15's identity-lens reasoning).
    """
    device = model.unembed_weight().device
    if args.backend == "tiny":
        lens = JacobianLens(
            jacobians={
                layer: torch.eye(model.d_model, dtype=torch.float32, device=device)
                for layer in range(model.n_layers)
            },
            n_prompts=0,
            d_model=model.d_model,
        )
        meta: dict[str, Any] = {
            "loaded": False,
            "note": "identity lens over the model's own layers (tiny backend)",
        }
    else:
        lenses, _ = load_lens_set(args.lens_dir)
        if args.mask not in lenses:
            raise ValueError(f"mask {args.mask!r} not in {args.lens_dir} (found {sorted(lenses)})")
        source = lenses[args.mask]
        if not source.source_layers:
            raise ValueError(f"lens {args.mask!r} in {args.lens_dir} has no fitted layers")
        if source.d_model != model.d_model:
            raise ValueError(f"lens d_model {source.d_model} != model d_model {model.d_model}")
        lens = JacobianLens(
            jacobians={layer: J.to(device) for layer, J in source.jacobians.items()},
            n_prompts=source.n_prompts,
            d_model=source.d_model,
        )
        meta = {
            "loaded": True,
            "dir": str(args.lens_dir),
            "mask": args.mask,
            "n_prompts": int(source.n_prompts),
        }
    return lens, meta


@dataclass(frozen=True)
class Calibration:
    """The bias-<mask>.pt payload as x14's four dicts (``--calibrated`` readouts only)."""

    bias: dict[int, torch.Tensor]
    scale: dict[int, float]
    temp: dict[int, float]
    logit_bias: dict[int, torch.Tensor]


def load_calibration(args: argparse.Namespace, model: LlavaLensModel) -> Calibration | None:
    """``--calibrated`` payload, or None for the paper's raw J-lens readout."""
    if not args.calibrated:
        return None
    payload, _meta = load_bias(Path(args.bias_dir) / f"bias-{args.mask}.pt")
    device = model.unembed_weight().device
    return Calibration(
        bias={int(layer): entry["bias"].to(device) for layer, entry in payload.items()},
        scale={int(layer): float(entry["scale"]) for layer, entry in payload.items()},
        temp={
            int(layer): float(entry["temp"])
            for layer, entry in payload.items()
            if "temp" in entry
        },
        logit_bias={
            int(layer): entry["logit_bias"].to(device)
            for layer, entry in payload.items()
            if "logit_bias" in entry
        },
    )


def resolve_layers(args: argparse.Namespace, lens: JacobianLens) -> tuple[list[int], str]:
    """The workspace-band layers the lens actually fitted (see BAND_NOTE)."""
    if args.layers:
        requested = [int(part) for part in args.layers.split(",")]
    else:
        requested = list(range(args.layer_lo, args.layer_hi + 1))
    fitted = set(lens.source_layers)
    resolved = sorted(layer for layer in requested if layer in fitted)
    note = BAND_NOTE
    if not resolved:
        resolved = list(lens.source_layers)
        note = BAND_NOTE + " [band empty after clamping; using ALL fitted layers]"
    return resolved, note


def _encode_no_special(model: LlavaLensModel, text: str) -> list[int]:
    """Token ids for ``text`` with no specials (the tiny fixture always adds BOS)."""
    ids = model.tokenizer(text, add_special_tokens=False)["input_ids"]
    if isinstance(ids, torch.Tensor):
        ids = ids.reshape(-1).tolist()
    ids = list(ids)
    bos = getattr(model.tokenizer, "bos_token_id", None)
    if bos is not None and ids and int(ids[0]) == int(bos):
        ids = ids[1:]
    return [int(i) for i in ids]


def token_last_id(model: LlavaLensModel, token: str) -> int:
    """The id of the token that would be emitted for ``token`` (edit-path convention)."""
    ids = _encode_no_special(model, token)
    if not ids:
        raise ValueError(f"token string {token!r} tokenizes to nothing")
    return ids[-1]


def vectors_for_ids(
    model: LlavaLensModel,
    lens: JacobianLens,
    token_ids: Sequence[int],
    layers: Sequence[int],
) -> dict[int, torch.Tensor]:
    """J-lens vectors ``v_t = (W_U J_l)[t]`` keyed by token id, on the model device.

    ``interventions.lens_vectors``' exact math (rows of ``W_U @ J_l``; the orientation
    pinned by ``test_lens_vectors_are_readout_gradients``) keyed by token id instead of
    string, so a model-chosen token never has to round-trip through the tokenizer.
    """
    weight = model.unembed_weight().detach().float()  # [vocab, d] on the model device
    ids = torch.as_tensor(list(token_ids), dtype=torch.long, device=weight.device)
    out: dict[int, torch.Tensor] = {}
    for layer in layers:
        J = lens.jacobians[layer].to(weight.device).float()
        vectors = weight[ids] @ J  # [n_ids, d]
        out[layer] = vectors
    return out


def lens_logits(
    model: LlavaLensModel,
    lens: JacobianLens,
    hidden: torch.Tensor,
    layer: int,
    calib: Calibration | None,
) -> torch.Tensor:
    """Per-layer J-lens readout for one residual (x14's ``_calibrated_lens_logits``).

    transport (+ bias inside the fitted branch) -> scale -> unembed -> temp ->
    logit_bias; raw (no corrections) unless ``--calibrated``. ``hidden`` may carry any
    number of leading rows; the result is ``[..., vocab]`` fp32.
    """
    residual = hidden
    if layer in lens.jacobians:
        residual = lens.transport(residual, layer)
        if calib is not None and layer in calib.bias:
            residual = residual + calib.bias[layer].to(residual.device)
    if calib is not None and layer in calib.scale:
        residual = residual * calib.scale[layer]
    logits = model.unembed(residual).float()
    if calib is not None and layer in calib.temp:
        logits = logits / calib.temp[layer]
    if calib is not None and layer in calib.logit_bias:
        logits = logits + calib.logit_bias[layer].to(logits.device)
    return logits


def rank_of_id(logits: torch.Tensor, token_id: int) -> int:
    """1-based rank of ``token_id`` in a logit row (RANK_RULE: ties count as better)."""
    return int((logits > logits[token_id]).sum().item()) + 1


def rankdata(values: Sequence[float]) -> list[float]:
    """1-based average ranks (ties share the mean rank) - Spearman's ranking."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Tie-aware Spearman rho; None when either side is constant (rho undefined)."""
    if len(x) != len(y) or len(x) < 2:
        return None
    rx, ry = rankdata(x), rankdata(y)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    var_x = sum((a - mx) ** 2 for a in rx)
    var_y = sum((b - my) ** 2 for b in ry)
    denom = (var_x * var_y) ** 0.5
    if denom <= 0:
        return None
    return cov / denom


def mean_or_none(values: Sequence[float | None]) -> float | None:
    """Mean of the non-None entries, or None when there are none."""
    present = [float(v) for v in values if v is not None]
    if not present:
        return None
    return round(sum(present) / len(present), 5)


@torch.no_grad()
def trace_generation(
    model: LlavaLensModel,
    batch: MultimodalBatch,
    layers: Sequence[int],
    *,
    max_new_tokens: int,
) -> tuple[torch.Tensor, list[dict[int, torch.Tensor]], list[torch.Tensor]]:
    """Greedy generation caching the RAW block-output residual per decoding position.

    x14/x15's hook machinery and step convention: with KV caching the prefill forward
    predicts token 0 at the last prompt position (its record is ``step_records[0]`` —
    the colon position for this script's prompts) and each decode forward predicts one
    more token (hook call i predicts generated token i). Also returns
    ``generated.scores``: raw lm-head logits per step (the model-side distribution).
    """
    record_at = sorted(set(int(layer) for layer in layers))
    first_layer = record_at[0]
    step_records: list[dict[int, torch.Tensor]] = []

    def make_hook(layer: int) -> Callable[..., Any]:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = output if torch.is_tensor(output) else output[0]
            # Blocks fire in call order within one forward pass, so the first recorded
            # layer opening a new record is exactly where a new step starts.
            if layer == first_layer:
                step_records.append({})
            step_records[-1][layer] = tensor[0, -1].detach().float()
        return hook

    handles = [
        model.layers[layer].register_forward_hook(make_hook(layer)) for layer in record_at
    ]
    try:
        generated = model.hf_model.generate(
            **batch.hf_kwargs(),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            output_scores=True,
            return_dict_in_generate=True,
        )
    finally:
        for handle in handles:
            handle.remove()

    new_tokens = generated.sequences[0, batch.seq_len :].detach().cpu()
    scores = [step[0].detach().float() for step in generated.scores]
    if len(step_records) != len(new_tokens) or len(scores) != len(new_tokens):
        raise RuntimeError(
            f"hook recorded {len(step_records)} forwards / {len(scores)} score rows but "
            f"{len(new_tokens)} tokens were generated; the step alignment convention "
            "does not hold for this generate() configuration"
        )
    return new_tokens, step_records, scores


@contextmanager
def hooked(model: LlavaLensModel, handles: list[Any]):
    """Install forward hooks for the duration of the block (x16's handle pattern)."""
    for handle in handles:
        handle.install()
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


class _LayerHook:
    """One forward hook on ``model.layers[layer]`` rewriting the block output."""

    def __init__(self, model: LlavaLensModel, layer: int, fn: Callable[[torch.Tensor], torch.Tensor]):
        self._model = model
        self._layer = layer
        self._fn = fn
        self._handle: Any = None

    def install(self) -> None:
        if self._handle is not None:
            raise RuntimeError(f"layer-{self._layer} hook installed twice")
        self._handle = self._model.layers[self._layer].register_forward_hook(self._call)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _call(self, _module: Any, _inputs: Any, output: Any) -> Any:
        tensor = output if torch.is_tensor(output) else output[0]
        patched = self._fn(tensor)
        if patched is tensor:
            return output
        if torch.is_tensor(output):
            return patched
        return (patched, *output[1:])


class SwapPatcher:
    """All-positions lens-coordinate swap at every spec layer (the paper's protocol 1b).

    Every forward pass of ``generate`` (the full prefill AND each single-token decode
    step) gets its block output rewritten by ``interventions.apply_edit``'s ``swap``
    mode: ``basis = [v_s v_t]``, ``coords = pinv(basis) h``, ``h' = h + alpha *
    basis @ flip(coords) - coords``. The vectors are the raw rows of ``W_U @ J_l``
    resolved by token id (SOURCE_NOTE); the ResidualEdit specs below carry placeholder
    token strings that are never resolved - apply_edit only reads ``mode``/``alpha``.
    """

    def __init__(
        self,
        model: LlavaLensModel,
        layer_vectors: dict[int, dict[str, torch.Tensor]],
        alpha: float,
    ) -> None:
        self._model = model
        self._layer_vectors = layer_vectors
        self._alpha = float(alpha)
        self.fires: dict[int, int] = {}

    def _make_fn(self, layer: int) -> Callable[[torch.Tensor], torch.Tensor]:
        edit = ResidualEdit(
            layer=layer, mode="swap", alpha=self._alpha, token="<target>", source_token="<source>"
        )
        vectors = self._layer_vectors[layer]

        def fn(tensor: torch.Tensor) -> torch.Tensor:
            self.fires[layer] = self.fires.get(layer, 0) + 1
            return apply_edit(tensor, edit, vectors)

        return fn

    def hooks(self) -> list[_LayerHook]:
        return [
            _LayerHook(self._model, layer, self._make_fn(layer)) for layer in sorted(self._layer_vectors)
        ]

    def assert_fired(self) -> None:
        """Every swapped layer must have fired, and on the same number of forwards."""
        counts = {layer: self.fires.get(layer, 0) for layer in sorted(self._layer_vectors)}
        if any(count < 1 for count in counts.values()):
            raise RuntimeError(f"swap hooks did not all fire: {counts}")
        if len(set(counts.values())) > 1:
            raise RuntimeError(f"swap hooks fired an unequal number of times: {counts}")


class InjectPatcher:
    """Prefill-scoped ``add`` injection of a J-lens vector at chosen prompt positions.

    The intro prompt's concept vector is added to the user-turn (or control assistant
    turn) rows of the prefill forward only - the decode steps read the injected context
    through attention over the cached prompt. Strengths are per-layer
    ``alpha_l = k * ||h_l||_span / ||v_l||`` (STRENGTH_NOTE), precomputed per trial.
    """

    def __init__(
        self,
        model: LlavaLensModel,
        prompt_len: int,
        layer_specs: dict[int, tuple[torch.Tensor, float, torch.Tensor]],
    ) -> None:
        self._model = model
        self._prompt_len = int(prompt_len)
        self._layer_specs = layer_specs  # layer -> (vector[d] fp32, alpha, mask[seq] bool)
        self.prefill_fires: dict[int, int] = {}

    def _make_fn(self, layer: int) -> Callable[[torch.Tensor], torch.Tensor]:
        vector, alpha, mask = self._layer_specs[layer]
        edit = ResidualEdit(layer=layer, mode="add", alpha=float(alpha), token="<concept>")

        def fn(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.shape[1] != self._prompt_len:
                return tensor  # a decode step: the prompt rows are cached, nothing to do
            self.prefill_fires[layer] = self.prefill_fires.get(layer, 0) + 1
            rows = apply_edit(tensor[0, mask].unsqueeze(0), edit, {"target": vector})
            patched = tensor.clone()
            patched[0, mask] = rows[0]
            return patched

        return fn

    def hooks(self) -> list[_LayerHook]:
        return [
            _LayerHook(self._model, layer, self._make_fn(layer)) for layer in sorted(self._layer_specs)
        ]

    def assert_fired(self) -> None:
        """Every injected layer must have patched exactly one prefill forward."""
        counts = {layer: self.prefill_fires.get(layer, 0) for layer in sorted(self._layer_specs)}
        if any(count != 1 for count in counts.values()):
            raise RuntimeError(f"expected exactly one prefill fire per injected layer, got {counts}")


def greedy_generate(
    model: LlavaLensModel,
    batch: MultimodalBatch,
    *,
    max_new_tokens: int,
    patcher: SwapPatcher | InjectPatcher | None = None,
    want_scores: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    """Greedy generation, optionally under a patcher; returns (new_tokens, scores|None)."""
    kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": False, "use_cache": True}
    if want_scores:
        kwargs.update(output_scores=True, return_dict_in_generate=True)
    if patcher is None:
        out = model.hf_model.generate(**batch.hf_kwargs(), **kwargs)
    else:
        with hooked(model, patcher.hooks()):
            out = model.hf_model.generate(**batch.hf_kwargs(), **kwargs)
    if want_scores:
        sequences = out.sequences
        scores = [step[0].detach().float() for step in out.scores]
    else:
        sequences, scores = out, None
    new_tokens = sequences[0, batch.seq_len :].detach().cpu()
    return new_tokens, scores


def encode_chat(model: LlavaLensModel, body: str, max_seq_len: int) -> MultimodalBatch:
    """Text-only chat-wrapped prompt through the canonical processor path."""
    return model.encode_mm(f"{CHAT_USER} {body}{CHAT_ASSISTANT}", None, max_length=max_seq_len)


def decode_new(model: LlavaLensModel, new_tokens: torch.Tensor) -> str:
    return model._decode(new_tokens)


def first_word(text: str) -> str:
    """The completion's first whitespace word, stripped and lowercased."""
    words = text.strip().split()
    return words[0].lower() if words else ""


def mentions(text: str, surface: str) -> bool:
    """Word-boundary match of ``surface`` in lowercased ``text`` (MATCH_NOTE)."""
    return re.search(rf"\b{re.escape(surface.strip().lower())}\b", text.lower()) is not None


def wrap_words(text: str, width: int) -> list[str]:
    """Greedy word wrap at ``width`` chars (WRAP_NOTE); returns the lines."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def load_experiment(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _segment_prompt(
    model: LlavaLensModel, pieces: Sequence[tuple[str, str]]
) -> tuple[torch.Tensor, dict[str, torch.Tensor], str]:
    """Assemble a prompt from (tag, text) pieces with exact per-tag token spans.

    Each piece is tokenized on its own (no specials) and the ids concatenated, so a
    tag's span is exact by construction (SPAN_NOTE). BOS is prepended once, matching
    the processor path's add_bos_token; the literal '</s>' piece becomes the tokenizer's
    eos id when it has one (TEMPLATE_NOTE).
    """
    ids: list[int] = []
    bos = getattr(model.tokenizer, "bos_token_id", None)
    if bos is not None:
        ids.append(int(bos))
    spans: dict[str, list[tuple[int, int]]] = {}
    texts: list[str] = []
    for tag, text in pieces:
        texts.append(text)
        start = len(ids)
        if text == "</s>" and getattr(model.tokenizer, "eos_token_id", None) is not None:
            ids.append(int(model.tokenizer.eos_token_id))
        else:
            ids.extend(_encode_no_special(model, text))
        spans.setdefault(tag, []).append((start, len(ids)))
    input_ids = torch.tensor([ids], dtype=torch.long, device=model.input_device)
    masks = {
        tag: torch.zeros(len(ids), dtype=torch.bool, device=model.input_device)
        for tag in spans
    }
    for tag, ranges in spans.items():
        for lo, hi in ranges:
            masks[tag][lo:hi] = True
    return input_ids, masks, "".join(texts)


def text_only_batch(
    input_ids: torch.Tensor, model: LlavaLensModel, text: str
) -> MultimodalBatch:
    """A prompt already encoded as ids into a batch (no image tokens anywhere)."""
    image_token_mask = input_ids == model.image_token_id
    return MultimodalBatch(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        pixel_values=None,
        image_token_mask=image_token_mask,
        text=text,
    )


def span_norms(
    model: LlavaLensModel,
    batch: MultimodalBatch,
    layers: Sequence[int],
    mask: torch.Tensor,
) -> dict[int, float]:
    """Mean residual L2 norm per layer over the masked prompt positions (one forward)."""
    with ActivationRecorder(model.layers, at=sorted(set(int(l) for l in layers))) as recorder:
        model.forward_mm(batch)
    return {
        int(layer): float(
            recorder.activations[int(layer)][0][mask].float().norm(dim=-1).mean().item()
        )
        for layer in layers
    }


def decode_id(model: LlavaLensModel, token_id: int) -> str:
    """One token's decoded surface (used to read the model's top-10 outputs)."""
    return str(model.tokenizer.decode([int(token_id)]))


def _numeric_token_id(model: LlavaLensModel, value: int) -> int | None:
    """The numeric token id for ``value``: leading-space form, else bare (NUMERIC_NOTE)."""
    for form in (f" {value}", str(value)):
        ids = _encode_no_special(model, form)
        if len(ids) == 1:
            return ids[0]
    return None


def numeric_id_map(model: LlavaLensModel, max_value: int) -> dict[int, int]:
    """Single-token numeric forms of ``0..max_value`` -> value (NUMERIC_NOTE)."""
    by_id: dict[int, int] = {}
    for value in range(max_value + 1):
        token_id = _numeric_token_id(model, value)
        if token_id is not None and token_id not in by_id:
            by_id[token_id] = value
    return by_id


def base_meta(
    args: argparse.Namespace,
    model: LlavaLensModel,
    lens_meta: dict[str, Any],
    calib: Calibration | None,
    layers: list[int],
    band_note: str,
) -> dict[str, Any]:
    """Meta shared by every arm's JSON report."""
    return {
        "experiment": "x21_verbal_report",
        "arm": args.arm,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "n_layers": model.n_layers,
        "d_model": model.d_model,
        "lens": lens_meta,
        "calibrated": calib is not None,
        "calibrated_note": (
            "readout = unembed(J_l h) + the repo's bias/scale/temp/logit_bias chain"
            if calib is not None
            else "readout = the paper's raw J-lens readout unembed(J_l h); "
            "lens vectors are always the raw rows of W_U @ J_l"
        ),
        "layers": layers,
        "band_note": band_note,
        "max_new_tokens": args.max_new_tokens or ARM_DEFAULT_NEW_TOKENS[args.arm],
        "max_seq_len": args.max_seq_len,
        "experiments_dir": str(args.experiments_dir),
    }


# ----------------------------------------------------------------- arm: report


@torch.no_grad()
def arm_report(
    args: argparse.Namespace,
    model: LlavaLensModel,
    lens: JacobianLens,
    calib: Calibration | None,
    layers: list[int],
) -> dict[str, Any]:
    """Verbal report: (a) lens-vs-output Spearman at the colon; (b) coordinate swaps."""
    data = load_experiment(Path(args.experiments_dir) / "verbal-report.json")
    categories = list(data["candidates"].keys())
    max_new = args.max_new_tokens or ARM_DEFAULT_NEW_TOKENS["report"]
    collected: dict[int, dict[str, list[float]]] = {
        layer: {"per_category": [], "pooled_lens": [], "pooled_model": []} for layer in layers
    }
    category_records: list[dict[str, Any]] = []
    swap_trials: list[dict[str, Any]] = []
    n_argmax_mismatch = 0

    print(
        f"== X21 report arm: {len(categories)} categories x {args.n_candidates} candidates, "
        f"layers {layers[0]}..{layers[-1]} ({len(layers)}), swaps={'off' if args.n_swaps == 0 else args.n_swaps}/cat =="
    )
    for category in categories:
        all_candidates = [str(name) for name in data["candidates"][category]]
        readout_candidates = all_candidates[: args.n_candidates]
        cand_ids = [token_last_id(model, f" {name}") for name in readout_candidates]
        batch = encode_chat(model, REPORT_TEMPLATE.format(category=category), args.max_seq_len)
        new_tokens, step_records, scores = trace_generation(
            model, batch, layers, max_new_tokens=max_new
        )
        baseline_text = decode_new(model, new_tokens)
        answer = first_word(baseline_text)
        source_id = int(new_tokens[0])
        colon_scores = scores[0]
        n_argmax_mismatch += int(colon_scores.argmax().item() != source_id)
        top10 = torch.topk(colon_scores, k=min(10, colon_scores.shape[-1])).indices.tolist()
        top10_surfaces = {decode_id(model, tid).strip().lower() for tid in top10}

        for layer in layers:
            z = lens_logits(model, lens, step_records[0][layer], layer, calib)
            cand_lens = [float(z[cid].item()) for cid in cand_ids]
            cand_model = [float(colon_scores[cid].item()) for cid in cand_ids]
            collected[layer]["per_category"].append(spearman(cand_lens, cand_model))
            collected[layer]["pooled_lens"].extend(cand_lens)
            collected[layer]["pooled_model"].extend(cand_model)

        # (b) swap targets: same category, not the answer, not in the model's top-10.
        excluded_ids = set(top10) | {source_id}
        targets: list[tuple[str, int]] = []
        for name in all_candidates:
            if len(targets) >= args.n_swaps:
                break
            if name.strip().lower() == answer or name.strip().lower() in top10_surfaces:
                continue
            target_id = token_last_id(model, f" {name}")
            if target_id in excluded_ids:
                continue
            targets.append((name, target_id))
            excluded_ids.add(target_id)

        swaps: list[dict[str, Any]] = []
        if targets:
            vectors = vectors_for_ids(
                model, lens, [source_id] + [tid for _name, tid in targets], layers
            )
            for index, (name, target_id) in enumerate(targets):
                layer_vectors = {
                    layer: {
                        "source": vectors[layer][0],
                        "target": vectors[layer][index + 1],
                    }
                    for layer in layers
                }
                patcher = SwapPatcher(model, layer_vectors, args.swap_alpha)
                swap_tokens, swap_scores = greedy_generate(
                    model, batch, max_new_tokens=max_new, patcher=patcher, want_scores=True
                )
                patcher.assert_fired()
                swap_text = decode_new(model, swap_tokens)
                swap_word = first_word(swap_text)
                target_lower = name.strip().lower()
                success = swap_word == target_lower or swap_text.strip().lower().startswith(target_lower)
                first_subword = _encode_no_special(model, f" {name}")[0]
                target_rank = rank_of_id(swap_scores[0], first_subword)
                swap_record = {
                    "target": name,
                    "target_id": target_id,
                    "success": bool(success),
                    "first_word": swap_word,
                    "completion": swap_text,
                    "target_step0_rank": target_rank,
                    "target_in_top5": bool(target_rank <= 5),
                }
                swaps.append(swap_record)
                swap_trials.append({"category": category, **swap_record})

        category_records.append(
            {
                "category": category,
                "prompt": batch.text,
                "baseline_text": baseline_text,
                "baseline_first_word": answer,
                "chosen_in_candidates": answer in {name.lower() for name in readout_candidates},
                "source_token_id": source_id,
                "model_top10": [decode_id(model, tid).strip() for tid in top10],
                "candidates": [
                    {
                        "name": name,
                        "token_id": cid,
                        "model_logit_rank": rank_of_id(colon_scores, cid),
                    }
                    for name, cid in zip(readout_candidates, cand_ids)
                ],
                "swaps": swaps,
            }
        )
        success_count = sum(1 for swap in swaps if swap["success"])
        top5_count = sum(1 for swap in swaps if swap["target_in_top5"])
        print(
            f"  {category}: answer={answer!r} in_candidates="
            f"{category_records[-1]['chosen_in_candidates']} swaps={success_count}/{len(swaps)}"
            f" (top5 {top5_count}/{len(swaps)})",
            flush=True,
        )

    per_layer = {
        str(layer): {
            "mean_spearman_per_category": mean_or_none(collected[layer]["per_category"]),
            "pooled_spearman": spearman(
                collected[layer]["pooled_lens"], collected[layer]["pooled_model"]
            ),
            "n_points": len(collected[layer]["pooled_lens"]),
        }
        for layer in layers
    }
    n_success = sum(1 for trial in swap_trials if trial["success"])
    n_top5 = sum(1 for trial in swap_trials if trial["target_in_top5"])
    summary = {
        "n_categories": len(categories),
        "n_candidates_per_category": args.n_candidates,
        "n_swap_trials": len(swap_trials),
        "n_swap_success": n_success,
        "swap_success_rate": round(n_success / len(swap_trials), 4) if swap_trials else None,
        "n_target_top5": n_top5,
        "target_top5_rate": round(n_top5 / len(swap_trials), 4) if swap_trials else None,
        "swap_alpha": args.swap_alpha,
        "n_argmax_mismatch": n_argmax_mismatch,
    }

    print(f"{'layer':<7}{'mean_rho':>10}{'pooled_rho':>11}{'n':>6}")
    for layer in layers:
        row = per_layer[str(layer)]
        mean_cell = f"{row['mean_spearman_per_category']:>10.3f}" if row["mean_spearman_per_category"] is not None else f"{'-':>10}"
        pooled_cell = f"{row['pooled_spearman']:>11.3f}" if row["pooled_spearman"] is not None else f"{'-':>11}"
        print(f"L{layer:<6}{mean_cell}{pooled_cell}{row['n_points']:>6}")
    scored = [
        (layer, per_layer[str(layer)]["mean_spearman_per_category"])
        for layer in layers
        if per_layer[str(layer)]["mean_spearman_per_category"] is not None
    ]
    if scored:
        best_layer, best_rho = max(scored, key=lambda pair: pair[1])
        print(f"best mean rho at L{best_layer} ({best_rho:.3f}); expect a rise toward the end of the band")
    if swap_trials:
        print(
            f"swaps: {n_success}/{len(swap_trials)} answers flipped to the target "
            f"({summary['swap_success_rate']}), target in top-5 {n_top5}/{len(swap_trials)} "
            f"({summary['target_top5_rate']})"
        )
    if n_argmax_mismatch:
        print(
            f"WARNING: {n_argmax_mismatch} categories where argmax(colon logits) != the "
            "generated token (generation-config processors?); rankings describe the raw "
            "next-token distribution",
            flush=True,
        )

    return {
        "meta": base_meta(args, model, {}, calib, layers, BAND_NOTE),
        "notes": {
            "colon": COLON_NOTE,
            "candidates": CANDIDATE_NOTE,
            "rank_rule": RANK_RULE,
            "swap": SWAP_NOTE,
            "source": SOURCE_NOTE,
            "target_selection": TARGET_NOTE,
            "success": SUCCESS_NOTE,
            "decomposition_arm": DECOMPOSITION_NOTE,
        },
        "per_layer": per_layer,
        "summary": summary,
        "categories": category_records,
    }


# ------------------------------------------------------------ arm: introspect


@torch.no_grad()
def arm_introspect(
    args: argparse.Namespace,
    model: LlavaLensModel,
    lens: JacobianLens,
    layers: list[int],
) -> dict[str, Any]:
    """Injected-thought introspection: inject v_t over the user turn, measure the report rate."""
    data = load_experiment(Path(args.experiments_dir) / "verbal-introspection.json")
    turns = data["intro_prompt"]
    user_1 = turns[0]["content"]
    assistant_1 = turns[1]["content"]
    user_2 = turns[2]["content"]
    prefill = data["prefills"][args.prefill_key]
    pieces = [
        ("frame", f"{CHAT_USER} "),
        ("u1", user_1),
        ("frame", f"{CHAT_ASSISTANT} "),
        ("a1", assistant_1),
        ("frame", "</s>\nUSER: "),
        ("u2", user_2),
        ("frame", CHAT_ASSISTANT),
        ("prefill", prefill),
    ]
    input_ids, masks, full_text = _segment_prompt(model, pieces)
    batch = text_only_batch(input_ids, model, full_text)
    user_mask = masks["u1"] | masks["u2"]
    assistant_mask = masks["a1"]
    concepts = data["concepts"][: args.n_concepts]
    strengths = [float(part) for part in args.strengths.split(",")]
    max_new = args.max_new_tokens or ARM_DEFAULT_NEW_TOKENS["introspect"]

    baseline_tokens, _scores = greedy_generate(model, batch, max_new_tokens=max_new)
    baseline_text = decode_new(model, baseline_tokens)
    baseline_reported = [
        concept["surface"] for concept in concepts if mentions(baseline_text, concept["surface"])
    ]
    norms = {
        "user": span_norms(model, batch, layers, user_mask),
        "assistant": span_norms(model, batch, layers, assistant_mask),
    }

    print(
        f"== X21 introspect arm: {len(concepts)} concepts x {len(strengths)} strengths x 2 spans "
        f"(user span {int(user_mask.sum())} tok, assistant control {int(assistant_mask.sum())} tok), "
        f"layers {layers[0]}..{layers[-1]} ({len(layers)}) =="
    )
    print(f"  prompt: {full_text[:90]!r}... (+{len(full_text) - 90} chars)")
    print(f"  baseline completion: {baseline_text!r}")
    if baseline_reported:
        print(f"  baseline already mentions: {baseline_reported}")

    hits: dict[str, dict[float, list[bool]]] = {
        span: {strength: [] for strength in strengths} for span in ("user", "assistant")
    }
    trials: list[dict[str, Any]] = []
    for concept in concepts:
        surface = str(concept["surface"])
        concept_id = token_last_id(model, f" {surface}")
        vectors = vectors_for_ids(model, lens, [concept_id], layers)
        per_strength = {strength: {"user": False, "assistant": False} for strength in strengths}
        for strength in strengths:
            for span_name, mask in (("user", user_mask), ("assistant", assistant_mask)):
                specs: dict[int, tuple[torch.Tensor, float, torch.Tensor]] = {}
                for layer in layers:
                    vector = vectors[layer][0]
                    v_norm = float(vector.norm().item())
                    alpha = strength * norms[span_name][layer] / max(v_norm, 1e-8)
                    specs[layer] = (vector, alpha, mask)
                patcher = InjectPatcher(model, batch.seq_len, specs)
                new_tokens, _scores = greedy_generate(
                    model, batch, max_new_tokens=max_new, patcher=patcher
                )
                patcher.assert_fired()
                text = decode_new(model, new_tokens)
                reported = mentions(text, surface)
                per_strength[strength][span_name] = reported
                hits[span_name][strength].append(reported)
                trials.append(
                    {
                        "concept": concept["name"],
                        "surface": surface,
                        "strength": strength,
                        "span": span_name,
                        "reported": bool(reported),
                        "completion": text,
                    }
                )
        flags = " ".join(
            f"k={strength:g}:{'T' if per_strength[strength]['user'] else 'F'}"
            f"{'/cT' if per_strength[strength]['assistant'] else '/cF'}"
            for strength in strengths
        )
        print(f"  {surface}: {flags}", flush=True)

    rates = {
        span: {
            str(strength): round(sum(hits[span][strength]) / len(hits[span][strength]), 4)
            if hits[span][strength]
            else None
            for strength in strengths
        }
        for span in ("user", "assistant")
    }
    print(f"{'strength':>9}{'user_rate':>11}{'asst_rate':>11}{'n_each':>8}")
    for strength in strengths:
        user_cell = (
            f"{rates['user'][str(strength)]:>11.3f}"
            if rates["user"][str(strength)] is not None
            else f"{'-':>11}"
        )
        asst_cell = (
            f"{rates['assistant'][str(strength)]:>11.3f}"
            if rates["assistant"][str(strength)] is not None
            else f"{'-':>11}"
        )
        print(f"{strength:>9g}{user_cell}{asst_cell}{len(hits['user'][strength]):>8}")
    print(
        "expected: the injected concept is reported on user-turn trials at high "
        "strengths and ~never on the assistant-position control or the no-injection "
        "baseline"
    )

    report = {
        "meta": base_meta(args, model, {}, None, layers, BAND_NOTE),
        "notes": {
            "template": TEMPLATE_NOTE,
            "spans": SPAN_NOTE,
            "strength": STRENGTH_NOTE,
            "match": MATCH_NOTE,
            "prefill_key": args.prefill_key,
            "user_span_tokens": int(user_mask.sum()),
            "assistant_span_tokens": int(assistant_mask.sum()),
        },
        "baseline_completion": baseline_text,
        "baseline_reported": baseline_reported,
        "rates": rates,
        "trials": trials,
    }
    return report


@torch.no_grad()
def arm_modulation(
    args: argparse.Namespace,
    model: LlavaLensModel,
    lens: JacobianLens,
    calib: Calibration | None,
    layers: list[int],
) -> dict[str, Any]:
    """Directed modulation: focus/ignore/baseline while copying, target-in-top-10 rates."""
    mod = load_experiment(Path(args.experiments_dir) / "directed-modulation.json")
    linecount = load_experiment(Path(args.experiments_dir) / "selectivity-linecount.json")
    carrier = str(mod["carrier_sentences"][0])
    focus_phrasings = [p for p in mod["phrasings"] if p["group"] == "focus"]
    ignore_phrasings = [p for p in mod["phrasings"] if p["group"] == "dismissal"]
    phrasing = {
        "focus": focus_phrasings[args.phrasing_index % len(focus_phrasings)]["text"],
        "ignore": ignore_phrasings[args.phrasing_index % len(ignore_phrasings)]["text"],
    }
    conditions = ("focus", "ignore", "baseline")
    n_items = args.n_items_per_family
    max_new = args.max_new_tokens or ARM_DEFAULT_NEW_TOKENS["modulation"]

    # Items: (family, x, display, target ids, wrapped lines or None)
    items: list[dict[str, Any]] = []
    for topic in mod["topic_categories"][:n_items]:
        names = [str(topic["name"])] + [str(m) for m in topic["members"]]
        ids = sorted({token_last_id(model, f" {name}") for name in names})
        items.append(
            {"family": "concept", "x": str(topic["name"]), "display": str(topic["name"]), "target_ids": ids}
        )
    for problem in mod["math_problems"][:n_items]:
        answer = str(problem["answer"])
        ids = sorted({token_last_id(model, f" {answer}"), token_last_id(model, answer)})
        items.append(
            {
                "family": "math",
                "x": str(problem["expr"]),
                "display": f"{problem['expr']}={problem['answer']}",
                "target_ids": ids,
            }
        )
    for passage in linecount["passages"][:n_items]:
        lines = wrap_words(str(passage["text"]), int(passage["width"]))
        truth = len(lines[0])
        ids = sorted({token_last_id(model, f" {truth}"), token_last_id(model, str(truth))})
        items.append(
            {
                "family": "linecount",
                "x": str(passage["tag"]),
                "display": f"{passage['tag']}(width {truth})",
                "target_ids": ids,
                "lines": lines,
                "truth": truth,
            }
        )

    numeric_map = numeric_id_map(model, args.numeric_max)
    numeric_ids = torch.tensor(
        sorted(numeric_map), dtype=torch.long, device=model.unembed_weight().device
    )

    print(
        f"== X21 modulation arm: {n_items} items x 3 families x 3 conditions "
        f"(carrier {carrier!r}), layers {layers[0]}..{layers[-1]} ({len(layers)}) =="
    )
    print(f"  phrasings: focus={phrasing['focus']!r} ignore={phrasing['ignore']!r}")

    trials: list[dict[str, Any]] = []
    for item in items:
        for condition in conditions:
            if item["family"] == "linecount":
                instruction = {"focus": COUNT_FOCUS, "ignore": COUNT_IGNORE, "baseline": ""}[condition]
                body = LC_BODY.format(passage="\n".join(item["lines"]), instruction=instruction)
            else:
                instruction = phrasing[condition].format(x=item["x"]) if condition != "baseline" else ""
                body = MOD_BODY.format(carrier=carrier, instruction=instruction)
            batch = encode_chat(model, body, args.max_seq_len)
            new_tokens, step_records, _scores = trace_generation(
                model, batch, layers, max_new_tokens=max_new
            )
            text = decode_new(model, new_tokens)
            target_set = set(item["target_ids"])
            hit = False
            n_hit_positions = 0
            numeric_errors: list[int] = []
            numeric_exact = 0
            numeric_reads = 0
            for step_record in step_records:  # every step of the copied region
                for layer in layers:
                    z = lens_logits(model, lens, step_record[layer], layer, calib)
                    top10 = torch.topk(z, k=min(10, z.shape[-1])).indices.tolist()
                    if any(tid in target_set for tid in top10):
                        hit = True
                        n_hit_positions += 1
                    if item["family"] == "linecount":
                        gathered = z[numeric_ids]
                        value = numeric_map[int(numeric_ids[int(gathered.argmax().item())].item())]
                        error = abs(value - int(item["truth"]))
                        numeric_errors.append(error)
                        numeric_exact += int(error == 0)
                        numeric_reads += 1
            trial = {
                "family": item["family"],
                "item": item["display"],
                "condition": condition,
                "prompt": batch.text,
                "completion": text,
                "n_steps": int(new_tokens.numel()),
                "hit": bool(hit),
                "n_hit_positions": n_hit_positions,
            }
            if item["family"] == "linecount":
                trial["numeric_mae"] = round(sum(numeric_errors) / len(numeric_errors), 3) if numeric_errors else None
                trial["numeric_exact_rate"] = round(numeric_exact / numeric_reads, 4) if numeric_reads else None
            trials.append(trial)
        flags = " ".join(
            f"{condition}={'T' if trials[-3 + index]['hit'] else 'F'}" for index, condition in enumerate(conditions)
        )
        print(f"  [{item['family']}] {item['display']}: {flags}", flush=True)

    digest: dict[str, dict[str, dict[str, Any]]] = {}
    for family in ("concept", "math", "linecount"):
        digest[family] = {}
        for condition in conditions:
            bucket = [t for t in trials if t["family"] == family and t["condition"] == condition]
            cell: dict[str, Any] = {
                "n": len(bucket),
                "hit_rate": round(sum(1 for t in bucket if t["hit"]) / len(bucket), 4) if bucket else None,
            }
            if family == "linecount":
                maes = [t["numeric_mae"] for t in bucket if t["numeric_mae"] is not None]
                exacts = [t["numeric_exact_rate"] for t in bucket if t["numeric_exact_rate"] is not None]
                cell["numeric_mae"] = round(sum(maes) / len(maes), 3) if maes else None
                cell["numeric_exact_rate"] = round(sum(exacts) / len(exacts), 4) if exacts else None
            digest[family][condition] = cell
    print(f"{'family':<10}{'cond':>9}{'n':>4}{'hit_rate':>10}{'mae':>8}{'exact':>8}")
    for family in ("concept", "math", "linecount"):
        for condition in conditions:
            cell = digest[family][condition]
            mae_cell = f"{cell['numeric_mae']:>8.2f}" if cell.get("numeric_mae") is not None else f"{'-':>8}"
            exact_cell = f"{cell['numeric_exact_rate']:>8.3f}" if cell.get("numeric_exact_rate") is not None else f"{'-':>8}"
            rate_cell = f"{cell['hit_rate']:>10.3f}" if cell["hit_rate"] is not None else f"{'-':>10}"
            print(f"{family:<10}{condition:>9}{cell['n']:>4}{rate_cell}{mae_cell}{exact_cell}")
    print(
        "expected ordering: focus > ignore > baseline ~0 (the copied region carries the "
        "directed thought in the lens readout); line-count precision should track the "
        "focus condition's count"
    )

    return {
        "meta": base_meta(args, model, {}, calib, layers, BAND_NOTE),
        "notes": {
            "template": TEMPLATE_NOTE,
            "carrier": carrier,
            "phrasing_index": args.phrasing_index,
            "count_focus": COUNT_FOCUS,
            "count_ignore": COUNT_IGNORE,
            "numeric": NUMERIC_NOTE,
            "wrap": WRAP_NOTE,
            "match": MATCH_NOTE,
        },
        "digest": digest,
        "trials": trials,
    }


# ------------------------------------------------------------------------ main


def main() -> int:
    args = parse_args()
    model = load_model(args)
    lens, lens_meta = load_lens(args, model)
    calib = load_calibration(args, model)
    layers, band_note = resolve_layers(args, lens)
    print(
        f"model ready: layers={model.n_layers} d_model={model.d_model} backend={args.backend}; "
        f"lens={lens_meta}; band layers={layers}"
    )
    if args.arm == "report":
        report = arm_report(args, model, lens, calib, layers)
    elif args.arm == "introspect":
        report = arm_introspect(args, model, lens, layers)
    else:
        report = arm_modulation(args, model, lens, calib, layers)
    report["meta"]["lens"] = lens_meta
    report["meta"]["band_note"] = band_note
    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
