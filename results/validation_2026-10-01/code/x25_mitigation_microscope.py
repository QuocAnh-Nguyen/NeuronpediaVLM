#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""X25: hallucination-mitigation decomposition arm — baseline vs M3ID vs VCD vs
prior-ablation, all greedy, same seeded prompts, scored on POPE + CHAIR + the lens.

GOAL (D48): convert the workspace findings into hallucination mitigation validated on
BOTH captioning (CHAIR against COCO instances annotations) and yes/no (POPE F1/accuracy).
This arm runs the mitigation CANDIDATES side by side under identical conditions and
records, per arm, the surface metrics AND the post-hoc workspace readout shift, so the
question "which mitigation helps" comes with "what did it do to the workspace band":

arms (one model load; all greedy ``do_sample=False``, identical prompts and corpora):
  baseline      plain greedy generate.
  m3id          ``logits_final - tau * logits_no_image`` at every decoding step, tau in
                {0.5, 1.0, 2.0}. The image-free prior is the SAME text WITHOUT image
                tokens, forwarded once per sample; its final-position logits are reused
                as a per-step CONSTANT contrast vector (CONTRAST_NOTE documents the
                sequence-length rationale). HF path: a LogitsProcessor on generate().
  vcd           same subtraction form, but the contrast vector = final-position logits
                of a DISTORTED-image forward (PIL-only numpy gaussian noise + mild
                GaussianBlur; DISTORTION_NOTE; no torchvision). Same tau grid.
  prior-ablate  the measured image-prior direction ``v_l = normalize(h_l(image) -
                h_l(no image))`` at the two prompts' final (handoff) positions, ablated
                during generation with projection magnitude alpha in {0.5, 1.0} at
                --ablate-layer (default: the x22 top1-onset layer per kind, D48) through
                interventions.ResidualEditor mode=ablate (DirectionEditor: a measured
                state-difference vector, not a token J-lens row; POSITIONS_NOTE).
  register-off  NOT implemented here — T30 owns the visual-register arm (placeholder
                kept in the arm numbering only).

tasks (--tasks pope,captions):
  pope      per split (random/popular/adversarial; --pope-dir coco_pope_<split>.jsonl):
            accuracy / yes-class precision-recall-F1 / yes-rate per arm; the answer is
            the LEADING yes/no token of the generation (tokenizer-agnostic parse;
            unparsed counts as incorrect and is reported).
  captions  seeded COCO sample (the manifest256 builder's select_images convention),
            greedy captions per arm; CHAIR-s / CHAIR-r against the image's instances
            annotations with an exact + plural-strip matcher (MATCHER_NOTE), plus mean
            caption length (words and tokens).
  lens      trajectory (captions only, post-hoc and cheap): each arm's captions are
            teacher-forced through the model once; at the band layers the calibrated
            J-lens readout gives mean P(yes) and the GT-object-in-top-k rate over the
            caption positions; reported per layer as the shift vs the baseline arm —
            did the mitigation move the workspace readout, not just the surface text?

Output JSON: meta (model/backend/seed/lens dir+fingerprint/bias) + per-arm entries
(params note, pope per split, captions metrics, lens trajectory + shift) + digest +
summary. One invocation covers every arm (--tasks prunes tasks, not arms).

tiny smoke (no GPU, no COCO, no semantics — mechanics only):
  python x25_mitigation_microscope.py --backend tiny --synthetic \
      --layers 1,2 --ablate-layer 2 --max-new-tokens 8 --pope-max-new-tokens 4 \
      --n-images 2 --json /tmp/x25_smoke.json
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def _find_repo_root() -> Path:
    """The repo owning ``third_party/jacobian-lens`` (walk up from this file).

    The orchestrator may run this script from an arbitrary directory (e.g.
    /home/nvidia/vlm-run), where ``__file__``-relative parents do not reach the
    repo; the imports then resolve via the environment's PYTHONPATH instead.
    """
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "third_party" / "jacobian-lens").is_dir():
            return candidate
    return here.parents[3] if len(here.parents) > 3 else Path(here.anchor)


REPO = _find_repo_root()
CODE = Path(__file__).resolve().parent
for _path in (str(REPO / "src"), str(CODE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import LogitsProcessor, LogitsProcessorList  # noqa: E402

import vlm_lens  # noqa: E402, F401  # installs the vendored jlens path: must precede jlens
from jlens.hooks import ActivationRecorder  # noqa: E402
from jlens.lens import JacobianLens  # noqa: E402

from vlm_lens.data.captions import DEFAULT_QUESTIONS  # noqa: E402
from vlm_lens.data.pope import read_pope_jsonl  # noqa: E402
from vlm_lens.interventions import (  # noqa: E402
    ResidualEdit,
    ResidualEditor,
)

import vlm_registry as registry  # noqa: E402

#: Sibling reuse (the x19/x23 pattern): the calibrated readout math and loaders come
#: from x14, the yes/no token convention from x18, the lens fingerprint from x21, and
#: the corpus/name/bias helpers from x23's workspace — imported, not re-derived.
from x14_workspace_probe import (  # noqa: E402
    _calibrated_lens_logits,
    _category_token_ids,
    identity_lens,
)
from x18_pope_handoff import select_yes_no_ids  # noqa: E402
from x21_verbal_report import FINGERPRINT_NOTE, lens_fingerprint  # noqa: E402
from x23_vlm_workspace import (  # noqa: E402
    Sample,
    _fmt,
    _mean,
    _round,
    coco_samples,
    drop_mismatched_bias,
    gt_name_set,
    load_bias_optional,
    load_lens_optional,
    name_map,
    synthetic_samples,
)

#: Any registry-built lens model (``LlavaLensModel`` for the llava kinds,
#: ``Qwen2VLLensModel`` for qwen2vl -- same public surface), so the annotations
#: stay honest without importing the qwen2vl class at module scope.
LensModel = Any

DEFAULT_LENS_DIR = "/data/anhnq/vlm-lens-out/validation/fits/llava-1.5-7b-hf/merged"
DEFAULT_BIAS_DIR = "/data/anhnq/vlm-lens-out/validation/step5e"
DEFAULT_IMAGES_DIR = "/data/baodq/coco2014/val2014"
DEFAULT_ANNOTATIONS = "/data/baodq/coco2014/annotations/instances_val2014.json"
#: The POPE release directory (standard file names coco_pope_<split>.jsonl). The repo
#: has no prior POPE-file convention (x18 synthesized its questions), so the production
#: launcher passes the real path explicitly; missing files fail closed.
DEFAULT_POPE_DIR = "/data/baodq/coco2014/pope"

#: The captioning question (x23's band-arm convention: DEFAULT_QUESTIONS[0]).
CAPTION_QUESTION = DEFAULT_QUESTIONS[0]

#: The POPE splits, in canonical order (synthetic negatives key off this index).
POPE_SPLITS: tuple[str, ...] = ("random", "popular", "adversarial")

#: The mitigation arms' parameter grids (the x25 spec).
DEFAULT_M3ID_TAUS = "0.5,1.0,2.0"
DEFAULT_VCD_TAUS = "0.5,1.0,2.0"
DEFAULT_ABLATE_ALPHAS = "0.5,1.0"

#: --ablate-layer defaults per kind: the x22 top1-onset layer (D48: the workspace band
#: = top1-onset layers, 7b L20-29, 13b L25-36, qwen L24-26; the shallowest onset layer
#: is where the image-prior direction first matters). The tiny fixture has 4 layers, so
#: the tiny smoke passes --ablate-layer explicitly.
ABLATE_LAYER_DEFAULTS: dict[str, int] = {"llava-7b": 20, "llava-13b": 25, "qwen2vl": 24}

#: VCD distortion defaults: gaussian noise on the 0-255 uint8 scale + a mild PIL blur.
#: PIL-only by the repo invariant (images reach processors as PIL; no torchvision).
DEFAULT_VCD_NOISE_SIGMA = 15.0
DEFAULT_VCD_BLUR_RADIUS = 2.0

TASKS: tuple[str, ...] = ("pope", "captions")

#: The baseline arm id (the shift/reference key everywhere).
BASELINE_ARM = "baseline"

#: Placeholder token string for the prior-ablate ResidualEdit spec. It is never
#: resolved through the lens (DirectionEditor bypasses _vectors_for); it only keeps
#: ResidualEdit's validation and JSON reporting happy.
PRIOR_TOKEN_LABEL = "prior:v_l"


# ------------------------------------------------------------------------ notes


CONTRAST_NOTE = (
    "m3id/vcd contrast: the contrast prompt (the image-free text, or the image with "
    "distortion) is forwarded ONCE per sample and its FINAL-POSITION logits are reused "
    "as a per-step CONSTANT contrast vector: step_logits - tau * contrast. The contrast "
    "prompt tokenizes to a different sequence length than the image prompt (image "
    "placeholders), so per-step re-forwarding would need a per-step position alignment "
    "that does not exist; the constant vector makes the contrast EXACT at the "
    "prefill/handoff decision (generated token 0) and an approximation afterwards. Both "
    "arms share one LogitsProcessor; only the contrast vector differs"
)
VCD_FORM_NOTE = (
    "vcd uses the m3id subtraction form logits - tau * logits_distorted (the x25 spec's "
    "'same but' reading) rather than canonical VCD's (1+alpha)*l - alpha*l_aug, which "
    "differs by the +alpha*l term; the same tau grid as m3id is used so the contrast "
    "weight is directly comparable across arms"
)
DISTORTION_NOTE = (
    "vcd distortion is PIL-only (the repo invariant: images reach processors as PIL, "
    "no torchvision): RGB float array + seeded numpy gaussian noise (sigma on the "
    "0-255 scale) clipped back to uint8, then PIL ImageFilter.GaussianBlur(radius). "
    "Seeded per sample (seed, sample index) so every arm sees the same distortion"
)
PRIOR_ABLATE_NOTE = (
    "prior-ablate direction: v_l = normalize(h_l(image) - h_l(no image)) measured at "
    "the FINAL PROMPT position of each prompt (the handoff position whose distribution "
    "predicts generated token 0; the x14 step_records[0] convention) in the RAW "
    "residual space at --ablate-layer. This is a measured state difference, NOT a "
    "token J-lens vector, so ResidualEditor's token-string resolution is bypassed "
    "(DirectionEditor) while its hook machinery and apply_edit ablate semantics are "
    "reused verbatim: h' = h - alpha * <h, v_l> v_l at the edited positions"
)
POSITIONS_NOTE = (
    "the ablation scope is ResidualEdit positions='last': the prefill forward's final "
    "row (the handoff position) and every single-token decode row — the DECISION "
    "positions. Image-token rows are untouched (they do not recur under KV caching), "
    "so visual content keeps flowing while the image-prior component is removed where "
    "tokens are chosen (x23 SWAP_NOTE's text-scope decode convention)"
)
POPE_PARSE_NOTE = (
    "the answer is the LEADING yes/no token of the generated text: the first "
    "alphabetic word, lowercased, must be 'yes' or 'no' (tokenizer-agnostic string "
    "parse; 'Yes, there is...' -> yes, 'No.' -> no, anything else -> unparsed). "
    "Unparsed counts as INCORRECT for accuracy/F1 and is reported separately (it is a "
    "real failure mode, not a tie-break). Stricter than x18's parse_answer, which "
    "falls back to a unique yes/no anywhere in the text"
)
POPE_SLICE_NOTE = (
    "per-split records are the FIRST --n-images records of the shipped jsonl (POPE "
    "files are balanced by construction; the prefix slice keeps that balance and is "
    "deterministic)"
)
CHAIR_NOTE = (
    "CHAIR-s (sentence level) = fraction of captions mentioning >= 1 COCO category "
    "that is NOT annotated on the image; CHAIR-r (recall) = mean over images of "
    "|mentioned AND annotated| / |annotated| (images with an empty annotation set are "
    "excluded from the caption pool entirely, x23's convention). 'Mentioned' scans the "
    "caption for every category name in the instances file (the 80-class list — the "
    "standard CHAIR-in-COCO simplification)"
)
MATCHER_NOTE = (
    "x25 in-arm matcher (the x25 spec's simple matcher; vlm_registry exposes no "
    "matcher to reuse): per category name, the case-insensitive word-boundary regex "
    r"\b<name>(s|es)?\b — exact + plural-strip match. Documented gaps: no span masking "
    "('hot dog' also matches 'dog'; the x13 matcher's span masking is the upgrade if "
    "CHAIR needs it), no irregular plurals ('mice' does not match 'mouse'), no synonym "
    "collapse"
)
TRAJECTORY_NOTE = (
    "lens trajectory (captions only, post-hoc): each arm's generated caption is "
    "teacher-forced onto its prompt (one extra forward per image per arm, no edits "
    "active) and the calibrated J-lens readout (x14's _calibrated_lens_logits: "
    "identity transport + bias, * scale, unembed, / temp, + logit_bias; unfitted "
    "layers read raw) is evaluated at the CAPTION token positions only: mean P(yes) "
    "= softmax probability of the x18 yes id, and object_in_top_k = the rate at which "
    "one of the image's annotated category emission ids (x14's ' '+name final-subword "
    "convention) is in the readout's top-k. shift_vs_baseline = arm mean - baseline "
    "mean per layer (same images, so the shift is paired)"
)
SYNTH_NOTE = (
    "--synthetic builds the corpus in-memory (x23's exact convention): --n-images "
    "deterministic random RGB images, ids 1..n, each annotated with 2 categories "
    "cycled from the 16-name SYNTH_CATEGORIES pool. Synthetic POPE: 2 present + 2 "
    "absent questions per image per split ('Is there a <name> in the image?'), "
    "balanced 50/50; the absent pool is sampled uniformly regardless of split, so the "
    "popular/adversarial split NAMES exercise the per-split machinery but carry no "
    "hard-negative semantics (tiny has no real semantics)"
)
NO_REGISTER_NOTE = (
    "arm 5 (register-off) is intentionally absent: T30 owns the visual-register arm"
)
ARM_NOTES: dict[str, str] = {
    "contrast": CONTRAST_NOTE,
    "vcd_form": VCD_FORM_NOTE,
    "distortion": DISTORTION_NOTE,
    "prior_ablate": PRIOR_ABLATE_NOTE,
    "positions": POSITIONS_NOTE,
    "pope_parse": POPE_PARSE_NOTE,
    "pope_slice": POPE_SLICE_NOTE,
    "chair": CHAIR_NOTE,
    "matcher": MATCHER_NOTE,
    "trajectory": TRAJECTORY_NOTE,
    "synthetic": SYNTH_NOTE,
    "no_register_arm": NO_REGISTER_NOTE,
}


# ------------------------------------------------------------------- arm specs


@dataclass(frozen=True)
class ArmSpec:
    """One mitigation arm: identity + generation-time parameters + its JSON note."""

    arm_id: str  # "baseline" | "m3id_tau_0.5" | "vcd_tau_1.0" | "prior_ablate_alpha_1.0"
    family: str  # baseline | m3id | vcd | prior-ablate
    tau: float | None = None  # m3id/vcd contrast weight
    alpha: float | None = None  # prior-ablate projection magnitude

    @property
    def params(self) -> dict[str, Any]:
        """The arm's parameters as stored under ``arms.<id>.params``."""
        if self.family == "baseline":
            return {"family": self.family, "note": "plain greedy generate (do_sample=False)"}
        if self.family in ("m3id", "vcd"):
            return {
                "family": self.family,
                "tau": self.tau,
                "form": "logits - tau * contrast",
                "contrast": (
                    "final-position logits of the image-free prompt forward"
                    if self.family == "m3id"
                    else "final-position logits of the distorted-image forward"
                ),
                "contrast_note": CONTRAST_NOTE,
                **({"form_note": VCD_FORM_NOTE} if self.family == "vcd" else {}),
            }
        return {
            "family": self.family,
            "alpha": self.alpha,
            "direction": PRIOR_ABLATE_NOTE,
            "positions_note": POSITIONS_NOTE,
        }


def build_arms(args: argparse.Namespace) -> list[ArmSpec]:
    """The full arm list: baseline + m3id(taus) + vcd(taus) + prior-ablate(alphas)."""
    arms = [ArmSpec(BASELINE_ARM, "baseline")]
    for tau in parse_floats(args.m3id_taus, "--m3id-taus"):
        arms.append(ArmSpec(f"m3id_tau_{tau:g}", "m3id", tau=tau))
    for tau in parse_floats(args.vcd_taus, "--vcd-taus"):
        arms.append(ArmSpec(f"vcd_tau_{tau:g}", "vcd", tau=tau))
    for alpha in parse_floats(args.ablate_alphas, "--ablate-alphas"):
        arms.append(ArmSpec(f"prior_ablate_alpha_{alpha:g}", "prior-ablate", alpha=alpha))
    return arms


# --------------------------------------------------------------------- parsing


def parse_floats(raw: str, flag: str) -> list[float]:
    """A comma-separated non-negative float list (arm parameter grids)."""
    try:
        values = [float(part) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"{flag}: comma-separated floats expected ({exc})") from exc
    if not values:
        raise ValueError(f"{flag}: at least one value is required")
    if any(value < 0 for value in values):
        raise ValueError(f"{flag}: values must be >= 0, got {values}")
    return values


def parse_tasks(raw: str) -> list[str]:
    """The --tasks switch: a non-empty subset of TASKS, canonical order."""
    picked = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [task for task in picked if task not in TASKS]
    if unknown:
        raise ValueError(f"--tasks: unknown task(s) {unknown}; choices are {list(TASKS)}")
    if not picked:
        raise ValueError("--tasks: at least one task is required")
    return [task for task in TASKS if task in picked]


def parse_layers(raw: str | None, flag: str = "--layers") -> list[int] | None:
    """A comma-separated layer list, or None (the registry default)."""
    if raw is None:
        return None
    try:
        layers = [int(part) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"{flag}: comma-separated ints expected ({exc})") from exc
    if not layers:
        raise ValueError(f"{flag}: at least one layer is required")
    return layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("hf-llava", "tiny"), default="hf-llava",
        help="model backend: 'hf-llava' (default, CUDA) or the tiny CPU smoke fixture "
             "(4 layers, d_model 32; pass --layers within 0..3, --ablate-layer, and "
             "--synthetic)",
    )
    parser.add_argument(
        "--model", choices=registry.MODEL_KINDS, default=registry.DEFAULT_KIND,
        help="checkpoint kind (HF backend): " + ", ".join(registry.MODEL_KINDS)
             + "; --backend tiny implies llava-7b geometry (no tiny stand-in exists "
               "for the other kinds)",
    )
    parser.add_argument(
        "--tasks", default="pope,captions",
        help="comma-separated subset of: " + ",".join(TASKS) + " (arms are not "
             "prunable — one invocation always runs every mitigation arm)",
    )
    parser.add_argument("--lens-dir", default=DEFAULT_LENS_DIR,
                        help="J source lens dir for the trajectory readout (its fitted "
                             "layers define the identity-transport lens; if it cannot "
                             "be loaded the identity lens covers ALL layers and the "
                             "readout falls back to raw, x23's fallback convention). "
                             "Unused by the prior-ablate direction (raw residuals)")
    parser.add_argument("--mask", default="text",
                        help="which lens-<mask>.pt and bias-<mask>.pt to use")
    parser.add_argument("--bias-dir", default=DEFAULT_BIAS_DIR,
                        help="directory holding bias-<mask>.pt (the affine payload "
                             "applied inside every calibrated J-lens readout)")
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS,
                        help="instances_val2014.json: per-image ground truth for CHAIR")
    parser.add_argument("--pope-dir", default=DEFAULT_POPE_DIR,
                        help="directory with the POPE release files "
                             "coco_pope_<split>.jsonl (random/popular/adversarial)")
    parser.add_argument("--pope-splits", default=",".join(POPE_SPLITS),
                        help="comma-separated subset of: " + ",".join(POPE_SPLITS))
    parser.add_argument("--n-images", type=int, default=60,
                        help="caption corpus size AND per-split POPE record cap")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--layers", default=None,
        help="comma-separated band layers for the trajectory readout; default is the "
             "registry band subset (llava-7b: 16,20,24,28)",
    )
    parser.add_argument(
        "--ablate-layer", type=int, default=None,
        help="the layer whose final-position state difference defines the "
             "prior-ablate direction; default is the kind's x22 top1-onset layer "
             f"{ABLATE_LAYER_DEFAULTS} (D48). The tiny fixture must pass one",
    )
    parser.add_argument("--m3id-taus", default=DEFAULT_M3ID_TAUS,
                        help="m3id contrast weights (see CONTRAST_NOTE)")
    parser.add_argument("--vcd-taus", default=DEFAULT_VCD_TAUS,
                        help="vcd contrast weights (see VCD_FORM_NOTE)")
    parser.add_argument("--ablate-alphas", default=DEFAULT_ABLATE_ALPHAS,
                        help="prior-ablate projection magnitudes (see PRIOR_ABLATE_NOTE)")
    parser.add_argument("--vcd-noise-sigma", type=float, default=DEFAULT_VCD_NOISE_SIGMA,
                        help="gaussian noise sigma (0-255 scale) for the vcd distortion")
    parser.add_argument("--vcd-blur-radius", type=float, default=DEFAULT_VCD_BLUR_RADIUS,
                        help="PIL GaussianBlur radius for the vcd distortion")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="caption generation budget (the manifest256 round's 256)")
    parser.add_argument("--pope-max-new-tokens", type=int, default=8,
                        help="POPE answer budget (the leading yes/no token decides)")
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--top-k", type=int, default=10,
                        help="k for the trajectory's object-in-top-k statistic")
    parser.add_argument("--synthetic", action="store_true",
                        help="build the corpus in-memory (tiny smoke; see SYNTH_NOTE)")
    parser.add_argument("--json", required=True)
    return parser.parse_args()


# ------------------------------------------------------------------- workspace


@dataclass
class Workspace:
    """Everything the arms share: model kind, calibrated lens, category tables."""

    kind: str
    model: LensModel
    lens: JacobianLens
    bias: dict[int, torch.Tensor] | None
    scale: dict[int, float] | None
    temp: dict[int, float] | None
    logit_bias: dict[int, torch.Tensor] | None
    readout_layer: int
    layers: list[int]
    device: torch.device
    calibrated: bool
    name_by_id: dict[int, str]
    emission: dict[str, int]  # category name -> emission token id
    yes_id: int
    no_id: int
    yesno_note: dict[str, Any]
    samples: list[Sample]  # the caption pool (empty when --tasks omits captions)
    lens_meta: dict[str, Any]
    bias_meta: dict[str, Any]


def default_layers(args: argparse.Namespace, kind: str) -> list[int]:
    """The trajectory's band layers (x23's default_layers, report/modulation grid)."""
    if args.layers is not None:
        return parse_layers(args.layers)
    n_layers = int(registry.MODEL_SPECS[kind]["n_layers"])
    return list(registry.band_layers(kind, n_layers))


def resolve_ablate_layer(args: argparse.Namespace, model: LensModel, kind: str) -> int:
    """The prior-ablate layer: --ablate-layer or the kind's x22 top1-onset default."""
    layer = (
        args.ablate_layer
        if args.ablate_layer is not None
        else ABLATE_LAYER_DEFAULTS.get(kind)
    )
    if layer is None:
        raise ValueError(
            f"no --ablate-layer default for kind {kind!r}; pass --ablate-layer explicitly"
        )
    if not 0 <= layer < model.n_layers:
        raise ValueError(
            f"--ablate-layer {layer} outside 0..{model.n_layers - 1} (backend "
            f"{args.backend!r}, n_layers={model.n_layers}); the tiny fixture must pass "
            "--ablate-layer within range"
        )
    return int(layer)


def build_workspace(args: argparse.Namespace, tasks: list[str]) -> Workspace:
    """One model load + lens/bias loads + category tables (x23's build_workspace).

    The prior-ablate direction needs NO lens (raw residual difference), so a missing
    lens only degrades the trajectory readout to the raw/identity fallback; the load
    failures are recorded, never fatal, exactly as in x23.
    """
    kind = registry.kind_for(args)
    layers = default_layers(args, kind)
    model = registry.load_lens_model(args)
    invalid = [layer for layer in layers if not 0 <= layer < model.n_layers]
    if invalid:
        raise ValueError(
            f"layers {invalid} outside 0..{model.n_layers - 1} (backend "
            f"{args.backend!r} model {kind!r}, n_layers={model.n_layers}); "
            "pass --layers within range"
        )
    resolve_ablate_layer(args, model, kind)
    final_layer = model.n_layers - 1
    device = model.unembed_weight().device

    source_lens, lens_meta = load_lens_optional(args)
    bias, scale, temp, logit_bias, bias_meta = load_bias_optional(args)
    bias, scale, temp, logit_bias, bias_meta = drop_mismatched_bias(
        model, kind, args, bias, scale, temp, logit_bias, bias_meta
    )
    # Same fail-closed class as x23: a lens fitted on another checkpoint would crash
    # transport's residual @ J.T, so a d_model mismatch takes the documented no-lens
    # fallback (identity over ALL layers, raw final-layer readout) with the drop recorded.
    if source_lens is not None and source_lens.d_model != model.d_model:
        print(
            f"note: dropping lens at {args.lens_dir} for model {kind!r}: source "
            f"d_model {source_lens.d_model} != model d_model {model.d_model}; "
            "identity lens over ALL layers, raw final-layer readout"
        )
        lens_meta = {
            **lens_meta, "loaded": False, "dropped": "dim_mismatch",
            "model_d_model": int(model.d_model),
            "note": f"lens d_model {source_lens.d_model} != model d_model "
                    f"{model.d_model}; dropped, identity fallback",
        }
        source_lens = None
    if source_lens is not None:
        lens = identity_lens(source_lens, device)
        readout_layer = max(lens.source_layers)
        lens_meta = {
            **lens_meta,
            "mask": args.mask,
            "n_prompts": int(source_lens.n_prompts),
            "source_layers": list(lens.source_layers),
            "fingerprint": lens_fingerprint(lens),
            "fingerprint_note": FINGERPRINT_NOTE,
        }
    else:
        lens = JacobianLens(
            jacobians={
                layer: torch.eye(model.d_model, dtype=torch.float32, device=device)
                for layer in range(model.n_layers)
            },
            n_prompts=0,
            d_model=model.d_model,
        )
        lens_meta = {
            **lens_meta,
            "identity_over_all_layers": True,
            "source_layers": list(lens.source_layers),
            "fingerprint": lens_fingerprint(lens),
            "fingerprint_note": FINGERPRINT_NOTE,
        }
        readout_layer = final_layer
    bias = {layer: t.to(device) for layer, t in bias.items()} if bias else None
    logit_bias = (
        {layer: t.to(device) for layer, t in logit_bias.items()} if logit_bias else None
    )
    calibrated = bias is not None and readout_layer in bias

    name_by_id = name_map(args.annotations, args.synthetic)
    all_names = sorted({name for name in name_by_id.values()})
    emission = dict(zip(all_names, _category_token_ids(model.tokenizer, all_names)))
    yes_id, no_id, yesno_note = select_yes_no_ids(model.tokenizer)

    samples: list[Sample] = []
    if "captions" in tasks:
        if args.synthetic:
            samples = synthetic_samples(args.n_images)
        else:
            samples = [
                sample
                for sample in coco_samples(
                    args.images_dir, args.annotations, args.n_images, args.seed,
                    full=False,
                )
                if sample.gt_ids
            ]
    return Workspace(
        kind=kind, model=model, lens=lens, bias=bias, scale=scale, temp=temp,
        logit_bias=logit_bias, readout_layer=readout_layer, layers=layers,
        device=device, calibrated=calibrated, name_by_id=name_by_id,
        emission=emission, yes_id=yes_id, no_id=no_id, yesno_note=yesno_note,
        samples=samples, lens_meta=lens_meta, bias_meta=bias_meta,
    )


# ------------------------------------------------------- contrast & edit machinery


class ContrastLogitsProcessor(LogitsProcessor):
    """One per-step constant subtraction: ``scores - tau * contrast``.

    The contrast vector is the contrast prompt's final-position logits, computed once
    per sample and reused at every decoding step (CONTRAST_NOTE: exact at the handoff
    decision, an approximation afterwards — the documented x25 choice). Math is done
    in float32 and cast back, so a bf16 generate path does not quantize the contrast.
    """

    def __init__(self, contrast: torch.Tensor, tau: float) -> None:
        if tau < 0:
            raise ValueError(f"tau must be >= 0, got {tau}")
        self.contrast = contrast.detach().float().flatten()
        self.tau = float(tau)
        self.steps = 0  # observed: how often generate() invoked the processor

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        dtype = scores.dtype
        adjusted = scores.float() - self.tau * self.contrast.to(
            device=scores.device, dtype=torch.float32
        )
        self.steps += 1
        return adjusted.to(dtype)


class DirectionEditor(ResidualEditor):
    """ResidualEditor whose target vectors are MEASURED directions, not token rows.

    The prior-ablate direction ``v_l = normalize(h_l(image) - h_l(no image))`` is a
    state difference at the handoff positions, so ``ResidualEditor``'s token-string
    resolution (``_vectors_for`` -> ``lens_vectors``) does not apply; its hook
    machinery is reused verbatim (per-layer forward hooks, ``apply_edit`` ablate
    semantics, positions scopes) with the measured vector passed as
    ``vectors["target"]``.
    """

    def __init__(
        self,
        model: LensModel,
        grouped: dict[int, list[tuple[ResidualEdit, dict[str, torch.Tensor]]]],
    ) -> None:
        self._model = model
        self._grouped = grouped
        self._handles: list[Any] = []
        self.forward_calls = 0


def prior_edit_spec(layer: int, alpha: float) -> ResidualEdit:
    """The JSON-reportable edit spec for one prior-ablate arm."""
    return ResidualEdit(
        layer=layer, mode="ablate", alpha=alpha, token=PRIOR_TOKEN_LABEL,
        positions="last",
    )


def ablate_editor(
    model: LensModel, layer: int, direction: torch.Tensor, alpha: float
) -> DirectionEditor:
    """The editor applying ``h - alpha * <h, v> v`` at the decision positions."""
    spec = prior_edit_spec(layer, alpha)
    return DirectionEditor(model, {layer: [(spec, {"target": direction})]})


# --------------------------------------------------------------- forward helpers


@torch.no_grad()
def forward_final(
    model: LensModel, batch: Any, *, layer: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """(final-position logits, block-output residual at ``layer``) in one forward.

    Both tensors come from the SAME pass over ``batch``: the LM head's last-position
    logits (the contrast vectors' source) and the raw block-output residual at the
    handoff position (the prior direction's source, pre-final-norm, the lens-native
    tensor). Returns CPU float32 vectors.
    """
    with ActivationRecorder(model.layers, at=[layer]) as recorder:
        output = model.hf_model(**batch.hf_kwargs(), use_cache=False)
    logits = output.logits[0, -1].detach().float().cpu()
    hidden = recorder.activations[layer][0, -1].detach().float().cpu()
    return logits, hidden


@torch.no_grad()
def arm_generate(
    model: LensModel,
    batch: Any,
    *,
    max_new_tokens: int,
    logits_processor: LogitsProcessor | None = None,
    editor: ResidualEditor | None = None,
) -> dict[str, Any]:
    """Greedy generation shared by every arm (do_sample=False, KV cache on).

    ``editor`` (prior-ablate) or ``logits_processor`` (m3id/vcd) may be active; the
    plain baseline passes neither. Returns the generated ids/text plus the arm's
    intervention bookkeeping (processor steps, editor forwards).
    """
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens, "do_sample": False, "use_cache": True,
    }
    if logits_processor is not None:
        kwargs["logits_processor"] = LogitsProcessorList([logits_processor])
    ctx = editor if editor is not None else contextlib.nullcontext()
    with ctx:
        generated = model.hf_model.generate(**batch.hf_kwargs(), **kwargs)
    prompt_len = int(batch.input_ids.shape[1])
    new_tokens = generated[0, prompt_len:].detach().cpu()
    return {
        "token_ids": [int(t) for t in new_tokens],
        "text": model._decode(new_tokens),
        "n_generated": int(new_tokens.shape[0]),
        "processor_steps": getattr(logits_processor, "steps", 0),
        "editor_forwards": getattr(editor, "forward_calls", 0),
    }


def distort_image(
    image: Any, *, noise_sigma: float, blur_radius: float, rng: np.random.Generator
) -> Any:
    """The VCD contrast image: gaussian noise + mild blur, PIL in / PIL out.

    numpy works on the RGB array directly and PIL's GaussianBlur closes the pipeline —
    no torchvision anywhere (the repo's PIL-image invariant).
    """
    from PIL import Image, ImageFilter

    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    noisy = array + rng.normal(0.0, noise_sigma, size=array.shape)
    uint8 = np.clip(noisy, 0, 255).astype(np.uint8)
    return Image.fromarray(uint8).filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))


# --------------------------------------------------------------- sample plumbing


@dataclass(frozen=True)
class PopeItem:
    """One POPE question: image + question + gold label."""

    question_id: str
    question: str
    image: Any  # Path (real COCO) or PIL.Image (synthetic)
    image_name: str
    label: str  # "yes" | "no"


def pope_split_file(pope_dir: str, split: str) -> Path:
    """The split's jsonl: standard ``coco_pope_<split>.jsonl``, else any ``*<split>.jsonl``."""
    base = Path(pope_dir)
    exact = base / f"coco_pope_{split}.jsonl"
    if exact.is_file():
        return exact
    matches = sorted(base.glob(f"*_{split}.jsonl"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"no POPE jsonl for split {split!r} under {pope_dir} "
        f"(looked for coco_pope_{split}.jsonl and *_{split}.jsonl)"
    )


def synthetic_pope_items(
    samples: list[Sample], name_by_id: dict[int, str], split: str, *, seed: int
) -> list[PopeItem]:
    """In-memory POPE-style items (SYNTH_NOTE): balanced yes/no per image.

    2 present + 2 absent questions per image ('Is there a <name> in the image?');
    the absent pool is uniform over the non-annotated categories, so the split NAME
    only exercises the per-split plumbing — no hard-negative semantics on tiny.
    """
    pool = sorted({name for name in name_by_id.values()})
    split_index = POPE_SPLITS.index(split) if split in POPE_SPLITS else 0
    items: list[PopeItem] = []
    for index, sample in enumerate(samples):
        gt = gt_name_set(sample, name_by_id)
        present = sorted(gt)
        absent_pool = [name for name in pool if name not in gt]
        rng = random.Random(seed * 7919 + index * 104729 + split_index)
        for k in range(2):
            name = present[k % len(present)]
            items.append(PopeItem(
                question_id=f"{split}-{sample.image_id}-y{k}",
                question=f"Is there a {name} in the image?",
                image=sample.image, image_name=sample.name, label="yes",
            ))
        for k in range(2):
            name = absent_pool[rng.randrange(len(absent_pool))]
            items.append(PopeItem(
                question_id=f"{split}-{sample.image_id}-n{k}",
                question=f"Is there a {name} in the image?",
                image=sample.image, image_name=sample.name, label="no",
            ))
    return items


def real_pope_items(
    pope_dir: str, split: str, images_dir: str, *, n_images: int
) -> tuple[list[PopeItem], int]:
    """The split's shipped records, capped at the first ``n_images`` (POPE_SLICE_NOTE).

    Returns the items plus the number of records skipped for missing image files
    (those images cannot be encoded; they are counted, never silently dropped).
    """
    records = read_pope_jsonl(pope_split_file(pope_dir, split))
    items: list[PopeItem] = []
    n_missing = 0
    for count, record in enumerate(records[:n_images]):
        label = str(record.get("label", "")).strip().lower()
        if label not in ("yes", "no"):
            raise ValueError(
                f"POPE record {record.get('question_id', count)!r}: label "
                f"{record.get('label')!r} is not yes/no"
            )
        image_path = Path(images_dir) / str(record["image"])
        if not image_path.is_file():
            n_missing += 1
            continue
        items.append(PopeItem(
            question_id=str(record.get("question_id", count)),
            question=str(record.get("text") or record.get("question") or ""),
            image=image_path, image_name=str(record["image"]), label=label,
        ))
    return items, n_missing


def parse_leading_yesno(text: str) -> str:
    """The leading yes/no token verdict (POPE_PARSE_NOTE): 'yes' | 'no' | 'unparsed'."""
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return "unparsed"
    if words[0] == "yes":
        return "yes"
    if words[0] == "no":
        return "no"
    return "unparsed"


def pope_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Accuracy / yes-class precision-recall-F1 / yes-rate over scored records.

    Unparsed predictions are neither yes nor no: they count as incorrect (standard
    POPE treats every item) and surface in ``n_unparsed`` so the failure mode is
    visible instead of folded into a silent tie-break.
    """
    n = len(rows)
    if n == 0:
        return {"n": 0}
    n_yes = sum(1 for row in rows if row["pred"] == "yes")
    n_unparsed = sum(1 for row in rows if row["pred"] == "unparsed")
    tp = sum(1 for row in rows if row["pred"] == "yes" and row["label"] == "yes")
    fp = sum(1 for row in rows if row["pred"] == "yes" and row["label"] == "no")
    fn = sum(1 for row in rows if row["pred"] != "yes" and row["label"] == "yes")
    # Unparsed never counts as correct: tn is a genuine parsed-"no" on a no label
    # (an unparsed record on a no label lands in no confusion cell; accuracy still
    # counts it as incorrect because accuracy = (tp + tn) / n).
    tn = sum(1 for row in rows if row["pred"] == "no" and row["label"] == "no")
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    return {
        "n": n,
        "n_yes": n_yes,
        "n_no": n - n_yes - n_unparsed,
        "n_unparsed": n_unparsed,
        "yes_rate": _round(n_yes / n),
        "accuracy": _round((tp + tn) / n),
        "precision": _round(precision),
        "recall": _round(recall),
        "f1": _round(f1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }



# -------------------------------------------------------------- shared arm context


@dataclass
class SampleContext:
    """One sample's shared encodes/forwards: computed once, consumed by every arm.

    The contrast prompts are forwarded ONCE per sample (their final-position logits
    and handoff residuals are reused across the whole arm grid), so the per-arm cost
    is the generate call alone.
    """

    image_batch: Any
    prior_logits: torch.Tensor | None = None  # m3id contrast (image-free final logits)
    vcd_logits: torch.Tensor | None = None  # vcd contrast (distorted final logits)
    direction: torch.Tensor | None = None  # prior-ablate direction v_l
    direction_norm: float | None = None


@torch.no_grad()
def build_sample_context(
    args: argparse.Namespace,
    ws: Workspace,
    arms: list[ArmSpec],
    ablate_layer: int,
    image: Any,
    question: str,
    *,
    sample_index: int,
) -> SampleContext:
    """Encode + forward the sample's prompts once for the whole arm grid.

    Three forwards, each reused by every arm that needs it:
      image-free text  -> prior_logits (m3id) AND the no-image residual (prior-ablate);
      distorted image  -> vcd_logits (vcd); skipped when no vcd arm runs;
      image prompt     -> the with-image residual (prior-ablate) — the generation
                          prefill itself is redone inside each generate() call.
    """
    kind = ws.kind
    prompt = registry.image_prompt(kind, question)
    image_batch = registry.image_encode(
        ws.model, kind, prompt, image, max_length=args.max_seq_len
    )
    text_batch = registry.chat_encode(ws.model, kind, question, max_length=args.max_seq_len)
    ctx = SampleContext(image_batch=image_batch)
    want_prior = any(arm.family == "m3id" for arm in arms) or any(
        arm.family == "prior-ablate" for arm in arms
    )
    want_vcd = any(arm.family == "vcd" for arm in arms)
    want_image_hidden = any(arm.family == "prior-ablate" for arm in arms)
    h_no_image: torch.Tensor | None = None
    if want_prior or want_image_hidden:
        prior_logits, h_no_image = forward_final(ws.model, text_batch, layer=ablate_layer)
        ctx.prior_logits = prior_logits
    if want_vcd:
        rng = np.random.default_rng(args.seed * 1_000_003 + sample_index)
        distorted = distort_image(
            image, noise_sigma=args.vcd_noise_sigma, blur_radius=args.vcd_blur_radius,
            rng=rng,
        )
        dist_batch = registry.image_encode(
            ws.model, kind, prompt, distorted, max_length=args.max_seq_len
        )
        ctx.vcd_logits, _ = forward_final(ws.model, dist_batch, layer=ablate_layer)
    if want_image_hidden:
        _, h_image = forward_final(ws.model, image_batch, layer=ablate_layer)
        assert h_no_image is not None  # want_image_hidden implies the text forward ran
        difference = h_image - h_no_image
        norm = float(difference.norm())
        ctx.direction_norm = norm
        ctx.direction = difference / max(norm, 1e-8)
    return ctx


def run_arms_on_context(
    arms: list[ArmSpec],
    ws: Workspace,
    ablate_layer: int,
    ctx: SampleContext,
    *,
    max_new_tokens: int,
) -> dict[str, dict[str, Any]]:
    """Every arm's greedy generation for one sample, from the shared context."""
    out: dict[str, dict[str, Any]] = {}
    for arm in arms:
        if arm.family == "baseline":
            result = arm_generate(ws.model, ctx.image_batch, max_new_tokens=max_new_tokens)
        elif arm.family in ("m3id", "vcd"):
            contrast = ctx.prior_logits if arm.family == "m3id" else ctx.vcd_logits
            assert contrast is not None, f"{arm.arm_id}: contrast forward missing"
            processor = ContrastLogitsProcessor(contrast, arm.tau or 0.0)
            result = arm_generate(
                ws.model, ctx.image_batch, max_new_tokens=max_new_tokens,
                logits_processor=processor,
            )
        else:
            assert ctx.direction is not None, "prior-ablate direction missing"
            editor = ablate_editor(ws.model, ablate_layer, ctx.direction, arm.alpha or 1.0)
            result = arm_generate(
                ws.model, ctx.image_batch, max_new_tokens=max_new_tokens, editor=editor,
            )
        out[arm.arm_id] = result
    return out


# ------------------------------------------------------------------ CHAIR metrics


def mention_patterns(name_by_id: dict[int, str]) -> dict[str, re.Pattern[str]]:
    """One word-boundary regex per category name (MATCHER_NOTE: exact + plural)."""
    return {
        name: re.compile(r"\b" + re.escape(name.lower()) + r"(?:s|es)?\b")
        for name in sorted({name for name in name_by_id.values()})
    }


def mentioned_categories(caption: str, patterns: dict[str, re.Pattern[str]]) -> set[str]:
    """The category names whose pattern matches the lowercased caption."""
    text = caption.lower()
    return {name for name, pattern in patterns.items() if pattern.search(text)}


def chair_metrics(
    rows: list[dict[str, Any]], patterns: dict[str, re.Pattern[str]]
) -> dict[str, Any]:
    """CHAIR-s / CHAIR-r / lengths over one arm's captions (CHAIR_NOTE).

    Each row: {"image": name, "caption": text, "token_ids": [...], "gt": set[str]}.
    ``hallucinated_counts`` (category -> n captions hallucinating it) is kept — the
    microscope's "what does the model invent" readout.
    """
    if not rows:
        return {"n_captions": 0}
    sentence_flags: list[float] = []
    recalls: list[float] = []
    word_lengths: list[float] = []
    token_lengths: list[float] = []
    hallucinated_counts: dict[str, int] = {}
    n_mentions = 0
    per_image: dict[str, dict[str, Any]] = {}
    for row in rows:
        mentioned = mentioned_categories(row["caption"], patterns)
        gt = row["gt"]
        hallucinated = mentioned - gt
        grounded = mentioned & gt
        n_mentions += len(mentioned)
        sentence_flags.append(1.0 if hallucinated else 0.0)
        if gt:
            recalls.append(len(grounded) / len(gt))
        word_lengths.append(float(len(row["caption"].split())))
        token_lengths.append(float(len(row["token_ids"])))
        for name in sorted(hallucinated):
            hallucinated_counts[name] = hallucinated_counts.get(name, 0) + 1
        per_image[row["image"]] = {
            "caption": row["caption"],
            "mentioned": sorted(mentioned),
            "hallucinated": sorted(hallucinated),
        }
    return {
        "n_captions": len(rows),
        "chair_s": _round(_mean(sentence_flags)),
        "chair_r": _round(_mean(recalls)),
        "mean_caption_words": _round(_mean(word_lengths), 2),
        "mean_caption_tokens": _round(_mean(token_lengths), 2),
        "n_mentioned_total": n_mentions,
        "n_hallucinated_total": sum(hallucinated_counts.values()),
        "hallucinated_counts": hallucinated_counts,
        "per_image": per_image,
    }


# ---------------------------------------------------------------- lens trajectory


@torch.no_grad()
def trajectory_stats(
    args: argparse.Namespace,
    ws: Workspace,
    image_batch: Any,
    token_ids: list[int],
    object_ids: list[int],
    *,
    chunk: int = 32,
) -> dict[str, dict[str, float | None]]:
    """Per band layer: (mean P_yes, object-in-top-k rate) at the caption positions.

    The caption is teacher-forced onto the prompt (one forward, no edits active —
    post-hoc by design, TRAJECTORY_NOTE) and the calibrated J-lens readout runs on the
    caption-position residual rows only. Empty captions return {} (nothing to read).
    """
    if not token_ids:
        return {}
    generated = torch.tensor([token_ids], dtype=image_batch.input_ids.dtype,
                             device=image_batch.input_ids.device)
    fields: dict[str, Any] = {
        "input_ids": torch.cat([image_batch.input_ids, generated], dim=1),
        "attention_mask": torch.cat(
            [image_batch.attention_mask, torch.ones_like(generated)], dim=1
        ),
        "image_token_mask": torch.cat(
            [image_batch.image_token_mask,
             torch.zeros_like(generated, dtype=image_batch.image_token_mask.dtype)],
            dim=1,
        ),
    }
    labels = getattr(image_batch, "mm_token_type_ids", None)
    if labels is not None:  # qwen2vl batches: appended text tokens are label 0
        fields["mm_token_type_ids"] = torch.cat(
            [labels,
             torch.zeros((labels.shape[0], generated.shape[1]), dtype=labels.dtype,
                         device=labels.device)],
            dim=1,
        )
    extended = dataclasses.replace(image_batch, **fields)
    prompt_len = int(image_batch.input_ids.shape[1])
    caption_len = len(token_ids)
    object_tensor = torch.tensor(sorted(set(object_ids)))
    stats: dict[str, dict[str, float | None]] = {}
    with ActivationRecorder(ws.model.layers, at=ws.layers) as recorder:
        ws.model.forward_mm(extended)
    for layer in ws.layers:
        residual = recorder.activations[layer][0, prompt_len:prompt_len + caption_len]
        residual = residual.detach().float().cpu()
        p_yes_values: list[float] = []
        object_hits: list[float] = []
        for start in range(0, caption_len, chunk):
            block = residual[start:start + chunk]
            logits = _calibrated_lens_logits(
                ws.model, ws.lens, block, layer, ws.bias, ws.scale, ws.temp,
                ws.logit_bias,
            )
            probabilities = torch.softmax(logits, dim=-1)
            p_yes_values.extend(probabilities[:, ws.yes_id].tolist())
            top = logits.topk(min(args.top_k, logits.shape[-1]), dim=-1)
            hits = torch.isin(top.indices, object_tensor.unsqueeze(0)).any(dim=-1)
            object_hits.extend(hits.float().tolist())
        stats[str(layer)] = {
            "p_yes_mean": _round(_mean(p_yes_values)),
            "object_topk_rate": _round(_mean(object_hits)),
            "n_positions": caption_len,
        }
    return stats


def trajectory_table(
    per_arm: dict[str, dict[str, list[dict[str, float | None]]]],
) -> dict[str, Any]:
    """Aggregate the per-image trajectory stats and pair them with the baseline.

    ``per_arm[arm][layer]`` holds one stats dict per image. Output: per-layer arm
    means plus the per-layer shift vs baseline (same images on both sides — a paired
    difference).
    """
    means: dict[str, dict[str, dict[str, float | None]]] = {}
    for arm, by_layer in per_arm.items():
        means[arm] = {}
        for layer, rows in by_layer.items():
            entry: dict[str, float | None] = {}
            for stat in ("p_yes_mean", "object_topk_rate"):
                values = [row[stat] for row in rows if row.get(stat) is not None]
                entry[stat] = _round(_mean(values)) if values else None
            entry["n_captions"] = len(rows)
            means[arm][layer] = entry
    shifts: dict[str, dict[str, dict[str, float | None]]] = {}
    baseline = means.get(BASELINE_ARM, {})
    for arm in per_arm:
        if arm == BASELINE_ARM:
            continue
        shifts[arm] = {}
        for layer in per_arm[arm]:
            base = baseline.get(layer, {})
            arm_row = means[arm][layer]
            shift: dict[str, float | None] = {}
            for stat in ("p_yes_mean", "object_topk_rate"):
                if arm_row.get(stat) is None or base.get(stat) is None:
                    shift[stat] = None
                else:
                    shift[stat] = _round(arm_row[stat] - base[stat])
            shifts[arm][layer] = shift
    return {"per_layer": means, "shift_vs_baseline": shifts}




# ------------------------------------------------------------------ pope runner


def parse_splits(raw: str) -> list[str]:
    """The --pope-splits switch: a non-empty subset of POPE_SPLITS, canonical order."""
    picked = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [split for split in picked if split not in POPE_SPLITS]
    if unknown:
        raise ValueError(
            f"--pope-splits: unknown split(s) {unknown}; choices are {list(POPE_SPLITS)}"
        )
    if not picked:
        raise ValueError("--pope-splits: at least one split is required")
    return [split for split in POPE_SPLITS if split in picked]


def run_pope(
    args: argparse.Namespace, ws: Workspace, arms: list[ArmSpec], ablate_layer: int
) -> dict[str, Any]:
    """Every arm on every requested POPE split (accuracy/F1/yes-rate per arm).

    One shared SampleContext per record (contrast forwards amortized over the whole
    arm grid); the answer is the leading yes/no token of each arm's generation.
    """
    splits = parse_splits(args.pope_splits)
    per_split: dict[str, dict[str, Any]] = {}
    for split in splits:
        if args.synthetic:
            items = synthetic_pope_items(
                synthetic_samples(args.n_images), ws.name_by_id, split, seed=args.seed
            )
            n_missing = 0
        else:
            items, n_missing = real_pope_items(
                args.pope_dir, split, args.images_dir, n_images=args.n_images
            )
        scored: dict[str, list[dict[str, str]]] = {arm.arm_id: [] for arm in arms}
        steps: dict[str, int] = {arm.arm_id: 0 for arm in arms}
        edits: dict[str, int] = {arm.arm_id: 0 for arm in arms}
        generated_total: dict[str, int] = {arm.arm_id: 0 for arm in arms}
        for index, item in enumerate(items):
            ctx = build_sample_context(
                args, ws, arms, ablate_layer, image=item.image,
                question=item.question, sample_index=index,
            )
            generations = run_arms_on_context(
                arms, ws, ablate_layer, ctx, max_new_tokens=args.pope_max_new_tokens
            )
            for arm in arms:
                generated = generations[arm.arm_id]
                scored[arm.arm_id].append({
                    "pred": parse_leading_yesno(generated["text"]),
                    "label": item.label,
                })
                steps[arm.arm_id] += generated["processor_steps"]
                edits[arm.arm_id] += generated["editor_forwards"]
                generated_total[arm.arm_id] += generated["n_generated"]
            if (index + 1) % 10 == 0 or index + 1 == len(items):
                print(f"  [{split} {index + 1}/{len(items)}] scored", flush=True)
        per_split[split] = {
            "n_records": len(items),
            "n_missing_images": n_missing,
            "n_yes_labels": sum(1 for item in items if item.label == "yes"),
            "arms": {
                arm.arm_id: {
                    **pope_metrics(scored[arm.arm_id]),
                    "n_generated_total": generated_total[arm.arm_id],
                    "n_processor_steps": steps[arm.arm_id],
                    "n_editor_forwards": edits[arm.arm_id],
                }
                for arm in arms
            },
        }
    return {
        "splits": per_split,
        "pope_parse": POPE_PARSE_NOTE,
        "pope_slice": POPE_SLICE_NOTE,
        "params": {
            "pope_dir": None if args.synthetic else str(args.pope_dir),
            "pope_splits": splits,
            "pope_max_new_tokens": args.pope_max_new_tokens,
            "n_images_cap": args.n_images,
        },
    }


# -------------------------------------------------------------- captions runner


def run_captions(
    args: argparse.Namespace, ws: Workspace, arms: list[ArmSpec], ablate_layer: int
) -> dict[str, Any]:
    """Every arm's captions over the seeded image pool + CHAIR + the lens trajectory."""
    if not ws.samples:
        raise ValueError(
            "captions task: the image pool is empty (synthetic corpus or filtered COCO "
            "pool); check --n-images/--images-dir/--annotations"
        )
    patterns = mention_patterns(ws.name_by_id)
    rows_by_arm: dict[str, list[dict[str, Any]]] = {arm.arm_id: [] for arm in arms}
    traj_by_arm: dict[str, dict[str, list[dict[str, float | None]]]] = {
        arm.arm_id: {str(layer): [] for layer in ws.layers} for arm in arms
    }
    n_skipped_trajectory = {arm.arm_id: 0 for arm in arms}
    steps: dict[str, int] = {arm.arm_id: 0 for arm in arms}
    edits: dict[str, int] = {arm.arm_id: 0 for arm in arms}
    generated_total: dict[str, int] = {arm.arm_id: 0 for arm in arms}
    for index, sample in enumerate(ws.samples):
        gt_names = gt_name_set(sample, ws.name_by_id)
        object_ids = [
            ws.emission[name] for name in sorted(gt_names) if name in ws.emission
        ]
        ctx = build_sample_context(
            args, ws, arms, ablate_layer, image=sample.image,
            question=CAPTION_QUESTION, sample_index=index,
        )
        generations = run_arms_on_context(
            arms, ws, ablate_layer, ctx, max_new_tokens=args.max_new_tokens
        )
        for arm in arms:
            generated = generations[arm.arm_id]
            rows_by_arm[arm.arm_id].append({
                "image": sample.name,
                "caption": generated["text"],
                "token_ids": generated["token_ids"],
                "gt": gt_names,
            })
            steps[arm.arm_id] += generated["processor_steps"]
            edits[arm.arm_id] += generated["editor_forwards"]
            generated_total[arm.arm_id] += generated["n_generated"]
            stats = trajectory_stats(
                args, ws, ctx.image_batch, generated["token_ids"], object_ids
            )
            if stats:
                for layer, row in stats.items():
                    traj_by_arm[arm.arm_id][layer].append(row)
            else:
                n_skipped_trajectory[arm.arm_id] += 1
        if (index + 1) % 10 == 0 or index + 1 == len(ws.samples):
            print(f"  [{index + 1}/{len(ws.samples)}] {sample.name} captioned", flush=True)
    return {
        "n_images": len(ws.samples),
        "question": CAPTION_QUESTION,
        "arms": {
            arm.arm_id: {
                "metrics": chair_metrics(rows_by_arm[arm.arm_id], patterns),
                "n_empty_captions": sum(
                    1 for row in rows_by_arm[arm.arm_id] if not row["caption"].strip()
                ),
                "n_skipped_trajectory": n_skipped_trajectory[arm.arm_id],
                "n_generated_total": generated_total[arm.arm_id],
                "n_processor_steps": steps[arm.arm_id],
                "n_editor_forwards": edits[arm.arm_id],
            }
            for arm in arms
        },
        "params": {
            "max_new_tokens": args.max_new_tokens,
            "trajectory_layers": ws.layers,
            "top_k": args.top_k,
        },
        "traj_by_arm": traj_by_arm,
    }


# ----------------------------------------------------------------------- digest


def build_digest(
    args: argparse.Namespace, ws: Workspace, arms: list[ArmSpec], report: dict[str, Any]
) -> str:
    """The ASCII run summary: one row per arm, the headline metrics only."""
    kind = ws.kind
    lines = [
        f"== X25 mitigation microscope: backend={args.backend} model={kind} "
        f"seed={args.seed} tasks={report['tasks']} ==",
        f"layers={ws.layers} ablate_layer={report['ablate_layer']} "
        f"calibrated={ws.calibrated} n_images={args.n_images}",
    ]
    if "pope" in report["tasks"]:
        lines.append("")
        lines.append(
            "-- POPE (leading yes/no token; unparsed = incorrect) --            "
            "acc    F1   yes%    n  unparsed"
        )
        for arm in arms:
            for split, entry in report["arms"][arm.arm_id].get("pope", {}).items():
                metrics = entry
                lines.append(
                    f"{arm.arm_id:<28} {split:<11} {_fmt(metrics.get('accuracy'), 6)}"
                    f"{_fmt(metrics.get('f1'), 6)}{_fmt(metrics.get('yes_rate'), 6)}"
                    f"{metrics.get('n', 0):>6}{metrics.get('n_unparsed', 0):>10}"
                )
    if "captions" in report["tasks"]:
        lines.append("")
        lines.append(
            "-- captions (CHAIR, exact+plural matcher) --                     "
            "CHAIRs  CHAIRr  words  n_hall"
        )
        for arm in arms:
            metrics = report["arms"][arm.arm_id].get("captions", {}).get("metrics", {})
            if not metrics:
                continue
            lines.append(
                f"{arm.arm_id:<28} {'':<11} {_fmt(metrics.get('chair_s'), 6)}"
                f"{_fmt(metrics.get('chair_r'), 7)}"
                f"{_fmt(metrics.get('mean_caption_words'), 7, 1)}"
                f"{metrics.get('n_hallucinated_total', 0):>7}"
            )
        lines.append("")
        lines.append(
            "-- lens trajectory shift vs baseline (caption positions; mean over "
            "band layers) --"
        )
        lines.append("arm                           dP_yes   dObj10")
        trajectory = report.get("lens_trajectory", {})
        shifts = trajectory.get("shift_vs_baseline", {})
        for arm in arms:
            if arm.arm_id == BASELINE_ARM or arm.arm_id not in shifts:
                continue
            p_shifts = [
                row["p_yes_mean"] for row in shifts[arm.arm_id].values()
                if row.get("p_yes_mean") is not None
            ]
            o_shifts = [
                row["object_topk_rate"] for row in shifts[arm.arm_id].values()
                if row.get("object_topk_rate") is not None
            ]
            lines.append(
                f"{arm.arm_id:<28} {_fmt(_mean(p_shifts), 7)}{_fmt(_mean(o_shifts), 8)}"
            )
    return "\n".join(lines)


def build_summary(report: dict[str, Any], arms: list[ArmSpec]) -> dict[str, Any]:
    """The run's observed headline: baseline vs the best arm per headline metric.

    'Best' is the observed optimum ON THIS RUN'S CORPUS (tiny/smoke corpora carry no
    semantics; the note travels with the summary).
    """
    summary: dict[str, Any] = {
        "n_arms": len(arms),
        "arm_ids": [arm.arm_id for arm in arms],
        "note": "best = observed optimum on this run's corpus only; the tiny backend "
                "is a random-weight fixture whose rates do not transfer",
    }
    if "pope" in report["tasks"]:
        per_split = {}
        for split, entry in next(iter(report["arms"].values()), {}).get("pope", {}).items():
            baseline = entry.get("accuracy")
            scored = [
                (arm_id, arm_pope.get("f1"))
                for arm_id, arm_entry in report["arms"].items()
                for split_name, arm_pope in arm_entry.get("pope", {}).items()
                if split_name == split
            ]
            usable = [(arm_id, f1) for arm_id, f1 in scored if f1 is not None]
            per_split[split] = {
                "baseline_accuracy": baseline,
                "best_f1_arm": max(usable, key=lambda pair: pair[1])[0] if usable else None,
                "best_f1": max((f1 for _, f1 in usable), default=None),
            }
        summary["pope"] = per_split
    if "captions" in report["tasks"]:
        scored = [
            (arm_id, metrics.get("chair_s"), metrics.get("chair_r"))
            for arm_id, entry in report["arms"].items()
            if (metrics := entry.get("captions", {}).get("metrics", {})).get("n_captions")
        ]
        usable_s = [(arm_id, s) for arm_id, s, _ in scored if s is not None]
        usable_r = [(arm_id, r) for arm_id, _, r in scored if r is not None]
        summary["captions"] = {
            "lowest_chair_s_arm": (
                min(usable_s, key=lambda pair: pair[1])[0] if usable_s else None
            ),
            "highest_chair_r_arm": (
                max(usable_r, key=lambda pair: pair[1])[0] if usable_r else None
            ),
        }
    return summary


# ------------------------------------------------------------------------- main


def main() -> int:
    args = parse_args()
    tasks = parse_tasks(args.tasks)
    if "pope" in tasks and not args.synthetic and not Path(args.pope_dir).is_dir():
        raise FileNotFoundError(
            f"--pope-dir {args.pope_dir} does not exist (or pass --synthetic / drop "
            "the pope task)"
        )
    torch.manual_seed(args.seed)
    ws = build_workspace(args, tasks)
    arms = build_arms(args)
    ablate_layer = resolve_ablate_layer(args, ws.model, ws.kind)
    print(
        f"== X25 arms={args.tasks} backend={args.backend} model={ws.kind} "
        f"device={ws.device} layers={ws.layers} ablate_layer={ablate_layer} "
        f"calibrated={ws.calibrated} =="
    )
    report: dict[str, Any] = {
        "script": Path(__file__).name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": args.backend,
        "model": ws.kind,
        "synthetic": args.synthetic,
        "tasks": tasks,
        "seed": args.seed,
        "n_images": args.n_images,
        "max_new_tokens": args.max_new_tokens,
        "pope_max_new_tokens": args.pope_max_new_tokens,
        "max_seq_len": args.max_seq_len,
        "top_k": args.top_k,
        "layers": ws.layers,
        "ablate_layer": ablate_layer,
        "ablate_layer_note": (
            "default = the kind's x22 top1-onset layer (D48 workspace band); the "
            "prior direction v_l is measured in the raw residual space at this layer"
        ),
        "m3id_taus": [arm.tau for arm in arms if arm.family == "m3id"],
        "vcd_taus": [arm.tau for arm in arms if arm.family == "vcd"],
        "ablate_alphas": [arm.alpha for arm in arms if arm.family == "prior-ablate"],
        "vcd_distortion": {
            "noise_sigma": args.vcd_noise_sigma,
            "blur_radius": args.vcd_blur_radius,
            "note": DISTORTION_NOTE,
        },
        "prompt_template": registry.image_prompt_template(ws.kind),
        "prompt_template_hash": registry.prompt_template_sha256(ws.kind),
        "lens": ws.lens_meta,
        "bias": ws.bias_meta,
        "calibrated": ws.calibrated,
        "yes_no_tokens": ws.yesno_note,
        "arms": {arm.arm_id: {"params": arm.params} for arm in arms},
    }
    if "pope" in tasks:
        pope = run_pope(args, ws, arms, ablate_layer)
        for arm in arms:
            report["arms"][arm.arm_id]["pope"] = {
                split: entry["arms"][arm.arm_id]
                for split, entry in pope["splits"].items()
            }
        report["pope_meta"] = {
            key: value for key, value in pope.items() if key != "splits"
        }
        report["pope_split_counts"] = {
            split: {
                "n_records": entry["n_records"],
                "n_missing_images": entry["n_missing_images"],
                "n_yes_labels": entry["n_yes_labels"],
            }
            for split, entry in pope["splits"].items()
        }
    if "captions" in tasks:
        captions = run_captions(args, ws, arms, ablate_layer)
        for arm in arms:
            report["arms"][arm.arm_id]["captions"] = captions["arms"][arm.arm_id]
        report["captions_meta"] = {
            key: value
            for key, value in captions.items()
            if key not in ("arms", "traj_by_arm")
        }
        report["lens_trajectory"] = trajectory_table(captions["traj_by_arm"])
        report["lens_trajectory"]["note"] = TRAJECTORY_NOTE
    report["notes"] = ARM_NOTES
    report["digest"] = build_digest(args, ws, arms, report)
    report["summary"] = build_summary(report, arms)
    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + report["digest"])
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
