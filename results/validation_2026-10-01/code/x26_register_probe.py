#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""X26: the visual-register investigation — is the sticky, object-free image band a
non-verbal visual workspace, and does it matter for hallucination?

D48 established (x23 band arm): in the workspace band the IMAGE positions are
hyper-sticky — the calibrated J-lens top-1 token repeats across neighboring image
positions at z ~ 8-20 over its permutation null — while staying OBJECT-FREE (an
image-object category sits in the position's top-10 at only 0-2% of in-band image
positions, rising to 4-12% only at the final layers). A block that is both extremely
self-similar and unreadable by the unembed looks less like "the image is decoded
everywhere" and more like a non-verbal workspace: a handful of positions — here called
VISUAL REGISTERS — that hold the image's visual state and feed the caption, with the
rest of the image block near-redundant filler.

This script tests that story in four linked measurements over one corpus pass (the 7b
caption manifest subset by default; x23's prompt-only corpus as the fallback):

registers   Per image and band layer: the calibrated J-lens top-1 at every position
            (x23 ``position_readout``), the per-image-position lag-1 stickiness over
            the image block's consecutive pairs, and its permutation null (x23's
            ``stickiness_null`` statistic tracked per position). Registers = the top
            ``--register-k`` positions by the mean-across-layers stickiness z.
            Attention wiring (``--n-attention`` samples, eager forwards): the mean
            attention mass each image position receives from the caption tokens at
            every band layer, correlated with the position's stickiness — registers
            should receive more caption attention than non-register positions.
probes      Closed-form ridge probes (leave-one-image-out) on the band-layer
            residuals with features taken at register / non-register-image / text
            positions, against per-class object presence (instances json) and, when a
            captions json is available, per-color-word presence in the image's
            ground-truth captions. The workspace claim has a two-sided signature:
            registers decode image content for a TRAINED probe well above chance
            while the UNEMBED readout at the same positions stays object-free (the
            x23 cross-check, recomputed here per position class).
ablation    Greedy captioning from the prompt-only body with the register rows zero-
            or mean-replaced at an ablate layer (``ResidualEdit(position_ids=...)`` —
            the vector-free whole-row replacement added to interventions.py; image
            positions live only in the prefill, so later decode forwards are no-ops).
            Per condition: CHAIR-s/r against the image's ground-truth objects
            (inlined exact+synonym matcher; T28's matcher was not committed when this
            was written), caption length, and the lens shift during generation
            (per-step object-in-top-10 rate, binary-softmax P(yes), first-step
            yes/no gap at the ablate layer) from x14 residual traces recorded
            POST-edit.

Shared conventions: vlm_registry dispatch (``--model``; tiny implies llava-7b
geometry), the x23 workspace loaders (identity-transport lens + bias payload,
fail-closed on dimension mismatch), the x21 lens content fingerprint, one model load
per invocation, per-image progress lines, and a tiny CPU smoke path: ``--backend tiny
--synthetic`` builds the tiny random-weight LLaVA plus synthetic images/annotations
(and a synthetic captions file so the color-probe path runs too). Images reach the
processors as PIL only — no torchvision anywhere.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["main"]


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

import torch  # noqa: E402
import vlm_registry as registry  # noqa: E402
import x23_vlm_workspace as x23  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
from jlens.lens import JacobianLens  # noqa: E402
from x13_cure_test import (  # noqa: E402
    detect_mentions,
    image_id_of,
    load_ground_truth,
)
from x14_workspace_probe import (  # noqa: E402
    _calibrated_lens_logits,
    _category_token_ids,
    generate_with_residual_traces,
    identity_lens,
)
from x18_pope_handoff import select_yes_no_ids  # noqa: E402
from x21_verbal_report import lens_fingerprint  # noqa: E402

import vlm_lens  # noqa: E402, F401  # installs the vendored jlens path: must precede jlens
from vlm_lens.artifacts import load_lens_set  # noqa: E402
from vlm_lens.data.captions import DEFAULT_QUESTIONS  # noqa: E402
from vlm_lens.data.manifest import manifest_meta, read_manifest  # noqa: E402
from vlm_lens.interventions import ResidualEdit, ResidualEditor  # noqa: E402
from vlm_lens.models.llava import MultimodalBatch  # noqa: E402

#: Any registry-built lens model (``LlavaLensModel`` for the llava kinds,
#: ``Qwen2VLLensModel`` for qwen2vl -- same public surface), so the annotations
#: stay honest without importing the qwen2vl class at module scope.
LensModel = Any

DEFAULT_LENS_DIR = (
    "/data/anhnq/vlm-lens-out/validation/fits/llava-1.5-7b-hf/merged/artifacts"
)
DEFAULT_BIAS_DIR = "/data/anhnq/vlm-lens-out/validation/step5e"
DEFAULT_MANIFEST = (
    "/data/anhnq/vlm-lens-out/validation/fits/llava-1.5-7b-hf/manifest256.jsonl"
)
DEFAULT_IMAGES_DIR = "/data/baodq/coco2014/val2014"
DEFAULT_ANNOTATIONS = "/data/baodq/coco2014/annotations/instances_val2014.json"
DEFAULT_CAPTIONS = "/data/baodq/coco2014/annotations/captions_val2014.json"

#: The register rule: per-position stickiness mirrors x23's band statistic. x23's
#: lag-1 stickiness is P(top1[p+1] == top1[p]) over consecutive same-modality pairs;
#: here the SAME pair set is scored per position (an interior image position belongs
#: to two pairs, a boundary position to one), averaged over the band layers, and the
#: null is x23's permutation of the image block's own top-1 values over its positions
#: (marginals preserved, temporal structure broken), tracked per position. The z is
#: x23's (observed - null_mean) / null_std, averaged over layers where the null has
#: spread; positions with no spread at any layer rank by their observed stickiness.
REGISTER_RULE = (
    "per image position p in the image block: stickiness(p, l) = mean over the "
    "position's consecutive image-block pairs (p-1,p) and (p,p+1) of "
    "1[top1_l == top1_l], averaged over band layers; null = permutations of the "
    "image block's top-1 values over its positions (x23 stickiness_null, per "
    "position), z(p) = mean over layers of (obs - null_mean)/null_std (layers with "
    "null spread <= 1e-12 dropped; all-spreadless positions rank by observed "
    "stickiness); registers = top --register-k positions by z (ties: observed, then "
    "lower position index)"
)

ATTENTION_RULE = (
    "one eager prefill forward with output_attentions=True per sampled image; per "
    "band layer the attention rows of the caption sources (the post-'ASSISTANT:' "
    "span of the manifest prompt+caption when the prompt prefix tokenizes as a "
    "strict prefix, else all text positions) are averaged over sources and heads, "
    "and the mass landing on each image position is correlated (Pearson, x23's "
    "helper) with that position's across-layer stickiness; the register mass ratio "
    "is mean mass at register positions / mean mass at the other image positions"
)

PROBE_RULE = (
    "closed-form ridge probe on the layer residual h_l (dual: w = X_tr^T (K + "
    "lambda I)^-1 y, K = X_tr X_tr^T), features standardized by TRAIN-fold moments; "
    "leave-one-image-out folds, held-out scores pooled over folds, AUC by average "
    "ranks; lambda = --probe-lambda-rel * mean train row norm^2; per position class "
    "(register / image_nonregister / text, --probe-rows-per-image rows per image, "
    "seeded subsample) and per target (object class present on the image per "
    "instances json; color word present in its ground-truth captions per captions "
    "json); targets with fewer than --probe-min-support positive images are skipped"
)

ABLATION_RULE = (
    "greedy captioning from the prompt-only body (the manifest sample's text minus "
    "its caption suffix; the fallback corpus's prompt verbatim) with one "
    "ResidualEdit(layer, mode='ablate', position_ids=register positions) active: "
    "mean=None zeroes the register rows at the ablate layer, mean=<corpus image-"
    "position mean residual> refills them (--ablate-mode); the edit fires only in "
    "the prefill (image positions do not recur under KV caching). Baseline = the "
    "same generation with no edits. Per condition: CHAIR-s/r vs the image's "
    "ground-truth objects, caption length (tokens/words), and the lens shift from "
    "x14 residual traces recorded post-edit at the ablate layer: per-step "
    "object-in-top-10 rate over the image's GT category emission ids, per-step "
    "binary-softmax P(yes) over the yes/no pair, and the step-0 calibrated gap "
    "z[yes] - z[no] (x18's select_yes_no_ids pair)"
)

CHAIR_RULE = (
    "inlined CHAIR-lite matcher (T28's matcher not committed at writing time): "
    "x13's detect_mentions (longest-first, word-boundary, (s|es) plural suffix, "
    "span masking so 'hot dog' does not also match 'dog') over the lowercased "
    "caption after a minimal noun-synonym fold (sofa->couch, motorbike->motorcycle, "
    "aeroplane->airplane, pottedplant->potted plant, diningtable->dining table, "
    "cellphone->cell phone, television->tv); CHAIR-s = fraction of captions with >=1 "
    "mention outside the image's GT objects; CHAIR-r = extra mentions / all mentions "
    "pooled (None when a condition produced no mentions). No hypernym expansion and "
    "irregular plurals are the documented gaps (as in x13)"
)

#: The position classes the probes/attention compare. ``image_nonregister`` = the
#: image block's positions minus the registers (the "non-register object positions").
POSITION_CLASSES: tuple[str, ...] = ("register", "image_nonregister", "text")

#: Basic color words scanned in the ground-truth captions (``grey`` folds to
#: ``gray``); each word is one binary probe target.
COLOR_WORDS: tuple[str, ...] = (
    "red", "orange", "yellow", "green", "blue", "purple", "pink",
    "brown", "black", "white", "gray", "grey",
)
COLOR_CANONICAL = {"grey": "gray"}

#: Minimal caption-noun synonym fold for the CHAIR-lite matcher (see CHAIR_RULE).
CHAIR_SYNONYMS: dict[str, str] = {
    "sofa": "couch",
    "motorbike": "motorcycle",
    "aeroplane": "airplane",
    "pottedplant": "potted plant",
    "diningtable": "dining table",
    "cellphone": "cell phone",
    "television": "tv",
}

#: Deterministic subsample seeds per position class (stable across runs).
_CLASS_SEED_SALT = {"register": 0, "image_nonregister": 1, "text": 2}


@dataclass(frozen=True)
class RegisterSample:
    """One corpus item: identity, ground truth, prompt pieces, image source."""

    name: str  # COCO file name, or "synth_%04d.jpg"
    image_id: int  # COCO image id, or the 1-based synthetic index
    image: Any  # Path for COCO, PIL.Image for synthetic (PIL-only processing)
    gt_ids: set[int]  # annotated category ids (ground truth)
    question: str  # the corpus question (manifest meta, or the default bank head)
    caption: str | None  # the model caption teacher-forced into ``text`` (manifest)
    text: str  # the encoded prompt body (prompt+caption on the manifest corpus)

    @property
    def generation_body(self) -> str:
        """The prompt-only body for generation: ``text`` minus its caption suffix.

        The manifest builds ``text = <base> + ' ' + caption`` (captions.prompt_text),
        so stripping exactly that suffix reproduces the deployment prompt without
        re-formatting; corpora without a caption generate from ``text`` verbatim.
        """
        if self.caption and self.text.endswith(" " + self.caption):
            return self.text[: -len(self.caption) - 1]
        return self.text


@dataclass
class RegisterWorkspace:
    """Everything the arms share: model kind, calibrated lens, corpus, tables.

    Field names mirror the x23 ``Workspace`` so ``x23.position_readout`` runs on this
    object verbatim (duck-typed ``ws``). ``corpus_meta``/``ablate_layers`` are set by
    :func:`build_workspace` after construction.
    """

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
    samples: list[RegisterSample]
    top_k: int
    lens_meta: dict[str, Any]
    bias_meta: dict[str, Any]


@dataclass
class RegisterContext:
    """State accumulated by the registers pass for the downstream arms."""

    records: list[dict[str, Any]] = field(default_factory=list)
    register_positions: list[list[int]] = field(default_factory=list)
    image_spans: list[tuple[int, int]] = field(default_factory=list)
    probe_features: dict[tuple[int, str], list[torch.Tensor]] = field(default_factory=dict)
    mean_image_residual: dict[int, torch.Tensor] = field(default_factory=dict)
    modality_pools: dict[tuple[int, str], dict[str, list[float]]] = field(default_factory=dict)
    class_rate_pools: dict[tuple[int, str], list[float]] = field(default_factory=dict)
    attention_records: list[dict[str, Any]] = field(default_factory=list)
    n_skipped_no_gt: int = 0


# --------------------------------------------------------------------------- args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("hf-llava", "tiny"), default="hf-llava",
        help="model backend: 'hf-llava' (default, CUDA) or the tiny CPU smoke fixture "
             "(4 layers, d_model 32; pass --layers within 0..3 and --synthetic)",
    )
    parser.add_argument(
        "--model", choices=registry.MODEL_KINDS, default=registry.DEFAULT_KIND,
        help="checkpoint kind (HF backend): " + ", ".join(registry.MODEL_KINDS)
             + "; --backend tiny implies llava-7b geometry (no tiny stand-in exists "
               "for the other kinds)",
    )
    parser.add_argument("--device", default="", help="'' -> cuda (registry convention)")
    parser.add_argument("--dtype", default=None, choices=(None, "bfloat16", "float32"))
    parser.add_argument(
        "--attn-impl", default="eager",
        help="attention implementation the model is loaded with (eager so the "
             "attention arm's output_attentions materializes; x17 convention)",
    )
    parser.add_argument("--lens-dir", default=DEFAULT_LENS_DIR,
                        help="J source lens dir: its fitted layers define the identity-"
                             "transport lens; if it cannot be loaded the identity lens "
                             "covers ALL layers and the readout falls back to the raw "
                             "final layer (x16 fallback convention)")
    parser.add_argument("--mask", default="text",
                        help="which lens-<mask>.pt and bias-<mask>.pt to use")
    parser.add_argument(
        "--bias-dir", default=DEFAULT_BIAS_DIR,
        help="directory holding bias-<mask>.pt (bias/scale, optional "
             "temp/logit_bias applied inside every calibrated readout)"
    )
    parser.add_argument("--probe-transport", action="store_true",
                        help="probe the J_l-TRANSPORTED residual rows (lens.transport) "
                             "instead of the raw h_l rows; tests whether the transport "
                             "preserves register content (default: raw rows, the D49 "
                             "numbers unchanged)")
    parser.add_argument(
        "--manifest", default=DEFAULT_MANIFEST,
        help="caption manifest (the 256-token-budget 7b fit corpus): its first "
             "--n-images prompt+caption samples are the corpus; pass '' for the "
             "x23 prompt-only corpus convention",
    )
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS,
                        help="instances_val2014.json: per-image ground-truth categories")
    parser.add_argument("--captions", default=DEFAULT_CAPTIONS,
                        help="captions_val2014.json: ground-truth captions for the "
                             "color-word probe targets (skipped with a note when "
                             "absent; --synthetic builds one instead)")
    parser.add_argument("--n-images", type=int, default=20,
                        help="corpus size (the manifest subset's first N samples)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--layers", default=None,
        help="comma-separated band layers for the readouts/probes; default is the "
             "registry sweep grid (x23 band convention; llava-7b: 12,16,20,24,28,31), "
             "falling back to every layer when the band sits below the model's depth "
             "(the tiny fixture)",
    )
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--top-k", type=int, default=10,
                        help="k for the calibrated lens top-k membership statistics")
    parser.add_argument("--register-k", type=int, default=8,
                        help="register positions kept per image (top-k by stickiness z)")
    parser.add_argument("--n-null-shuffles", type=int, default=200,
                        help="permutations per (image, layer) for the per-position "
                             "stickiness null (x23's n-null-shuffles convention)")
    parser.add_argument("--n-attention", type=int, default=5,
                        help="first N corpus images run through the eager attention "
                             "forward (0 disables the attention arm)")
    parser.add_argument("--probe-rows-per-image", type=int, default=8,
                        help="probe rows per image per position class (registers use "
                             "exactly --register-k; the other classes subsample)")
    parser.add_argument("--probe-lambda-rel", type=float, default=0.1,
                        help="ridge lambda as a fraction of the mean train row norm^2")
    parser.add_argument("--probe-min-support", type=int, default=3,
                        help="minimum images annotated with a target across the corpus")
    parser.add_argument(
        "--ablate-layers", default=None,
        help="comma-separated ablate layers for the causal arm; default is the "
             "band's midpoint layer (llava-7b: 20; tiny: the middle swept layer)",
    )
    parser.add_argument("--ablate-mode", choices=("zero", "mean"), default="zero",
                        help="register-row replacement: zeros, or the corpus "
                             "image-position mean residual at the ablate layer")
    parser.add_argument("--max-new-tokens", type=int, default=64,
                        help="greedy caption budget for the ablation arm")
    parser.add_argument("--synthetic", action="store_true",
                        help="build the corpus in-memory (tiny smoke; x23 SYNTH_NOTE)")
    parser.add_argument("--json", required=True)
    return parser.parse_args()


# ------------------------------------------------------------------ shared loaders


def load_model(args: argparse.Namespace) -> LensModel:
    """The registry loader (one model load per invocation) + the eager switch.

    The attention arm needs ``output_attentions`` to materialize, which sdpa (the
    transformers default) silently does not do — x17's convention is to load eager;
    ``set_attn_implementation`` on the instantiated model reaches the same state
    without a second load (verified on the tiny fixture).
    """
    model = registry.load_lens_model(args)
    effective = getattr(model.hf_model.config, "_attn_implementation", None)
    if args.attn_impl and effective != args.attn_impl:
        model.hf_model.set_attn_implementation(args.attn_impl)
    return model


def default_layers(args: argparse.Namespace, kind: str, n_layers: int) -> list[int]:
    """The arm's layer list: the x23 band sweep grid, or every layer below the band.

    ``registry.sweep_layers`` indexes its grid's last element, which raises when the
    model is shallower than the band (the tiny fixture: 4 layers vs band_lo 12), so
    the default falls back to all layers — the tiny smoke then runs without
    ``--layers`` while production keeps the x23 band convention. ``--layers`` wins.
    """
    if args.layers is not None:
        return [int(part) for part in args.layers.split(",") if part.strip()]
    try:
        base = list(registry.sweep_layers(kind, n_layers))
    except IndexError:  # pragma: no cover - only the tiny fixture hits this
        base = []
    if not base:
        base = list(range(n_layers))
    return base


def default_ablate_layers(
    args: argparse.Namespace, kind: str, n_layers: int, layers: list[int]
) -> list[int]:
    """The causal arm's ablate layers: the band's midpoint, clamped into the model."""
    if args.ablate_layers is not None:
        return [int(part) for part in args.ablate_layers.split(",") if part.strip()]
    lo, hi = registry.spec_band(kind)
    mid = int(round((lo + hi) / 2))
    if not 0 <= mid < n_layers:
        mid = layers[len(layers) // 2] if layers else 0
    return [mid]


def corpus_from_manifest(
    args: argparse.Namespace, kind: str
) -> tuple[list[RegisterSample], dict[str, Any]]:
    """The manifest corpus: the first --n-images prompt+caption samples, in file order.

    The manifest's build order IS the seeded image sample (phase33 convention), so
    the subset is deterministic. Image paths are re-anchored on ``--images-dir`` (the
    manifest stores the build host's absolute paths); samples whose image resolves
    nowhere are skipped with a count. Samples without the prompt+caption variant are
    used as-is (older manifests), recorded in the corpus meta.
    """
    header = manifest_meta(args.manifest)
    fits = read_manifest(args.manifest)
    prompt_caption = [
        fit for fit in fits
        if str(fit.meta.get("variant") or "prompt+caption") == "prompt+caption"
    ]
    variant_used = "prompt+caption" if prompt_caption else "as_stored"
    chosen = (prompt_caption or list(fits))[: max(args.n_images, 0)]
    _, by_image = load_ground_truth(args.annotations)
    samples: list[RegisterSample] = []
    skipped: list[str] = []
    for fit in chosen:
        if not fit.images:
            skipped.append(fit.sample_id)
            continue
        name = str(fit.meta.get("image") or Path(fit.images[0]).name)
        path = Path(args.images_dir) / name
        if not path.is_file():
            path = Path(fit.images[0])
        if not path.is_file():
            skipped.append(name)
            continue
        image_id = image_id_of(Path(name))
        samples.append(RegisterSample(
            name=name,
            image_id=image_id,
            image=path,
            gt_ids=set(by_image.get(image_id, set())),
            question=str(fit.meta.get("question") or DEFAULT_QUESTIONS[0]),
            caption=(
                str(fit.meta["caption"]) if fit.meta.get("caption") is not None else None
            ),
            text=str(fit.text),
        ))
    meta = {
        "mode": "manifest",
        "manifest": str(args.manifest),
        "manifest_n_samples": header.get("n_samples"),
        "manifest_max_new_tokens": header.get("max_new_tokens"),
        "variant_used": variant_used,
        "n_subset": len(samples),
        "n_skipped_unresolved": len(skipped),
        "subset_rule": "first --n-images samples in manifest file order",
    }
    return samples, meta


def corpus_from_images(
    args: argparse.Namespace, kind: str
) -> tuple[list[RegisterSample], dict[str, Any]]:
    """The fallback corpus: x23's seeded COCO sample + the default bank question.

    Used when ``--manifest ''`` (any manifest-less run): the x23 band arm's
    prompt-only convention, seeded selection, GT-annotated images only.
    """
    from vlm_lens.data.captions import list_coco_images, select_images

    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"images dir {images_dir} does not exist; pass --images-dir (or "
            "--synthetic for the in-memory corpus)"
        )
    _, by_image = load_ground_truth(args.annotations)
    question = DEFAULT_QUESTIONS[0]
    text = registry.image_prompt(kind, question)
    files = list_coco_images(str(images_dir))
    samples = []
    for path in select_images(files, max(args.n_images * 4, args.n_images), seed=args.seed):
        image_id = image_id_of(path)
        if not by_image.get(image_id):
            continue
        samples.append(RegisterSample(
            name=path.name,
            image_id=image_id,
            image=path,
            gt_ids=set(by_image.get(image_id, set())),
            question=question,
            caption=None,
            text=text,
        ))
        if len(samples) >= args.n_images:
            break
    meta = {
        "mode": "prompt_only",
        "manifest": None,
        "question": question,
        "n_subset": len(samples),
        "n_skipped_unresolved": 0,
        "subset_rule": "x23 coco_samples (seeded, GT-annotated only)",
    }
    return samples, meta


def synthetic_corpus(
    args: argparse.Namespace, kind: str
) -> tuple[list[RegisterSample], dict[str, Any]]:
    """The tiny smoke corpus: x23's synthetic images + prompt-only bodies."""
    question = DEFAULT_QUESTIONS[0]
    text = registry.image_prompt(kind, question)
    samples = [
        RegisterSample(
            name=sample.name,
            image_id=sample.image_id,
            image=sample.image,
            gt_ids=set(sample.gt_ids),
            question=question,
            caption=None,
            text=text,
        )
        for sample in x23.synthetic_samples(args.n_images)
    ]
    meta = {
        "mode": "synthetic",
        "manifest": None,
        "question": question,
        "n_subset": len(samples),
        "n_skipped_unresolved": 0,
        "subset_rule": "x23 synthetic_samples (deterministic random images)",
    }
    return samples, meta


def load_corpus(
    args: argparse.Namespace, kind: str
) -> tuple[list[RegisterSample], dict[str, Any]]:
    """Dispatch the corpus: manifest (default) / prompt-only fallback / synthetic."""
    if args.synthetic:
        return synthetic_corpus(args, kind)
    if args.manifest:
        return corpus_from_manifest(args, kind)
    return corpus_from_images(args, kind)


def build_workspace(args: argparse.Namespace) -> RegisterWorkspace:
    """One model load + lens/bias loads + corpus and category tables (x23 mirror).

    The lens/bias/bias-drop logic is x23's ``build_workspace`` verbatim; the corpus
    swaps in the manifest subset and the sample record grows the prompt pieces.
    Layers default from the LOADED model's depth (the tiny fixture is shallower
    than every registry band, so the sweep-grid default falls back to all layers).
    """
    kind = registry.kind_for(args)
    model = load_model(args)
    n_layers = int(model.n_layers)
    layers = default_layers(args, kind, n_layers)
    invalid = [layer for layer in layers if not 0 <= layer < model.n_layers]
    if invalid:
        raise ValueError(
            f"layers {invalid} outside 0..{model.n_layers - 1} (backend "
            f"{args.backend!r} model {kind!r}, n_layers={model.n_layers}); "
            "pass --layers within range"
        )
    final_layer = model.n_layers - 1
    source_lens, lens_meta = x23.load_lens_optional(args)
    bias, scale, temp, logit_bias, bias_meta = x23.load_bias_optional(args)
    device = model.unembed_weight().device
    bias, scale, temp, logit_bias, bias_meta = x23.drop_mismatched_bias(
        model, kind, args, bias, scale, temp, logit_bias, bias_meta
    )
    if source_lens is not None and source_lens.d_model != model.d_model:
        print(
            f"note: dropping lens at {args.lens_dir} for model {kind!r}: source "
            f"d_model {source_lens.d_model} != model d_model {model.d_model}; "
            "identity lens over ALL layers, raw final-layer readout"
        )
        lens_meta = {
            **lens_meta, "loaded": False, "dropped": "dim_mismatch",
            "model_d_model": int(model.d_model),
        }
        source_lens = None
    if source_lens is not None:
        lens = identity_lens(source_lens, device)
        readout_layer = max(lens.source_layers)
        lens_meta = {
            **lens_meta,
            "source_layers": list(lens.source_layers),
            "fingerprint": lens_fingerprint(lens),
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
        lens_meta = {**lens_meta, "identity_over_all_layers": True}
        readout_layer = final_layer
    bias = {layer: t.to(device) for layer, t in bias.items()} if bias else None
    logit_bias = (
        {layer: t.to(device) for layer, t in logit_bias.items()} if logit_bias else None
    )
    calibrated = bias is not None and readout_layer in bias

    samples, corpus_meta = load_corpus(args, kind)
    name_by_id = x23.name_map(args.annotations, args.synthetic)
    all_names = sorted({name for name in name_by_id.values()})
    emission = dict(zip(all_names, _category_token_ids(model.tokenizer, all_names)))
    ws = RegisterWorkspace(
        kind=kind, model=model, lens=lens, bias=bias, scale=scale, temp=temp,
        logit_bias=logit_bias, readout_layer=readout_layer, layers=layers,
        device=device, calibrated=calibrated, name_by_id=name_by_id,
        emission=emission, samples=samples, top_k=args.top_k,
        lens_meta=lens_meta, bias_meta=bias_meta,
    )
    ws.lens_meta["fingerprint_note"] = (
        "sha256 over the loaded lens's fitted J_l values (x21 lens_fingerprint): "
        "equal register metrics with equal fingerprints = the same fitted lens "
        "content; different fingerprints mean the readout aliased another lens"
    )
    ws.corpus_meta = corpus_meta  # type: ignore[attr-defined]
    ws.ablate_layers = default_ablate_layers(args, kind, n_layers, layers)  # type: ignore[attr-defined]
    return ws


# ------------------------------------------------------------------ stickiness


def position_stickiness(values: torch.Tensor) -> torch.Tensor:
    """Per-position lag-1 stickiness over one modality's consecutive pairs.

    ``values`` is the modality's top-1 ids in position order (x23
    ``modality_stickiness``'s ``values``). Interior positions belong to two pairs,
    boundary positions to one; the returned score averages the position's pair
    agreements. Returns ``[m]`` float (m >= 2 expected; callers guard).
    """
    m = int(values.numel())
    pair = (values[1:] == values[:-1]).float()  # pair j = positions (j, j+1)
    acc = torch.zeros(m)
    acc[:-1] += pair
    acc[1:] += pair
    counts = torch.zeros(m)
    counts[:-1] += 1
    counts[1:] += 1
    return acc / counts


def position_stickiness_null(
    values: torch.Tensor, n_shuffles: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """The per-position permutation null of :func:`position_stickiness`.

    x23's ``stickiness_null`` permutes the modality's top-1 values over its
    positions (marginals preserved, temporal structure broken) and recomputes the
    SAME pair set; here the per-position scores are tracked across permutations.
    Returns ``(null_mean [m], null_std [m])`` with x23's population-std convention
    (``_std`` divides by n), or ``(None, None)`` when the statistic is undefined.
    """
    m = int(values.numel())
    if m < 2 or n_shuffles < 2:
        return None, None
    perms = torch.stack(
        [torch.randperm(m, generator=generator) for _ in range(n_shuffles)]
    )
    shuffled = values[perms]  # [S, m]
    pair = (shuffled[:, 1:] == shuffled[:, :-1]).float()  # [S, m-1]
    acc = torch.zeros(n_shuffles, m)
    acc[:, :-1] += pair
    acc[:, 1:] += pair
    counts = torch.zeros(m)
    counts[:-1] += 1
    counts[1:] += 1
    samples = acc / counts
    return samples.mean(dim=0), samples.std(dim=0, correction=0)


def select_registers(
    z_by_layer: list[torch.Tensor | None],
    obs_by_layer: list[torch.Tensor],
    register_k: int,
) -> list[int]:
    """Top-k image-block indices by the mean-across-layers z (see REGISTER_RULE)."""
    m = int(obs_by_layer[0].numel())
    zs = [z for z in z_by_layer if z is not None]
    z_mean = torch.stack(zs).mean(dim=0) if zs else torch.full((m,), float("nan"))
    obs_mean = torch.stack(obs_by_layer).mean(dim=0)
    order = sorted(
        range(m),
        key=lambda i: (
            0 if not math.isnan(float(z_mean[i])) else 1,
            -float(z_mean[i]) if not math.isnan(float(z_mean[i])) else 0.0,
            -float(obs_mean[i]),
            i,
        ),
    )
    return order[: max(register_k, 0)]


# ------------------------------------------------------------------ attention


@torch.no_grad()
def caption_attention_mass(
    model: LensModel,
    batch: MultimodalBatch,
    layers: list[int],
    source_positions: torch.Tensor,
) -> dict[int, tuple[torch.Tensor, float]]:
    """Per band layer: the mean attention mass each image position RECEIVES.

    x17's ``llm_image_attention`` machinery: one eager prefill forward with
    ``output_attentions=True``; per layer the source rows (caption tokens) are
    averaged over sources and heads, then indexed onto the image-token keys.
    Returns ``{layer: (mass [n_image], share)}`` where ``share`` is the fraction of
    the sources' mass landing anywhere on the image block.
    """
    output = model.hf_model.model(
        **batch.hf_kwargs(), use_cache=False, output_attentions=True, return_dict=True
    )
    attentions = output.attentions
    if (
        attentions is None
        or len(attentions) <= max(layers)
        or any(weight is None for weight in attentions)
    ):
        raise RuntimeError(
            "the model returned no attention weights: reload with "
            f"--attn-impl eager (requested implementation is "
            f"{model.hf_model.config._attn_implementation!r})"
        )
    image_positions = batch.image_token_mask[0].nonzero(as_tuple=True)[0]
    valid_keys = batch.attention_mask[0].bool().nonzero(as_tuple=True)[0]
    per_layer: dict[int, tuple[torch.Tensor, float]] = {}
    for layer in layers:
        attention = attentions[layer][0].float()  # [heads, seq, seq]
        rows = attention.index_select(1, source_positions).mean(dim=1)  # [heads, seq]
        patch = rows.index_select(1, image_positions)  # [heads, n_image]
        mass = patch.mean(dim=0)  # [n_image] head- and source-averaged
        total = rows.index_select(1, valid_keys).sum(dim=1)  # [heads]
        share = float((patch.sum(dim=1) / total.clamp_min(1e-12)).mean())
        per_layer[layer] = (mass.detach().cpu(), share)
    del output
    return per_layer


def caption_source_positions(
    model: LensModel,
    sample: RegisterSample,
    batch: MultimodalBatch,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """The caption-token source positions for the attention arm.

    The manifest corpus teacher-forces the model's own caption after
    ``'ASSISTANT:'``; when the prompt-only prefix tokenizes as a strict prefix of
    the full body (it does for the llava kinds; the wrapper never merges across the
    ``': '`` boundary), the caption span is the tail after the prefix. Otherwise all
    text positions are the sources (x17's mean-text fallback), recorded as such.
    """
    text_mask = x23.scope_masks(batch)["text"]
    text_positions = text_mask.nonzero(as_tuple=True)[0]
    if not sample.caption:
        return text_positions.cpu(), {"mode": "all_text"}
    # the prefix batch needs the image slot filled (encode_mm refuses placeholder
    # mismatches); only its token ids are used
    prefix = model.encode_mm(sample.generation_body, sample.image)
    prefix_ids = prefix.input_ids[0]
    full_ids = batch.input_ids[0]
    if (
        prefix_ids.numel() <= full_ids.numel()
        and bool(torch.equal(full_ids[: prefix_ids.numel()].cpu(), prefix_ids.cpu()))
    ):
        span = torch.arange(
            int(prefix_ids.numel()), int(full_ids.numel()), dtype=torch.long
        )
        if span.numel():
            return span, {"mode": "caption_span", "n_sources": int(span.numel())}
    return text_positions.cpu(), {"mode": "all_text_fallback"}


# ------------------------------------------------------------------ probes


def ridge_folds(
    feature_rows: list[torch.Tensor], lambda_rel: float
) -> list[
    tuple[int, list[int], list[int], torch.Tensor, torch.Tensor, torch.Tensor, float]
]:
    """Leave-one-image-out folds with train-standardized features and cached K.

    Features per image are ``[rows_i, d]``; each fold trains on the other images'
    rows (standardized by the train moments) and scores the held-out image's rows.
    ``lambda = lambda_rel * mean train row norm^2`` (after standardization this is
    ``lambda_rel * d`` up to sampling noise). K = X_tr X_tr^T is target-independent,
    so it is computed once per fold and reused across every probe target. Each tuple
    carries its held-out image index and the train rows' image indices (an image's
    rows all share its label — the solve's right-hand side is per ROW, not per
    image).
    """
    n = len(feature_rows)
    folds: list[
        tuple[int, list[int], list[int], torch.Tensor, torch.Tensor, torch.Tensor, float]
    ] = []
    for fold in range(n):
        train_idx = [i for i in range(n) if i != fold]
        if not train_idx:
            continue
        x_train = torch.cat([feature_rows[i] for i in train_idx], dim=0)
        row_image_idx = [
            image for image in train_idx for _ in range(int(feature_rows[image].shape[0]))
        ]
        mu = x_train.mean(dim=0, keepdim=True)
        sd = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
        x_train = (x_train - mu) / sd
        x_test = (feature_rows[fold] - mu) / sd
        lam = lambda_rel * float(x_train.pow(2).sum(dim=1).mean())
        kernel = x_train @ x_train.T
        folds.append((fold, train_idx, row_image_idx, x_train, x_test, kernel, lam))
    return folds


def ridge_auc_from_folds(
    folds: list[
        tuple[int, list[int], list[int], torch.Tensor, torch.Tensor, torch.Tensor, float]
    ],
    labels: list[int],
) -> tuple[float | None, dict[str, Any]]:
    """Pooled leave-one-image-out ridge AUC for one binary target (see PROBE_RULE)."""
    scores: list[torch.Tensor] = []
    pooled_labels: list[int] = []
    n_folds_used = 0
    for fold, _train_idx, row_image_idx, x_train, x_test, kernel, lam in folds:
        y_train = [labels[image] for image in row_image_idx]
        if len(set(y_train)) < 2:
            continue  # the fold's training pool must contain both classes
        y_tensor = torch.tensor(y_train, dtype=torch.float32)
        eye = torch.eye(kernel.shape[0], dtype=kernel.dtype)
        alpha = torch.linalg.solve(kernel + lam * eye, y_tensor)
        weights = x_train.T @ alpha
        scores.append(x_test @ weights)
        pooled_labels.extend([int(labels[fold])] * int(x_test.shape[0]))
        n_folds_used += 1
    if not scores:
        return None, {"n_folds_used": 0}
    score_vector = torch.cat(scores)
    label_vector = torch.tensor(pooled_labels, dtype=torch.long)
    auc = average_rank_auc(score_vector, label_vector)
    return auc, {
        "n_folds_used": n_folds_used,
        "n_scores": int(score_vector.numel()),
        "n_positives": int((label_vector == 1).sum()),
        "n_negatives": int((label_vector == 0).sum()),
    }


def average_rank_auc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    """Mann-Whitney AUC with average ranks for ties; None without both classes."""
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = torch.argsort(scores)
    sorted_scores = scores[order].double()
    ranks = torch.empty(scores.numel(), dtype=torch.float64)
    index = 0
    while index < scores.numel():
        end = index
        while end + 1 < scores.numel() and sorted_scores[end + 1] == sorted_scores[index]:
            end += 1
        ranks[index : end + 1] = (index + end) / 2.0 + 1.0  # mean of 1-based ranks
        index = end + 1
    positives = ranks[(labels == 1)[order]].sum()
    return float((positives - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def subsample_positions(
    positions: torch.Tensor, k: int, rng: random.Random
) -> list[int]:
    """Deterministic row subsample for one (image, position class) probe set."""
    pool = [int(p) for p in positions.tolist()]
    if len(pool) <= k:
        return pool
    return sorted(rng.sample(pool, k))


def load_color_captions(
    path: str | None, samples: list[RegisterSample]
) -> tuple[dict[int, list[str]], dict[str, Any]]:
    """``image_id -> ground-truth captions`` from a COCO captions json.

    Missing file (the default path on a non-deployment box) is not an error: the
    color targets are simply skipped with a recorded note. ``--synthetic`` builds a
    matching synthetic captions file first, so the loader path itself is exercised.
    """
    if path is None:
        return {}, {"loaded": False, "path": None, "note": "no captions path given"}
    captions_path = Path(path)
    if not captions_path.is_file():
        return {}, {
            "loaded": False,
            "path": str(captions_path),
            "note": "captions json absent; color targets skipped",
        }
    data = json.loads(captions_path.read_text(encoding="utf-8"))
    by_image: dict[int, list[str]] = {}
    for annotation in data.get("annotations", []):
        by_image.setdefault(int(annotation["image_id"]), []).append(
            str(annotation["caption"])
        )
    covered = sum(1 for sample in samples if by_image.get(sample.image_id))
    return by_image, {
        "loaded": True,
        "path": str(captions_path),
        "n_images_with_captions": covered,
    }


def build_synthetic_captions(
    samples: list[RegisterSample], name_by_id: dict[int, str]
) -> str:
    """A synthetic captions_val2014.json for the tiny smoke (color path exercise).

    Deterministic: each image gets two captions; the color word cycles over
    ``max(2, n_images // 3)`` colors so every cycled color reaches support >= 2
    with BOTH classes present (a single-color corpus would be all-positive and the
    probe AUC undefined). Caller owns the temp file.
    """
    color_names = [COLOR_CANONICAL.get(word, word) for word in COLOR_WORDS]
    cycle = max(2, len(samples) // 3)
    annotations = []
    for index, sample in enumerate(samples):
        color = color_names[index % cycle]
        object_name = (
            name_by_id.get(sorted(sample.gt_ids)[0], "object") if sample.gt_ids else "object"
        )
        annotations.append({
            "image_id": sample.image_id,
            "caption": f"a {color} {object_name} and other items",
        })
        annotations.append({
            "image_id": sample.image_id,
            "caption": f"a scene with a {object_name} in it",
        })
    handle = tempfile.NamedTemporaryFile(
        "w", suffix="_captions.json", delete=False, encoding="utf-8"
    )
    json.dump({"annotations": annotations}, handle)
    handle.close()
    return handle.name


def color_targets(
    captions_by_image: dict[int, list[str]],
    samples: list[RegisterSample],
    min_support: int,
) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Per-color binary targets: the word appears in one of the image's GT captions."""
    labels_by_color: dict[str, list[int]] = {
        COLOR_CANONICAL.get(word, word): [] for word in COLOR_WORDS
    }
    support: Counter[str] = Counter()
    for sample in samples:
        captions = captions_by_image.get(sample.image_id, [])
        lowered = [caption.lower() for caption in captions]
        present = {
            word
            for word in labels_by_color
            if any(re.search(rf"\b{re.escape(word)}\b", caption) for caption in lowered)
        }
        support.update(present)
        for word in labels_by_color:
            labels_by_color[word].append(1 if word in present else 0)
    kept = {
        word: labels for word, labels in labels_by_color.items() if sum(labels) >= min_support
    }
    return kept, dict(support)


def object_targets(
    samples: list[RegisterSample], name_by_id: dict[int, str], min_support: int
) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Per-class binary targets: the COCO category is annotated on the image."""
    support: Counter[str] = Counter()
    for sample in samples:
        support.update(name_by_id[cid] for cid in sample.gt_ids if cid in name_by_id)
    labels_by_class: dict[str, list[int]] = {}
    for name in sorted(support):
        if support[name] < min_support:
            continue
        labels_by_class[name] = [
            1 if name in {name_by_id[cid] for cid in sample.gt_ids if cid in name_by_id}
            else 0
            for sample in samples
        ]
    return labels_by_class, dict(support)


# ------------------------------------------------------------------ chair


def normalize_caption(text: str) -> str:
    """Lowercase + the minimal noun-synonym fold (see CHAIR_RULE)."""
    lowered = text.lower()
    for synonym, canonical in CHAIR_SYNONYMS.items():
        lowered = re.sub(rf"\b{re.escape(synonym)}\b", canonical, lowered)
    return lowered


def chair_row(
    text: str, gt_names: set[str], name_by_id: dict[int, str]
) -> dict[str, Any]:
    """CHAIR-lite metrics for one generated caption (see CHAIR_RULE)."""
    mentions = detect_mentions(name_by_id, normalize_caption(text))
    extra = [name for name in mentions if name not in gt_names]
    return {
        "n_mentions": len(mentions),
        "n_extra_mentions": len(extra),
        "sentence_flag": 1.0 if extra else 0.0,
        "mentions": mentions,
        "extra_mentions": extra,
    }


def chair_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """CHAIR-s/r, caption lengths, and mention/lens-shift totals over one condition."""
    n_mentions = sum(row["n_mentions"] for row in rows)
    n_extra = sum(row["n_extra_mentions"] for row in rows)
    aggregated = {
        "n_images": len(rows),
        "chair_s": x23._round(x23._mean([row["sentence_flag"] for row in rows])),
        "chair_r": x23._round(n_extra / n_mentions) if n_mentions else None,
        "n_mentions_total": n_mentions,
        "n_extra_mentions_total": n_extra,
        "mean_len_tokens": x23._round(x23._mean(
            [float(row["n_new_tokens"]) for row in rows]
        ), 2),
        "mean_len_words": x23._round(x23._mean(
            [float(row["n_words"]) for row in rows]
        ), 2),
    }
    for key in ("object_rate_mean", "p_yes_binary_softmax_mean", "lens_gap_first_step"):
        values = [row[key] for row in rows if row.get(key) is not None]
        aggregated[key] = x23._round(x23._mean(values), 4) if values else None
    return aggregated


def _transport_rows(rows: torch.Tensor, lens: Any, layer: int) -> torch.Tensor:
    """``J_l @ rows`` with the identity-lens no-op detection.

    x26's workspace builds the identity-transport lens (the x23 convention for raw
    readouts); an identity block is NOT the fitted J. Load the fitted Jacobians
    once (cached), and if the supplied lens is identity, transport with the FITTED
    ``J_l`` instead; layers absent from the fitted lens pass through unchanged.
    """
    fitted = _fitted_jacobians()
    jac = fitted.get(layer) if fitted else None
    if jac is None:
        jacobians = getattr(lens, "jacobians", None)
        if not jacobians or layer not in jacobians:
            return rows
        block = jacobians[layer]
        if block.ndim == 2 and float(torch.diagonal(block).sum()) == float(block.sum()):
            return rows
        return rows.to(block.device, torch.float32) @ block.T
    return rows.to(jac.device, torch.float32) @ jac.T


_FITTED_CACHE: dict[int, torch.Tensor] | None = None


def _fitted_jacobians() -> dict[int, torch.Tensor] | None:
    """The fitted lens's ``J_l`` blocks (loaded once; the identity-lens no-op path)."""
    global _FITTED_CACHE
    if _FITTED_CACHE is not None:
        return _FITTED_CACHE
    lens_dir = os.environ.get("X26_FITTED_LENS_DIR", "")
    if not lens_dir:
        _FITTED_CACHE = {}
        return None
    try:
        lenses, _ = load_lens_set(lens_dir)
    except Exception:  # noqa: BLE001 - any load failure = no fitted path
        _FITTED_CACHE = {}
        return None
    mask = os.environ.get("X26_FITTED_LENS_MASK", "text")
    lens = lenses.get(mask)
    _FITTED_CACHE = dict(lens.jacobians) if lens is not None else {}
    return _FITTED_CACHE


# ------------------------------------------------------------------ arm a+b: registers


@torch.no_grad()
def run_registers(args: argparse.Namespace, ws: RegisterWorkspace) -> RegisterContext:
    """The per-image pass: readouts, per-position stickiness, registers, features.

    One recorded forward per image (x23 band's ActivationRecorder pattern) feeds the
    calibrated per-position readout at every band layer; the same residuals seed the
    probe features, the mean-image-residual table (the mean-ablate fill), and the
    x23-style modality pools. The attention arm runs on the first ``--n-attention``
    images (a second, eager forward with output_attentions).
    """
    model = ws.model
    layers = ws.layers
    ctx = RegisterContext()
    ctx.modality_pools = {
        (layer, modality): {
            "stick": [], "null_mean": [], "null_std": [], "n_positions": [],
        }
        for layer in layers
        for modality in ("image", "text")
    }
    ctx.class_rate_pools = {
        (layer, pos_class): [] for layer in layers for pos_class in POSITION_CLASSES
    }
    ctx.probe_features = {
        (layer, pos_class): [] for layer in layers for pos_class in POSITION_CLASSES
    }
    mean_sum = {layer: torch.zeros(model.d_model) for layer in layers}
    mean_count = 0
    object_ids_all = sorted(set(ws.emission.values()))

    for index, sample in enumerate(ws.samples):
        if not sample.gt_ids:
            ctx.n_skipped_no_gt += 1
        batch = registry.image_encode(
            model, ws.kind, sample.text, sample.image, max_length=args.max_seq_len
        )
        masks = x23.scope_masks(batch)
        image_mask = masks["image"]
        text_mask = masks["text"]
        image_positions = image_mask.nonzero(as_tuple=True)[0].cpu()
        if image_positions.numel() < 2:
            print(f"  skip {sample.name}: {image_positions.numel()} image tokens")
            ctx.image_spans.append((-1, -1))
            ctx.register_positions.append([])
            ctx.records.append({"name": sample.name, "skipped": "no_image_block"})
            for layer in layers:
                for pos_class in POSITION_CLASSES:
                    ctx.probe_features[(layer, pos_class)].append(torch.zeros(0, model.d_model))
            continue
        span = (int(image_positions[0]), int(image_positions[-1]) + 1)
        ctx.image_spans.append(span)
        mean_count += 1
        with ActivationRecorder(model.layers, at=layers) as recorder:
            model.forward_mm(batch)
        residuals = {
            layer: recorder.activations[layer][0].detach().float().cpu()
            for layer in layers
        }

        object_ids = [
            ws.emission[name]
            for name in sorted({ws.name_by_id[cid] for cid in sample.gt_ids if cid in ws.name_by_id})
            if name in ws.emission
        ]
        sample_objects = (
            torch.tensor(object_ids) if object_ids else torch.tensor(object_ids_all)
        )

        # probe row sets (position classes); registers are chosen after the layer loop
        row_positions: dict[str, list[int]] = {}
        for pos_class in POSITION_CLASSES:
            if pos_class == "register":
                row_positions[pos_class] = []  # filled after the layer loop
            elif pos_class == "image_nonregister":
                row_positions[pos_class] = image_positions.tolist()  # pruned below
            else:
                row_positions[pos_class] = subsample_positions(
                    text_mask.nonzero(as_tuple=True)[0].cpu(),
                    args.probe_rows_per_image,
                    random.Random(
                        args.seed * 9176 + index * 131 + _CLASS_SEED_SALT[pos_class]
                    ),
                )

        z_by_layer: list[torch.Tensor | None] = []
        obs_by_layer: list[torch.Tensor] = []
        object_hit_by_layer: dict[int, torch.Tensor] = {}
        per_layer_obs: dict[int, list[float]] = {layer: [] for layer in layers}
        for layer in layers:
            residual = residuals[layer]
            mean_sum[layer] += residual[image_positions].sum(dim=0)
            top1, topk_ids, _ = x23.position_readout(ws, residual, layer)
            values = top1[image_positions]
            obs = position_stickiness(values)
            obs_by_layer.append(obs)
            generator = torch.Generator().manual_seed(
                args.seed * 17 + index * 977 + layer
            )
            null_mean, null_std = position_stickiness_null(
                values, args.n_null_shuffles, generator
            )
            if null_mean is None or null_std is None:
                z_by_layer.append(None)
            else:
                spread = null_std.clamp_min(0.0)
                usable = spread > 1e-12
                if bool(usable.any()):
                    z = torch.full((int(values.numel()),), float("nan"))
                    z[usable] = (obs[usable] - null_mean[usable]) / spread[usable]
                    z_by_layer.append(z)
                else:
                    z_by_layer.append(None)
            # x23-style modality pools (the D48 anchor: block-level stickiness + z)
            for modality, mask in (("image", image_mask), ("text", text_mask)):
                pool = ctx.modality_pools[(layer, modality)]
                stick, mod_values, pair_valid = x23.modality_stickiness(top1, mask)
                positions = mask.detach().cpu().nonzero(as_tuple=True)[0]
                pool["n_positions"].append(float(positions.numel()))
                if stick is not None:
                    null_m, null_s = x23.stickiness_null(
                        mod_values, pair_valid, args.n_null_shuffles,
                        random.Random(args.seed * 17 + index * 977 + layer),
                    )
                    pool["stick"].append(stick)
                    if null_m is not None:
                        pool["null_mean"].append(null_m)
                    if null_s is not None:
                        pool["null_std"].append(null_s)
            # the unembed cross-check: object-in-top-10 rate per position class
            # (registers are scored after the layer loop, from the same tensor)
            object_hit_by_layer[layer] = torch.isin(topk_ids, sample_objects).any(-1)
            for pos_class in ("image_nonregister", "text"):
                positions = (
                    image_positions.tolist() if pos_class == "image_nonregister"
                    else row_positions["text"]
                )
                if positions:
                    ctx.class_rate_pools[(layer, pos_class)].append(
                        float(object_hit_by_layer[layer][positions].float().mean())
                    )
            per_layer_obs[layer] = obs.tolist()

        register_indices = select_registers(z_by_layer, obs_by_layer, args.register_k)
        register_positions = [int(image_positions[i]) for i in register_indices]
        ctx.register_positions.append(register_positions)
        row_positions["register"] = register_positions
        nonregister = [p for p in image_positions.tolist() if p not in set(register_positions)]
        row_positions["image_nonregister"] = subsample_positions(
            torch.tensor(nonregister),
            args.probe_rows_per_image,
            random.Random(
                args.seed * 9176 + index * 131 + _CLASS_SEED_SALT["image_nonregister"]
            ),
        )
        for layer in layers:
            for pos_class in POSITION_CLASSES:
                positions = row_positions[pos_class]
                rows = residuals[layer][positions] if positions else torch.zeros(
                    0, model.d_model
                )
                if args.probe_transport and rows.numel():
                    rows = _transport_rows(rows, ws.lens, layer)
                ctx.probe_features[(layer, pos_class)].append(rows.clone())
            if register_positions:
                ctx.class_rate_pools[(layer, "register")].append(
                    float(object_hit_by_layer[layer][register_positions].float().mean())
                )

        z_mean: list[float | None] | None = None
        usable_z = [z for z in z_by_layer if z is not None]
        if usable_z:
            stacked = torch.stack(usable_z)  # [n_usable_layers, m], NaN where spreadless
            z_mean = [
                float(stacked[:, i].nanmean()) if bool((stacked[:, i] == stacked[:, i]).any())
                else None
                for i in range(stacked.shape[1])
            ]
        obs_mean = torch.stack(obs_by_layer).mean(dim=0).tolist()
        record = {
            "name": sample.name,
            "image_id": sample.image_id,
            "n_image_positions": int(image_positions.numel()),
            "image_span": list(span),
            "register_positions_absolute": register_positions,
            "register_patch_indices": [int(i) for i in register_indices],
            "registers": [
                {
                    "position": int(image_positions[i]),
                    "patch": int(i),
                    "z_mean": (
                        None if z_mean is None else x23._round(z_mean[i], 3)
                    ),
                    "stickiness_mean": x23._round(float(obs_mean[i]), 4),
                    "stickiness_per_layer": {
                        str(layer): x23._round(per_layer_obs[layer][i], 4)
                        for layer in layers
                    },
                }
                for i in register_indices
            ],
            "stickiness_z_all_positions": (
                None if z_mean is None
                else [x23._round(value, 3) for value in z_mean]
            ),
            "stickiness_obs_all_positions": [
                x23._round(value, 4) for value in obs_mean
            ],
        }
        if args.n_attention > 0 and len(ctx.attention_records) < args.n_attention:
            attention = run_attention_for_sample(
                ws, sample, batch, register_indices, obs_mean
            )
            ctx.attention_records.append(attention)
            record["attention"] = attention
        ctx.records.append(record)
        if (index + 1) % 5 == 0 or index + 1 == len(ws.samples):
            print(
                f"  [{index + 1}/{len(ws.samples)}] {sample.name} read out", flush=True
            )

    ctx.mean_image_residual = {
        layer: (mean_sum[layer] / max(mean_count, 1)) for layer in layers
    }
    return ctx


def run_attention_for_sample(
    ws: RegisterWorkspace,
    sample: RegisterSample,
    batch: MultimodalBatch,
    register_indices: list[int],
    obs_mean: list[float],
) -> dict[str, Any]:
    """The eager attention forward for one sample (see ATTENTION_RULE)."""
    sources, source_meta = caption_source_positions(ws.model, sample, batch)
    sources = sources.to(batch.input_ids.device)
    per_layer = caption_attention_mass(ws.model, batch, ws.layers, sources)
    n_image = int(sum(1 for _ in obs_mean))
    register_set = set(register_indices)
    per_layer_out: dict[str, Any] = {}
    for layer, (mass, share) in per_layer.items():
        mass_list = [float(value) for value in mass.tolist()]
        correlation = x23.pearson(mass_list, obs_mean)
        register_mass = [mass_list[i] for i in register_indices]
        other_mass = [
            mass_list[i] for i in range(n_image) if i not in register_set
        ]
        ratio = (
            (sum(register_mass) / len(register_mass))
            / (sum(other_mass) / len(other_mass))
            if register_mass and other_mass and sum(other_mass) > 0
            else None
        )
        per_layer_out[str(layer)] = {
            "corr_stickiness_mass": x23._round(correlation, 4),
            "mass_share_on_image": x23._round(share, 6),
            "register_mass_ratio": x23._round(ratio, 4),
            "mass_vector": [x23._round(value, 8) for value in mass_list],
        }
    return {
        "sources": source_meta,
        "per_layer": per_layer_out,
    }


def register_summary(args: argparse.Namespace, ws: RegisterWorkspace, ctx: RegisterContext) -> dict[str, Any]:
    """Per-layer aggregates: x23-style modality stickiness + per-class object rates."""
    layer_table: dict[str, dict[str, Any]] = {}
    for layer in ws.layers:
        entry: dict[str, Any] = {}
        for modality in ("image", "text"):
            pool = ctx.modality_pools[(layer, modality)]
            z = None
            if pool["stick"] and pool["null_mean"] and pool["null_std"]:
                observed = x23._mean(pool["stick"])
                null = x23._mean(pool["null_mean"])
                spread = x23._mean(pool["null_std"]) or 0.0
                if observed is not None and null is not None and spread > 1e-12:
                    z = (observed - null) / spread
            entry[modality] = {
                "n_images": len(pool["stick"]),
                "n_positions_mean": x23._round(x23._mean(pool["n_positions"]), 1),
                "top1_stickiness": x23._round(x23._mean(pool["stick"])),
                "stickiness_null_mean": x23._round(x23._mean(pool["null_mean"])),
                "stickiness_null_std": x23._round(x23._mean(pool["null_std"])),
                "stickiness_z": x23._round(z, 2),
            }
        for pos_class in POSITION_CLASSES:
            entry[f"object_in_top10_rate_{pos_class}"] = x23._round(
                x23._mean(ctx.class_rate_pools[(layer, pos_class)])
            )
        layer_table[str(layer)] = entry
    return layer_table


# ------------------------------------------------------------------ arm c: probes


def run_probes(
    args: argparse.Namespace,
    ws: RegisterWorkspace,
    ctx: RegisterContext,
) -> dict[str, Any]:
    """The ridge-probe table + the unembed cross-check (see PROBE_RULE).

    Folds are cached per (layer, position class) — K is target-independent — and each
    target (object class / color word) costs one dual solve per fold. Targets with
    too little support are skipped and counted. Images skipped by the registers pass
    contribute zero-row feature blocks and are excluded from the targets' labels.
    """
    usable = [
        index
        for index, record in enumerate(ctx.records)
        if record.get("image_span", (-1, -1))[0] >= 0
    ]
    if len(usable) < 3:
        return {
            "skipped": f"only {len(usable)} usable images; probes need >= 3",
            "object_targets": {}, "color_targets": {}, "unembed_crosscheck": {},
        }
    samples = [ws.samples[index] for index in usable]
    object_labels, object_support = object_targets(
        samples, ws.name_by_id, args.probe_min_support
    )
    if args.synthetic:
        synthetic_captions_path = build_synthetic_captions(samples, ws.name_by_id)
        captions_path = synthetic_captions_path
    else:
        synthetic_captions_path = None
        captions_path = args.captions
    captions_by_image, captions_meta = load_color_captions(captions_path, samples)
    color_labels, color_support = color_targets(
        captions_by_image, samples, args.probe_min_support
    )
    if synthetic_captions_path:
        Path(synthetic_captions_path).unlink(missing_ok=True)

    folds_cache: dict[tuple[int, str], list] = {}
    for layer in ws.layers:
        for pos_class in POSITION_CLASSES:
            rows = [ctx.probe_features[(layer, pos_class)][i] for i in usable]
            if all(int(block.shape[0]) == 0 for block in rows):
                continue
            folds_cache[(layer, pos_class)] = ridge_folds(rows, args.probe_lambda_rel)

    def probe_table(labels_by_target: dict[str, list[int]]) -> dict[str, Any]:
        per_layer: dict[str, dict[str, Any]] = {}
        for layer in ws.layers:
            per_class: dict[str, Any] = {}
            for pos_class in POSITION_CLASSES:
                folds = folds_cache.get((layer, pos_class), [])
                if not folds:
                    per_class[pos_class] = {"mean_auc": None, "n_targets": 0}
                    continue
                entries: dict[str, float] = {}
                details: dict[str, Any] = {}
                for target_name, labels in labels_by_target.items():
                    auc, info = ridge_auc_from_folds(folds, labels)
                    if auc is not None:
                        entries[target_name] = round(auc, 4)
                        details[target_name] = info
                per_class[pos_class] = {
                    "mean_auc": x23._round(x23._mean(list(entries.values()))),
                    "n_targets": len(entries),
                    "per_target": entries,
                    "fold_detail": details,
                }
            per_layer[str(layer)] = per_class
        return per_layer

    object_table = probe_table(object_labels)
    color_table = (
        probe_table(color_labels)
        if color_labels
        else {"skipped": "no color word reached --probe-min-support", "per_layer": {}}
    )
    crosscheck = {
        str(layer): {
            pos_class: x23._round(x23._mean(ctx.class_rate_pools[(layer, pos_class)]))
            for pos_class in POSITION_CLASSES
        }
        for layer in ws.layers
    }
    return {
        "object_targets": {
            "support": object_support,
            "n_targets": len(object_labels),
            "per_layer": object_table,
        },
        "color_targets": {
            "captions": captions_meta,
            "support": color_support,
            "n_targets": len(color_labels),
            "per_layer": color_table,
        },
        "unembed_crosscheck_object_in_top10": crosscheck,
        "n_usable_images": len(usable),
    }


# ------------------------------------------------------------------ arm d: ablation


def lens_shift_stats(
    ws: RegisterWorkspace,
    step_records: list[dict[int, torch.Tensor]],
    layer: int,
    object_ids: list[int],
    yes_no: tuple[int, int] | None,
) -> dict[str, Any]:
    """The post-edit lens shift over the generation's decoding steps (ABLATION_RULE)."""
    emission_ids = (
        torch.tensor(object_ids) if object_ids else torch.tensor(
            sorted(set(ws.emission.values()))
        )
    )
    rates: list[float] = []
    p_yes_values: list[float] = []
    gap_first: float | None = None
    for step, record in enumerate(step_records):
        hidden = record[layer].unsqueeze(0)
        logits = _calibrated_lens_logits(
            ws.model, ws.lens, hidden, layer,
            ws.bias, ws.scale, ws.temp, ws.logit_bias,
        )[0]
        k = min(ws.top_k, int(logits.numel()))
        top_ids = logits.topk(k).indices
        rates.append(float(bool(torch.isin(top_ids, emission_ids).any())))
        if yes_no is not None:
            yes_logit = float(logits[yes_no[0]])
            no_logit = float(logits[yes_no[1]])
            pair = torch.tensor([yes_logit, no_logit], dtype=torch.float64)
            softmax = torch.softmax(pair, dim=0)
            p_yes_values.append(float(softmax[0]))
            if step == 0:
                gap_first = yes_logit - no_logit
    return {
        "object_rate_mean": x23._round(x23._mean(rates)),
        "p_yes_binary_softmax_mean": x23._round(x23._mean(p_yes_values), 4),
        "lens_gap_first_step": x23._round(gap_first, 4),
        "n_steps": len(rates),
    }


@torch.no_grad()
def run_ablation(
    args: argparse.Namespace,
    ws: RegisterWorkspace,
    ctx: RegisterContext,
) -> dict[str, Any]:
    """Causal register ablation during greedy captioning (see ABLATION_RULE)."""
    try:
        yes_id, no_id, yesno_note = select_yes_no_ids(ws.model.tokenizer)
        yes_no: tuple[int, int] | None = (yes_id, no_id)
    except ValueError as exc:
        yes_no = None
        yesno_note = {"error": str(exc), "note": "P(yes)/gap skipped"}
        print(f"note: no usable yes/no token pair ({exc}); P(yes)/gap skipped")

    def generate_condition(
        sample: RegisterSample, positions: list[int], layer: int
    ) -> dict[str, Any]:
        batch_gen = registry.image_encode(
            ws.model, ws.kind, sample.generation_body, sample.image,
            max_length=args.max_seq_len,
        )
        gen_positions = batch_gen.image_token_mask[0].nonzero(as_tuple=True)[0]
        gen_span = (
            (int(gen_positions[0]), int(gen_positions[-1]) + 1)
            if gen_positions.numel() else (-1, -1)
        )
        corpus_span = ctx.image_spans[ws.samples.index(sample)]
        if gen_span != corpus_span:
            raise ValueError(
                f"{sample.name}: generation image span {gen_span} != corpus span "
                f"{corpus_span}; register positions would not transfer"
            )
        if positions:
            edits = [
                ResidualEdit(
                    layer=layer,
                    mode="ablate",
                    position_ids=tuple(positions),
                    mean=(
                        ctx.mean_image_residual[layer]
                        if args.ablate_mode == "mean"
                        else None
                    ),
                ),
            ]
            with ResidualEditor(ws.model, ws.lens, edits):
                new_tokens, step_records = generate_with_residual_traces(
                    ws.model, batch_gen, [layer], max_new_tokens=args.max_new_tokens
                )
        else:
            new_tokens, step_records = generate_with_residual_traces(
                ws.model, batch_gen, [layer], max_new_tokens=args.max_new_tokens
            )
        text = ws.model._decode(new_tokens)
        gt_names = {
            ws.name_by_id[cid] for cid in sample.gt_ids if cid in ws.name_by_id
        }
        object_ids = [
            ws.emission[name] for name in sorted(gt_names) if name in ws.emission
        ]
        shift = lens_shift_stats(ws, step_records, layer, object_ids, yes_no)
        return {
            "name": sample.name,
            "text": text,
            "n_new_tokens": int(new_tokens.numel()),
            "n_words": len(text.split()),
            **chair_row(text, gt_names, ws.name_by_id),
            **shift,
        }

    per_layer: dict[str, Any] = {}
    for layer in ws.ablate_layers:
        conditions: dict[str, list[dict[str, Any]]] = {"baseline": [], "ablated": []}
        n_ablation_skipped = 0
        for sample_index, sample in enumerate(ws.samples):
            if ctx.image_spans[sample_index][0] < 0:
                n_ablation_skipped += 1  # the registers pass could not read this image
                continue
            positions = ctx.register_positions[sample_index]
            conditions["baseline"].append(generate_condition(sample, [], layer))
            if positions:
                conditions["ablated"].append(
                    generate_condition(sample, positions, layer)
                )
            else:
                conditions["ablated"].append({
                    "name": sample.name, "skipped": "no registers",
                })
        entry: dict[str, Any] = {"n_skipped_no_image_block": n_ablation_skipped}
        baseline = chair_aggregate(
            [row for row in conditions["baseline"] if "skipped" not in row]
        )
        ablated_rows = [row for row in conditions["ablated"] if "skipped" not in row]
        ablated = chair_aggregate(ablated_rows)
        deltas = {
            key: (
                None
                if baseline.get(key) is None or ablated.get(key) is None
                else x23._round(ablated[key] - baseline[key], 4)
            )
            for key in (
                "chair_s", "chair_r", "mean_len_tokens", "mean_len_words",
                "object_rate_mean", "p_yes_binary_softmax_mean",
                "lens_gap_first_step",
            )
        }
        entry.update({
            "baseline": baseline,
            "ablated": ablated,
            "deltas_ablated_minus_baseline": deltas,
            "per_image_baseline": conditions["baseline"],
            "per_image_ablated": conditions["ablated"],
            "edit_spec": {
                "layer": layer,
                "mode": "ablate",
                "position_ids": "per-image register positions",
                "mean": (
                    "corpus image-position mean residual at the layer"
                    if args.ablate_mode == "mean" else None
                ),
            },
        })
        per_layer[str(layer)] = entry
        print(
            f"  ablate L{layer} [{args.ablate_mode}]: chair_s "
            f"{baseline['chair_s']} -> {ablated['chair_s']}; len "
            f"{baseline['mean_len_tokens']} -> {ablated['mean_len_tokens']}; "
            f"obj_rate {baseline['object_rate_mean']} -> "
            f"{ablated['object_rate_mean']}",
            flush=True,
        )
    return {
        "ablate_layers": ws.ablate_layers,
        "ablate_mode": args.ablate_mode,
        "max_new_tokens": args.max_new_tokens,
        "yesno": yesno_note,
        "per_layer": per_layer,
    }


# ------------------------------------------------- attention aggregate helper


def attention_summary(ctx: RegisterContext) -> dict[str, Any]:
    """Mean attention correlation / register mass ratio per layer across samples."""
    per_layer: dict[str, dict[str, list[float]]] = {}
    for record in ctx.attention_records:
        for layer, entry in record["per_layer"].items():
            bucket = per_layer.setdefault(
                layer, {"corr": [], "share": [], "reg_ratio": []}
            )
            if entry["corr_stickiness_mass"] is not None:
                bucket["corr"].append(entry["corr_stickiness_mass"])
            if entry["mass_share_on_image"] is not None:
                bucket["share"].append(entry["mass_share_on_image"])
            if entry.get("register_mass_ratio") is not None:
                bucket["reg_ratio"].append(entry["register_mass_ratio"])
    return {
        layer: {
            "mean_corr_stickiness_mass": x23._round(x23._mean(bucket["corr"]), 4),
            "mean_mass_share_on_image": x23._round(x23._mean(bucket["share"]), 6),
            "mean_register_mass_ratio": x23._round(x23._mean(bucket["reg_ratio"]), 4),
            "n_samples": len(bucket["corr"]),
        }
        for layer, bucket in per_layer.items()
    }


# --------------------------------------------------------------------------- main


def build_layer_digest(
    args: argparse.Namespace,
    ws: RegisterWorkspace,
    ctx: RegisterContext,
    probes: dict[str, Any],
    ablation: dict[str, Any],
) -> str:
    """The per-layer ASCII table (x23 digest convention)."""
    lines = [
        "== X26 visual registers: "
        f"{len(ctx.records)} images, layers={ws.layers}, register_k={args.register_k} ==",
        f"{'layer':<7}{'stick_img':>10}{'z_img':>8}{'stick_txt':>10}{'z_txt':>8}"
        f"{'obj_reg':>9}{'obj_img':>9}{'obj_txt':>9}"
        f"{'attn_r':>8}{'attn_ratio':>11}"
        f"{'auc_reg':>9}{'auc_img':>9}{'auc_txt':>9}",
    ]
    object_table = probes.get("object_targets", {}).get("per_layer", {})
    attention = attention_summary(ctx)
    layer_rows = register_summary(args, ws, ctx)
    for layer in ws.layers:
        row = layer_rows.get(str(layer), {})
        image_row = row.get("image", {})
        text_row = row.get("text", {})
        attn = attention.get(str(layer), {})
        probe_row = object_table.get(str(layer), {})
        lines.append(
            f"L{layer:<6}"
            f"{x23._fmt(image_row.get('top1_stickiness'), 10)}"
            f"{x23._fmt(image_row.get('stickiness_z'), 8, 2)}"
            f"{x23._fmt(text_row.get('top1_stickiness'), 10)}"
            f"{x23._fmt(text_row.get('stickiness_z'), 8, 2)}"
            f"{x23._fmt(row.get('object_in_top10_rate_register'), 9)}"
            f"{x23._fmt(row.get('object_in_top10_rate_image_nonregister'), 9)}"
            f"{x23._fmt(row.get('object_in_top10_rate_text'), 9)}"
            f"{x23._fmt(attn.get('mean_corr_stickiness_mass'), 8, 2)}"
            f"{x23._fmt(attn.get('mean_register_mass_ratio'), 11, 2)}"
            f"{x23._fmt(probe_row.get('register', {}).get('mean_auc'), 9)}"
            f"{x23._fmt(probe_row.get('image_nonregister', {}).get('mean_auc'), 9)}"
            f"{x23._fmt(probe_row.get('text', {}).get('mean_auc'), 9)}"
        )
    lines.append(
        "workspace signature: registers carry the stickiness (z_img >> z_txt) while "
        "obj_reg stays ~0 (unembed-blind); trained-probe AUC at registers > 0.5 "
        "with obj_reg ~ 0 = visual state present but un-readable; attention_ratio "
        "> 1 = caption tokens attend registers preferentially"
    )
    for layer, entry in ablation.get("per_layer", {}).items():
        base, abl = entry["baseline"], entry["ablated"]
        lines.append(
            f"ablate L{layer} [{args.ablate_mode}]: chair_s {base['chair_s']} -> "
            f"{abl['chair_s']}; chair_r {base['chair_r']} -> {abl['chair_r']}; "
            f"len {base['mean_len_tokens']} -> {abl['mean_len_tokens']}; obj_rate "
            f"{base['object_rate_mean']} -> {abl['object_rate_mean']}; p_yes "
            f"{base['p_yes_binary_softmax_mean']} -> "
            f"{abl['p_yes_binary_softmax_mean']}"
        )
    return "\n".join(lines)


def atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    """Write the report through a temp file + rename (atomic-write convention)."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temp, destination)


def main() -> int:
    args = parse_args()
    ws = build_workspace(args)
    print(
        f"== X26 visual-register investigation: backend={args.backend} kind={ws.kind} "
        f"images={len(ws.samples)} layers={ws.layers} register_k={args.register_k} "
        f"ablate={ws.ablate_layers}({args.ablate_mode}) =="
    )
    ctx = run_registers(args, ws)
    summary = register_summary(args, ws, ctx)
    probes = run_probes(args, ws, ctx)
    ablation = run_ablation(args, ws, ctx)
    digest = build_layer_digest(args, ws, ctx, probes, ablation)
    report = {
        "script": Path(__file__).name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": args.backend,
        "synthetic": args.synthetic,
        "model_kind": ws.kind,
        "model_id": registry.MODEL_SPECS[ws.kind]["model_id"],
        "images_dir": None if args.synthetic else str(args.images_dir),
        "annotations": None if args.synthetic else str(args.annotations),
        "n_images_requested": args.n_images,
        "n_samples": len(ws.samples),
        "n_skipped_no_gt": ctx.n_skipped_no_gt,
        "seed": args.seed,
        "layers": ws.layers,
        "top_k": ws.top_k,
        "register_k": args.register_k,
        "max_seq_len": args.max_seq_len,
        "max_new_tokens_ablation": args.max_new_tokens,
        "attn_implementation": args.attn_impl,
        "prompt_template": registry.image_prompt_template(ws.kind),
        "prompt_template_hash": registry.prompt_template_sha256(ws.kind),
        "workspace_band": {
            "band_lo": registry.MODEL_SPECS[ws.kind]["band_lo"],
            "band_hi": registry.MODEL_SPECS[ws.kind]["band_hi"],
            "readout_layer": ws.readout_layer,
        },
        "corpus": getattr(ws, "corpus_meta", {}),
        "lens": ws.lens_meta,
        "bias": ws.bias_meta,
        "calibrated": ws.calibrated,
        "registers": {
            "per_layer": summary,
            "per_image": ctx.records,
            "register_rule": REGISTER_RULE,
            "n_attention_samples": len(ctx.attention_records),
        },
        "attention": {
            "rule": ATTENTION_RULE,
            "n_samples": len(ctx.attention_records),
            "aggregate": attention_summary(ctx),
            "per_sample": ctx.attention_records,
        },
        "probes": {"rule": PROBE_RULE, **probes},
        "ablation": {"rule": ABLATION_RULE, "chair_matcher": CHAIR_RULE, **ablation},
        "digest": digest,
        "notes": {
            "mean_ablate_fill": (
                "the mean-ablate arm replaces register rows with the corpus image-"
                "position mean residual at the ablate layer (from the unedited "
                "registers pass), so 'mean' = the average image token, not the "
                "image's own mean"
            ),
            "register_transfer": (
                "register positions are identified on the corpus text (prompt+caption "
                "on the manifest) and applied to the prompt-only generation prefill; "
                "the arm asserts the image block span is identical in both"
            ),
            "lens_fingerprint": ws.lens_meta.get("fingerprint_note"),
        },
        "interpretation_limits": [
            "--n-images 20 makes every AUC/correlation a coarse estimate; the D48 "
            "band numbers (stickiness z 8-20, object rate 0-2%) come from the "
            "production 60-image x23 run, not this subset",
            "the probe AUC compares position classes within one architecture; it "
            "does not by itself establish that registers CAUSE caption content — "
            "the ablation arm is the causal half",
            "CHAIR-lite is an exact+synonym matcher (no hypernyms, irregular "
            "plurals a documented gap); its chair_s/r are comparable ACROSS "
            "conditions of this run, not against published CHAIR numbers",
            "the tiny backend is a random-weight fixture: it exercises the "
            "machinery, its numbers do not transfer to the real model",
        ],
    }
    atomic_write_json(args.json, report)
    print("\n" + digest)
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
