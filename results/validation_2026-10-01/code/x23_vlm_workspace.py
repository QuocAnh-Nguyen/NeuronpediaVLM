#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""X23: VLM-only global-workspace arms — does the text-LLM workspace story hold ACROSS
MODALITIES in LLaVA-1.5-7B?

Prior workspace work (text LLMs) found a "global workspace" band of layers where J-lens
readouts carry task-relevant content and re-index. For LLaVA-1.5-7B (32 layers) the
repo's prior results pin the text-mediated handoff at ~L20-25 and show that image-token
J-lens readouts never contain the image's objects at ANY layer; the paper's band
reindexes to ~[12, 29] of 32 layers. ``--model`` selects the checkpoint (default
llava-7b; llava-13b and qwen2vl bring their registry bands, chat conventions and
vision-projector module — see vlm_registry). This script runs the four experiments
that ONLY a VLM can run, one arm per ``--arm``:

report      CROSS-MODAL VERBAL REPORT SWAP. Prompt ``USER: <image>\\nWhat {category} is
            in the picture? Answer with one word:`` ({category} from the image's COCO
            ground truth). The model's answer concept ``a`` = its first generated token;
            target ``t`` = a COCO category annotated on OTHER images, excluded from the
            model's top-10 next-token outputs. The paper's lens-coordinate swap
            (``V = [v_a v_t]``, ``c = pinv(V) h``, ``h' = h + V (sigma(c) - c)`` with
            ``sigma`` the coordinate exchange) is applied in-forward (x16 clamp pattern)
            at workspace-band layers under THREE position scopes — TEXT positions only /
            IMAGE positions only / ALL — during prefill+decode (text/all) or prefill
            only (image). Measured: does the answer flip to ``t`` (token match and
            string match). PREDICTED: flip rate obeys TEXT ~= ALL >> IMAGE, because
            image-position J-lens readouts never carry the objects — the workspace for
            image-derived concepts lives at text positions.

modulation  MODULATION DURING CAPTIONING (dual-task). Prompt ``Concentrate on
            {concept} while describing the image.`` vs baseline ``Describe the image.``
            (10 concepts x 2 conditions). At caption positions: how often the instructed
            concept is in the J-lens top-10; AND the paper's co-occupancy statistic —
            P(an image-object category also in the top-10 | concept in top-10) vs a
            shuffled control. The paper found concept+concept interleave (0.46) but
            concept+computed-answer exclusion (0.09); the VLM question: does visual
            content behave like a CONCURRENT TASK (exclusion) or like passive context?

ignition    IMAGE-MIXING IGNITION (the ambiguous-input experiment). Blend two images'
            projected visual features ``f_mix = (1 - alpha) f_A + alpha f_B`` at the 576
            placeholder positions (hook on the multi-modal projector output; pixel
            values enter only at prefill, so the blend persists through the KV cache).
            alpha sweep 0..1 (41 steps at the default 0.025). Per layer at the handoff
            position (last prompt token): the J-lens rank of each category, and the
            projection share along the pure-A -> pure-B activation line — for the FULL
            activation and for the J-lens-vector component (projection onto the two
            categories' J-lens vector difference). The paper's sharpness statistic = the
            alpha width of the 10%-90% transition of that share. EXPECT: smooth tracking
            early, sharp all-or-none commitment emerging in the workspace band; the
            caption's committed class (which object the caption mentions first) flips at
            an alpha aligned with the lens sharpness onset (~L20-25). Per-pair threshold
            alpha + the distribution of shares at the ambiguous alpha (bimodal =
            committed per-trial) are recorded.

band        MODALITY-SPLIT WORKSPACE BAND. The paper's four layer-band statistics run
            SEPARATELY on image-token positions vs text positions over the captioning
            corpus: next-token top-k accuracy (teacher-forced: is ``ids[p+1]`` in the
            calibrated J-lens top-k at position p), readout excess kurtosis, top-1
            autocorrelation (lag-1 stickiness of the top-1 token identity) vs a shuffled
            null, and — position-independent, included for completeness — the effective
            dimensionality (participation ratio) of the category J-lens vector matrix.
            Plus the modality asymmetry that quantifies the novel claim: the rate at
            which an image-object category is in the top-10 at image vs text positions
            (prior evidence: image-position readouts stay object-free at ALL layers).

Shared helpers, one model load per invocation, per-arm digests (printed and stored in
the JSON), and a tiny CPU smoke path: ``--backend tiny --synthetic`` builds a tiny
random-weight LLaVA plus synthetic images and synthetic annotations, exercising every
arm's full pipeline without COCO files or a GPU (pass ``--layers`` within the fixture's
4 layers, e.g. ``1,2``).
"""

from __future__ import annotations

import argparse
import json
import math
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

import torch  # noqa: E402

import vlm_lens  # noqa: E402, F401  # installs the vendored jlens path: must precede jlens
from jlens.hooks import ActivationRecorder  # noqa: E402
from jlens.lens import JacobianLens  # noqa: E402

from vlm_lens.artifacts import load_lens_set  # noqa: E402
from vlm_lens.data.captions import (  # noqa: E402
    DEFAULT_QUESTIONS,
    list_coco_images,
    select_images,
)
from vlm_lens.interventions import generate_with_edits  # noqa: E402
from vlm_lens.models.llava import MultimodalBatch  # noqa: E402

import vlm_registry as registry  # noqa: E402

#: Any registry-built lens model (``LlavaLensModel`` for the llava kinds,
#: ``Qwen2VLLensModel`` for qwen2vl -- same public surface), so the annotations
#: stay honest without importing the qwen2vl class at module scope.
LensModel = Any

from x13_cure_test import (  # noqa: E402
    category_in_caption,
    detect_mentions,
    image_id_of,
    load_ground_truth,
)
from x14_workspace_probe import (  # noqa: E402
    _calibrated_lens_logits,
    _category_token_ids,
    _rank_of_id,
    generate_with_residual_traces,
    identity_lens,
    load_calibrated_params,
)

DEFAULT_LENS_DIR = "/data/anhnq/vlm-lens-out/validation/s2-merged/artifacts"
DEFAULT_BIAS_DIR = "/data/anhnq/vlm-lens-out/validation/step5e"
DEFAULT_IMAGES_DIR = "/data/baodq/coco2014/val2014"
DEFAULT_ANNOTATIONS = "/data/baodq/coco2014/annotations/instances_val2014.json"

#: Residual readout layers default from the registry's per-model band: four stride-4
#: layers inside the band for report/modulation (llava-7b: 16, 20, 24, 28 of [12, 29]),
#: and a wider stride-4 grid from band_lo through the final layer (the readout layer's
#: own block) for ignition/band so the band's emergence is visible.

#: Position scopes for the report arm. ``text``/``all`` recur on the single-token decode
#: forwards (x16's DECODE_GROUPS convention); image positions exist only in the prefill.
SCOPES: tuple[str, ...] = ("text", "image", "all")

#: The fixed 10-concept list for the modulation arm (COCO categories, so the
#: category-token machinery and the word matcher apply to them directly).
DEFAULT_CONCEPTS: tuple[str, ...] = (
    "dog", "cat", "car", "person", "cup", "chair", "bird", "horse", "bus", "pizza",
)

#: Synthetic-annotation category pool (real COCO names; ids are 1-based indices).
SYNTH_CATEGORIES: tuple[str, ...] = (
    "dog", "cat", "car", "bus", "person", "bird", "horse", "chair",
    "cup", "pizza", "truck", "sheep", "boat", "train", "cow", "bicycle",
)

REPORT_QUESTION_TEMPLATE = "What {category} is in the picture? Answer with one word:"
MODULATION_BASELINE_QUESTION = "Describe the image."
MODULATION_INSTRUCTED_TEMPLATE = "Concentrate on {concept} while describing the image."

TOPK_RULE = (
    "top-k = the k highest calibrated J-lens logits at the position; membership counts "
    "a token's emission id (last subword of ' ' + name) among the k ids"
)
SWAP_NOTE = (
    "the paper's lens-coordinate swap: V = [v_a v_t] (columns), c = pinv(V) @ h, "
    "sigma(c) = flip(c) (the two coordinates exchange), h' = h + swap_alpha * V @ "
    "(sigma(c) - c); v_x = (W_U J_l)[x] rows (interventions.lens_vectors convention, "
    "keyed by token id instead of string). Applied in-forward by per-layer hooks on "
    "model.layers[l] (x16 SinkClamper pattern): the prefill forward swaps every row in "
    "the arm's position mask; each single-token decode forward swaps the new token's "
    "row for the text/all scopes (image positions do not recur under KV caching)"
)
HANDOFF_NOTE = (
    "the handoff position is the LAST PROMPT token (the prefill forward's decoding "
    "position, whose distribution predicts generated token 0; the x14 step_records[0] "
    "convention) — here the row residual[-1] of the recorded prefill"
)
TARGET_RULE = (
    "target t: a COCO category annotated on OTHER images (never on this one), not the "
    "answer's own category, whose emission id is not among the model's top-10 "
    "next-token ids at the handoff position and whose name does not word-match the "
    "decoded top-10; tiers relax the top-10 exclusions in order (string+id -> id-only "
    "-> none) and the tier used is recorded; the choice is seeded per image"
)
SHARE_RULE = (
    "projection share along the pure-A -> pure-B activation line at the handoff "
    "position: share_full(alpha) = <h_a - h_B, h_A - h_B>/||h_A - h_B||^2 for the full "
    "residual; share_lens(alpha) = (<h_a - h_B, u>)/(<h_A - h_B, u>) with "
    "u = v_A - v_B the two categories' J-lens vector difference — both are the fraction "
    "of the way from the pure-B to the pure-A state, one measured in the full space and "
    "one along the categories' J-lens direction"
)
SHARPNESS_RULE = (
    "the paper's sharpness statistic = the alpha width of the 10%-90% transition of the "
    "share curve (linear interpolation on the measured alpha grid; smaller = sharper "
    "all-or-none commitment, ~0.8 = smooth linear tracking); None when the curve never "
    "crosses both levels; the threshold alpha is the lower crossing"
)
BAND_STATS_RULE = (
    "next-token top-k accuracy: teacher-forced, position p predicts ids[p+1] (the last "
    "prompt position has no in-prompt successor and is excluded); readout excess "
    "kurtosis: mean over the modality's positions of kurtosis(logit_row) - 3; top-1 "
    "stickiness: P(top1[p+1] == top1[p]) over positionally consecutive same-modality "
    "positions, vs a shuffled null (the modality's top-1 ids permuted over its "
    "positions, the same pair set recomputed; analytic permutation mean + empirical "
    "std over --n-null-shuffles per image); effective dimensionality: participation "
    "ratio of the covariance of the category J-lens vectors (position-independent)"
)
MATCHER_NOTE = (
    "x13 detect_mentions matcher: one whole-sequence decode, lowercased, category names "
    "matched longest-first with word boundaries, optional (s|es) plural suffix, span "
    "masking (so 'hot dog' does not also match 'dog'); irregular plurals are a "
    "documented gap"
)
SYNTH_NOTE = (
    "--synthetic builds the corpus in-memory: --n-images deterministic random RGB "
    "images, ids 1..n, each annotated with 2 categories cycled from the 16-name "
    "SYNTH_CATEGORIES pool (i mod 16 and (i+5) mod 16), so the disjoint pairing for the "
    "ignition arm and the other-images target pool for the report arm both work; no "
    "COCO files are touched"
)


@dataclass(frozen=True)
class Sample:
    """One corpus item: identity, ground truth, and the image source for encode_mm."""

    name: str  # COCO file name, or "synth_%04d.jpg"
    image_id: int  # COCO image id, or the 1-based synthetic index
    image: Any  # Path for COCO, PIL.Image for synthetic
    gt_ids: set[int]  # annotated category ids (ground truth)


def gt_name_set(sample: Sample, name_by_id: dict[int, str]) -> set[str]:
    """The sample's annotated category names."""
    return {name_by_id[cid] for cid in sample.gt_ids if cid in name_by_id}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", required=True, choices=("report", "modulation", "ignition", "band"),
        help="which VLM-only arm to run (one model load per invocation)",
    )
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
    parser.add_argument("--lens-dir", default=DEFAULT_LENS_DIR,
                        help="J source lens dir: its fitted layers define the identity-"
                             "transport lens; if it cannot be loaded the identity lens "
                             "covers ALL layers and the readout falls back to the raw "
                             "final layer (x16 fallback convention)")
    parser.add_argument("--mask", default="text",
                        help="which lens-<mask>.pt and bias-<mask>.pt to use")
    parser.add_argument("--bias-dir", default=DEFAULT_BIAS_DIR,
                        help="directory holding bias-<mask>.pt (the step5e payload: "
                             "bias/scale, optional temp/logit_bias) applied inside "
                             "every calibrated J-lens readout")
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS,
                        help="instances_val2014.json: per-image ground-truth categories")
    parser.add_argument("--n-images", type=int, default=60,
                        help="captioning/report corpus size (the ignition arm pairs "
                             "from the full annotated list instead)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--layers", default=None,
        help="comma-separated residual layers for the arm's readouts/edits; default "
             "is the registry band subset for the model (llava-7b: 16,20,24,28 for "
             "report/modulation; 12,16,20,24,28,31 for ignition/band)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="generation budget; per-arm default (report 8, modulation "
                             "40, ignition 24; unused by band)")
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--top-k", type=int, default=10,
                        help="k for every top-k membership statistic (default 10)")
    parser.add_argument("--synthetic", action="store_true",
                        help="build the corpus in-memory (tiny smoke; see SYNTH_NOTE)")
    # -- report arm
    parser.add_argument("--scopes", default=",".join(SCOPES),
                        help="comma-separated subset of: " + ",".join(SCOPES))
    parser.add_argument("--swap-alpha", type=float, default=1.0,
                        help="swap strength in h' = h + alpha * V (sigma(c) - c)")
    # -- modulation arm
    parser.add_argument("--concepts", default=",".join(DEFAULT_CONCEPTS),
                        help="comma-separated 10-concept list (COCO category names)")
    parser.add_argument("--n-shuffles", type=int, default=20,
                        help="permutations of the concept indicator for the co-occupancy "
                             "shuffled control")
    # -- ignition arm
    parser.add_argument("--n-pairs", type=int, default=10,
                        help="category pairs; 2 disjoint image-pair trials each "
                             "(=> 2*n-pairs trials)")
    parser.add_argument("--alpha-step", type=float, default=0.025,
                        help="mixing sweep step (0..1; 0.025 => 41 steps)")
    parser.add_argument("--caption-alpha-step", type=float, default=0.125,
                        help="coarser sweep for the committed-caption runs")
    # -- band arm
    parser.add_argument("--n-null-shuffles", type=int, default=200,
                        help="permutations per (image, layer, modality) for the top-1 "
                             "stickiness null std")
    parser.add_argument("--json", required=True)
    return parser.parse_args()


# --------------------------------------------------------------------------- shared


def load_model(args: argparse.Namespace) -> LensModel:
    """The arm model: dispatched by --model via the registry, or the tiny fixture."""
    return registry.load_lens_model(args)


def load_lens_optional(
    args: argparse.Namespace,
) -> tuple[JacobianLens | None, dict[str, Any]]:
    """The identity-transport lens over ``--lens-dir``'s fitted layers, or None."""
    try:
        lenses, _ = load_lens_set(args.lens_dir)
    except Exception as exc:  # noqa: BLE001 - any load failure takes the fallback
        print(f"note: no usable lens at {args.lens_dir} ({exc}); identity lens over "
              "ALL layers, raw final-layer readout")
        return None, {"loaded": False, "dir": str(args.lens_dir), "error": str(exc)}
    if args.mask not in lenses:
        raise ValueError(
            f"mask {args.mask!r} not in {args.lens_dir} (found {sorted(lenses)})"
        )
    source = lenses[args.mask]
    if not source.source_layers:
        raise ValueError(f"lens {args.mask!r} in {args.lens_dir} has no fitted layers")
    return source, {"loaded": True, "dir": str(args.lens_dir), "mask": args.mask}


def load_bias_optional(
    args: argparse.Namespace,
) -> tuple[dict[int, torch.Tensor] | None, dict[int, float] | None,
           dict[int, float] | None, dict[int, torch.Tensor] | None, dict[str, Any]]:
    """The bias-<mask>.pt payload as x14's four dicts, or all-None on any failure."""
    try:
        bias, scale, temp, logit_bias, meta = load_calibrated_params(
            args.bias_dir, args.mask
        )
    except Exception as exc:  # noqa: BLE001 - any load failure skips the corrections
        print(f"note: no usable bias payload at {args.bias_dir} ({exc}); the calibrated "
              "readout runs without bias/scale/temp/logit_bias")
        return None, None, None, None, {
            "loaded": False, "dir": str(args.bias_dir), "error": str(exc),
        }
    return bias, scale, temp, logit_bias, {"loaded": True, "dir": str(args.bias_dir)}



def drop_mismatched_bias(
    model: LensModel,
    kind: str,
    args: argparse.Namespace,
    bias: dict[int, torch.Tensor] | None,
    scale: dict[int, float] | None,
    temp: dict[int, float] | None,
    logit_bias: dict[int, torch.Tensor] | None,
    bias_meta: dict[str, Any],
) -> tuple[dict[int, torch.Tensor] | None, dict[int, float] | None,
           dict[int, float] | None, dict[int, torch.Tensor] | None, dict[str, Any]]:
    """Fail closed on a bias payload saved for a DIFFERENT model (x16 fallback style).

    The step5e payload is per-model: each layer's ``bias`` row is added to the
    d_model residual and each ``logit_bias`` row to the vocab-dim logits, so a
    payload saved for another checkpoint (e.g. the 7B moment census run against
    llava-13b or qwen2vl) crashes the calibrated readout's residual add instead of
    merely misreading. Every tensor is validated against the dimension it is
    applied at (bias -> model.d_model, logit_bias -> vocab); on any mismatch the
    WHOLE payload is dropped (bias/scale/temp/logit_bias are one per-model
    artifact) and the arm runs uncalibrated with the drop recorded in ``bias_meta``.
    """
    if bias is None and logit_bias is None:
        return bias, scale, temp, logit_bias, bias_meta
    vocab = int(model.unembed_weight().shape[0])
    mismatches = []
    for name, expected, tensors in (
        ("bias", model.d_model, (bias or {}).values()),
        ("logit_bias", vocab, (logit_bias or {}).values()),
    ):
        wrong = sorted({int(t.shape[-1]) for t in tensors if int(t.shape[-1]) != expected})
        if wrong:
            mismatches.append(f"{name} last dim {wrong} != {expected}")
    if not mismatches:
        return bias, scale, temp, logit_bias, bias_meta
    detail = "; ".join(mismatches)
    print(
        f"note: dropping bias payload at {args.bias_dir} for model {kind!r} "
        f"(d_model {model.d_model}, vocab {vocab}): {detail}; the calibrated "
        "readout runs without bias/scale/temp/logit_bias"
    )
    return None, None, None, None, {
        **bias_meta,
        "loaded": False,
        "dropped": "dim_mismatch",
        "model_kind": kind,
        "model_d_model": int(model.d_model),
        "model_vocab": vocab,
        "note": f"payload dims do not match this model; dropped, arm runs "
                f"uncalibrated: {detail}",
    }


def synthetic_samples(n: int) -> list[Sample]:
    """Deterministic random images + synthetic annotations (see SYNTH_NOTE)."""
    from PIL import Image

    from vlm_lens.models.tiny_llava import random_image

    samples = []
    for index in range(n):
        array = random_image(seed=1000 + index)
        image = Image.fromarray(array)
        gt = {index % len(SYNTH_CATEGORIES) + 1,
              (index + 5) % len(SYNTH_CATEGORIES) + 1}
        samples.append(Sample(f"synth_{index:04d}.jpg", index + 1, image, gt))
    return samples


def name_map(annotations: str | None, synthetic: bool) -> dict[int, str]:
    """Category id -> name for the corpus (COCO's 80, or the synthetic 16)."""
    if synthetic:
        return {index + 1: name for index, name in enumerate(SYNTH_CATEGORIES)}
    names, _ = load_ground_truth(annotations)  # type: ignore[arg-type]
    return names


@dataclass
class Workspace:
    """Everything the arms share: model kind, calibrated lens, corpus, category tables."""

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
    samples: list[Sample]
    top_k: int
    lens_meta: dict[str, Any]
    bias_meta: dict[str, Any]


def build_workspace(args: argparse.Namespace) -> Workspace:
    """One model load + lens/bias loads + corpus and category tables for the arm."""
    kind = registry.kind_for(args)
    layers = default_layers(args, kind)
    model = load_model(args)
    invalid = [layer for layer in layers if not 0 <= layer < model.n_layers]
    if invalid:
        raise ValueError(
            f"layers {invalid} outside 0..{model.n_layers - 1} (backend "
            f"{args.backend!r} model {kind!r}, n_layers={model.n_layers}); "
            "pass --layers within range"
        )
    final_layer = model.n_layers - 1
    source_lens, lens_meta = load_lens_optional(args)
    bias, scale, temp, logit_bias, bias_meta = load_bias_optional(args)
    device = model.unembed_weight().device
    bias, scale, temp, logit_bias, bias_meta = drop_mismatched_bias(
        model, kind, args, bias, scale, temp, logit_bias, bias_meta
    )
    # Same fail-closed class for the lens: an identity-transport lens fitted on
    # another checkpoint would crash transport's residual @ J.T the same way, so a
    # d_model mismatch takes the documented no-lens fallback (identity over ALL
    # layers, raw final-layer readout) with the drop recorded.
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

    if args.synthetic:
        samples = synthetic_samples(args.n_images)
    elif args.arm == "ignition":
        samples = coco_samples(
            args.images_dir, args.annotations, args.n_images, args.seed, full=True
        )
    else:
        samples = [
            sample
            for sample in coco_samples(
                args.images_dir, args.annotations, args.n_images, args.seed, full=False
            )
            if sample.gt_ids
        ]

    name_by_id = name_map(args.annotations, args.synthetic)
    all_names = sorted({name for name in name_by_id.values()})
    emission = dict(zip(all_names, _category_token_ids(model.tokenizer, all_names)))
    return Workspace(
        kind=kind, model=model, lens=lens, bias=bias, scale=scale, temp=temp,
        logit_bias=logit_bias, readout_layer=readout_layer, layers=layers,
        device=device, calibrated=calibrated, name_by_id=name_by_id,
        emission=emission, samples=samples, top_k=args.top_k,
        lens_meta=lens_meta, bias_meta=bias_meta,
    )


def coco_samples(
    images_dir: str, annotations: str, n_images: int, seed: int, *, full: bool
) -> list[Sample]:
    """COCO samples with per-image ground truth (x13 loaders; x14 id mapping)."""
    _, by_image = load_ground_truth(annotations)
    files = list_coco_images(images_dir)
    if not full:
        files = select_images(files, n_images, seed=seed)
    samples = []
    for path in files:
        image_id = image_id_of(path)
        samples.append(Sample(path.name, image_id, path, set(by_image.get(image_id, set()))))
    return samples


def default_layers(args: argparse.Namespace, kind: str) -> list[int]:
    """The arm's layer list (band subset for report/modulation, wider sweep otherwise).

    Defaults come from the registry band at the spec's depth (llava-7b reproduces
    the old hardcoded (16, 20, 24, 28) and (12, 16, 20, 24, 28, 31) exactly); the
    ``--layers`` flag remains the override.
    """
    if args.layers is not None:
        return [int(part) for part in args.layers.split(",") if part.strip()]
    n_layers = int(registry.MODEL_SPECS[kind]["n_layers"])
    base = (
        registry.band_layers(kind, n_layers)
        if args.arm in ("report", "modulation")
        else registry.sweep_layers(kind, n_layers)
    )
    return list(base)


def max_new_tokens(args: argparse.Namespace) -> int:
    """The arm's generation budget when --max-new-tokens is not given."""
    if args.max_new_tokens is not None:
        return args.max_new_tokens
    return {"report": 8, "modulation": 40, "ignition": 24, "band": 0}[args.arm]


def scope_masks(batch: MultimodalBatch) -> dict[str, torch.Tensor]:
    """Boolean ``[seq_len]`` selectors for image/text/all (x16 conventions: text
    excludes bos and image; all is every position)."""
    image = batch.image_token_mask[0].detach()
    bos = torch.zeros_like(image)
    bos[0] = True
    return {
        "image": image,
        "text": ~(image | bos),
        "all": torch.ones_like(image),
    }


def lens_vectors_ids(
    model: LensModel,
    lens: JacobianLens,
    token_ids: list[int],
    layers: list[int],
) -> dict[int, torch.Tensor]:
    """J-lens vectors ``v = (W_U J_l)[t]`` for token *ids* (interventions.lens_vectors
    with ids instead of strings): ``{layer: [n_ids, d_model] fp32 CPU}``. Layers the
    lens did not fit read through an identity transport."""
    weight = model.unembed_weight().detach().float()  # [vocab, d_model]
    out: dict[int, torch.Tensor] = {}
    for layer in layers:
        if layer in lens.jacobians:
            jacobian = lens.jacobians[layer].to(weight.device).float()
        else:
            jacobian = torch.eye(lens.d_model, dtype=torch.float32, device=weight.device)
        out[layer] = (weight[token_ids] @ jacobian).cpu()
    return out


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _fmt(value: float | None, width: int, precision: int = 3) -> str:
    return f"{value:>{width}.{precision}f}" if value is not None else "-" * width


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation, None when either side is constant/empty/short."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom = math.sqrt(sum(a * a for a in dx) * sum(b * b for b in dy))
    if denom <= 1e-12:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denom


# ------------------------------------------------------------------ arm a: report


@dataclass(frozen=True)
class SwapSpec:
    """One layer's swap spec for the prefill/decode hooks (x16 LayerClamp shape)."""

    basis: torch.Tensor  # [d, 2] fp32: columns v_a, v_t
    pinv: torch.Tensor  # [2, d] fp32
    mask: torch.Tensor  # [prompt_len] bool: prefill positions to swap
    decode: bool  # also swap the single-token decode forwards (text/all scopes)
    alpha: float


class SwapPatcher:
    """Forward hooks applying the paper's lens-coordinate swap at chosen positions.

    x16 SinkClamper pattern: hooks on ``model.layers[l]`` rewrite the block output.
    The prefill forward (sequence length == the prompt length) swaps every row in the
    spec's mask; each single-token decode forward swaps the new token's row when
    ``SwapSpec.decode`` (text/all scopes; image positions do not recur under KV
    caching). The prefill fire count is asserted == 1 per swapped layer.
    """

    def __init__(
        self, model: LensModel, prompt_len: int, specs: dict[int, SwapSpec]
    ) -> None:
        self._model = model
        self._prompt_len = int(prompt_len)
        self._specs = specs
        self._handles: list[Any] = []
        self.n_prefill_fires: dict[int, int] = {}
        self.n_decode_swaps = 0

    def _make_hook(self, layer: int, spec: SwapSpec) -> Any:
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            tensor = output if torch.is_tensor(output) else output[0]
            is_prefill = tensor.shape[1] == self._prompt_len
            if is_prefill:
                self.n_prefill_fires[layer] = self.n_prefill_fires.get(layer, 0) + 1
            if not (is_prefill or spec.decode):
                return output
            basis = spec.basis.to(tensor.device, torch.float32)
            pinv = spec.pinv.to(tensor.device, torch.float32)
            if is_prefill:
                rows = tensor[0, spec.mask.to(tensor.device)].float()  # [n, d]
            else:
                rows = tensor[0, -1].float().unsqueeze(0)  # [1, d]
            coords = rows @ pinv.T  # [n, 2]
            delta = (coords.flip(-1) - coords) @ basis.T  # [n, d]
            swapped = (rows + spec.alpha * delta).to(tensor.dtype)
            modified = tensor.clone()
            if is_prefill:
                modified[0, spec.mask.to(tensor.device)] = swapped
            else:
                modified[0, -1] = swapped[0]
                self.n_decode_swaps += 1
            if torch.is_tensor(output):
                return modified
            return (modified, *output[1:])

        return hook

    def __enter__(self) -> SwapPatcher:
        for layer, spec in self._specs.items():
            self._handles.append(
                self._model.layers[layer].register_forward_hook(
                    self._make_hook(layer, spec)
                )
            )
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        return None

    def assert_fired(self) -> None:
        for layer in self._specs:
            fires = self.n_prefill_fires.get(layer, 0)
            if fires != 1:
                raise RuntimeError(
                    f"expected exactly one prefill fire at layer {layer}, got {fires}"
                )


def generate_swapped(
    model: LensModel,
    batch: MultimodalBatch,
    specs: dict[int, SwapSpec],
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Greedy answer under the arm's swap (x16 generate_clamped pattern)."""
    patcher = SwapPatcher(model, batch.seq_len, specs)
    with patcher:
        generated = model.hf_model.generate(
            **batch.hf_kwargs(),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    patcher.assert_fired()
    new_tokens = generated[0, batch.seq_len :].detach().cpu()
    return {
        "text": model._decode(new_tokens),
        "token_ids": [int(token) for token in new_tokens],
        "n_decode_swaps": patcher.n_decode_swaps,
    }


def choose_target(
    candidates: list[str],
    gt_names: set[str],
    answer_category: str | None,
    answer_id: int,
    top10_ids: list[int],
    top10_text: str,
    emission: dict[str, int],
    rng: random.Random,
) -> tuple[str | None, int]:
    """Pick the target category t (see TARGET_RULE); returns (name, tier).

    Tiers relax the top-10 exclusions in order; tier 4 is the last-resort pool (still
    never the answer's own category). None only when even tier 4 is empty.
    """
    ordered = sorted(candidates)
    tiers = [
        [name for name in ordered
         if name not in gt_names
         and name != answer_category
         and emission[name] != answer_id
         and emission[name] not in top10_ids
         and not category_in_caption(name, top10_text)],
        [name for name in ordered
         if name not in gt_names
         and name != answer_category
         and emission[name] != answer_id
         and emission[name] not in top10_ids],
        [name for name in ordered
         if name not in gt_names
         and name != answer_category
         and emission[name] != answer_id],
        [name for name in ordered if name != answer_category],
    ]
    for tier, pool in enumerate(tiers, start=1):
        if pool:
            return rng.choice(pool), tier
    return None, 5


def run_report(args: argparse.Namespace, ws: Workspace) -> dict[str, Any]:
    """Arm (a): cross-modal verbal-report swap under TEXT/IMAGE/ALL position scopes."""
    model = ws.model
    tokenizer = model.tokenizer
    names = ws.name_by_id
    all_names = sorted(ws.emission)
    layers = ws.layers
    scopes = [part.strip() for part in args.scopes.split(",") if part.strip()]
    unknown = [scope for scope in scopes if scope not in SCOPES]
    if unknown:
        raise ValueError(f"unknown scopes {unknown}; choose from {list(SCOPES)}")
    budget = max_new_tokens(args)
    print(
        f"== X23 arm=report: {len(ws.samples)} images, {len(layers)} layers x "
        f"{len(scopes)} scopes, swap_alpha={args.swap_alpha}, "
        f"max_new_tokens={budget} =="
    )

    results: list[dict[str, Any]] = []
    cells: dict[tuple[int, str], dict[str, list[bool]]] = {
        (layer, scope): {"flip_token": [], "flip_string": [], "kept": [], "changed": []}
        for layer in layers
        for scope in scopes
    }
    n_skipped_no_target = 0
    tier_counts: dict[int, int] = {}

    for index, sample in enumerate(ws.samples):
        gt_names = gt_name_set(sample, names)
        if not gt_names:
            continue
        category = random.Random(args.seed + index).choice(sorted(gt_names))
        question = REPORT_QUESTION_TEMPLATE.format(category=category)
        prompt = registry.image_prompt(ws.kind, question)
        batch = registry.image_encode(model, ws.kind, prompt, sample.image, max_length=args.max_seq_len)

        baseline = generate_with_edits(model, None, batch, max_new_tokens=budget)
        answer_ids = baseline["token_ids"]
        answer_id = answer_ids[0]
        answer_text = baseline["text"]
        answer_category = next(
            (name for name in all_names if category_in_caption(name, answer_text)),
            None,
        )

        with torch.no_grad():
            model_logits = model.hf_model(**batch.hf_kwargs()).logits
        top10_ids = [
            int(token)
            for token in model_logits[0, -1]
            .topk(min(args.top_k, model_logits.shape[-1]))
            .indices.tolist()
        ]
        top10_text = " ".join(tokenizer.decode([token]) for token in top10_ids)

        target, tier = choose_target(
            all_names, gt_names, answer_category, answer_id, top10_ids, top10_text,
            ws.emission, random.Random(args.seed * 1_000_003 + index),
        )
        if target is None:
            print(f"  [{index + 1}] {sample.name}: no target candidate - skipped")
            n_skipped_no_target += 1
            continue
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        target_id = ws.emission[target]
        vectors = lens_vectors_ids(model, ws.lens, [answer_id, target_id], layers)
        first_word = answer_text.split()[0] if answer_text.split() else answer_text

        print(
            f"  [{index + 1}] {sample.name}: q={category!r} answer={first_word!r} "
            f"(id {answer_id}) -> target {target!r} (id {target_id}, tier {tier})"
        )

        masks = scope_masks(batch)
        for layer in layers:
            pair = vectors[layer].float()  # [2, d]: rows v_a, v_t
            basis = pair.T.contiguous()  # [d, 2]: columns v_a, v_t
            pinv = torch.linalg.pinv(basis)
            cos = float(
                abs(torch.dot(pair[0], pair[1]))
                / (pair[0].norm() * pair[1].norm()).clamp_min(1e-12)
            )
            for scope in scopes:
                spec = SwapSpec(
                    basis=basis,
                    pinv=pinv,
                    mask=masks[scope],
                    decode=scope in ("text", "all"),
                    alpha=args.swap_alpha,
                )
                out = generate_swapped(
                    model, batch, {layer: spec}, max_new_tokens=budget
                )
                swapped_ids = out["token_ids"]
                flip_token = bool(swapped_ids) and swapped_ids[0] == target_id
                flip_string = category_in_caption(target, out["text"])
                kept = bool(swapped_ids) and swapped_ids[0] == answer_id
                changed = swapped_ids != answer_ids
                cell = cells[(layer, scope)]
                cell["flip_token"].append(flip_token)
                cell["flip_string"].append(flip_string)
                cell["kept"].append(kept)
                cell["changed"].append(changed)
                results.append({
                    "image": sample.name,
                    "image_id": sample.image_id,
                    "prompt_category": category,
                    "question": question,
                    "answer_id": answer_id,
                    "answer_text": answer_text,
                    "answer_category": answer_category,
                    "target": target,
                    "target_id": target_id,
                    "target_tier": tier,
                    "vector_cos_abs": _round(cos),
                    "layer": layer,
                    "scope": scope,
                    "n_swap_positions_prefill": int(masks[scope].sum()),
                    "n_decode_swaps": out["n_decode_swaps"],
                    "text": out["text"],
                    "token_ids": swapped_ids,
                    "flip_token": flip_token,
                    "flip_string": flip_string,
                    "answer_kept": kept,
                    "changed": changed,
                })

    table: dict[str, dict[str, dict[str, Any]]] = {}
    for layer in layers:
        row: dict[str, dict[str, Any]] = {}
        for scope in scopes:
            cell = cells[(layer, scope)]
            row[scope] = {
                "n": len(cell["flip_token"]),
                "flip_token_rate": _round(_mean([float(v) for v in cell["flip_token"]])),
                "flip_string_rate": _round(_mean([float(v) for v in cell["flip_string"]])),
                "answer_kept_rate": _round(_mean([float(v) for v in cell["kept"]])),
                "changed_rate": _round(_mean([float(v) for v in cell["changed"]])),
            }
        table[str(layer)] = row

    pooled = {
        scope: _round(_mean([
            float(value)
            for layer in layers
            for value in cells[(layer, scope)]["flip_token"]
        ]))
        for scope in scopes
    }
    lines = [
        f"== X23 arm=report: verbal-report swap, {len(results)} arm rows ==",
        f"{'layer':<7}{'scope':<8}{'n':>5}{'flip_tok':>9}{'flip_str':>9}"
        f"{'kept':>9}{'changed':>9}",
    ]
    for layer in layers:
        for scope in scopes:
            cell = table[str(layer)][scope]
            lines.append(
                f"L{layer:<6}{scope:<8}{cell['n']:>5}"
                f"{_fmt(cell['flip_token_rate'], 9)}"
                f"{_fmt(cell['flip_string_rate'], 9)}"
                f"{_fmt(cell['answer_kept_rate'], 9)}"
                f"{_fmt(cell['changed_rate'], 9)}"
            )
    lines.append(
        "pooled flip_token rates: "
        + "  ".join(f"{scope}={pooled[scope]}" for scope in scopes)
    )
    lines.append("predicted ordering: TEXT ~= ALL >> IMAGE")
    digest = "\n".join(lines)

    summary = {
        "n_images_processed": len({row["image"] for row in results}),
        "n_arms": len(layers) * len(scopes),
        "n_results": len(results),
        "n_skipped_no_target": n_skipped_no_target,
        "target_tier_counts": tier_counts,
        "pooled_flip_token_rate": pooled,
    }
    return {
        "per_layer_scope": table,
        "digest": digest,
        "summary": summary,
        "results": results,
        "notes": {
            "swap": SWAP_NOTE,
            "handoff": HANDOFF_NOTE,
            "target": TARGET_RULE,
            "matcher": MATCHER_NOTE,
            "topk": TOPK_RULE,
        },
        "interpretation_limits": [
            "the prompt NAMES one of the image's annotated categories, so the baseline "
            "answer is usually that category (or a variant); the swap tests whether the "
            "answer concept can be redirected once it exists, not comprehension",
            "'What {plural-only category} is in the picture?' is ungrammatical for a few "
            "COCO names (scissors, skis); the template is kept verbatim by design",
            "flip_string scans the whole answer for the target word (an answer like 'a "
            "dog' flips by string even when token 0 is the article); flip_token is the "
            "strict first-token readout",
            "only text/all scopes swap decode steps: with KV caching the decode forward "
            "carries exactly the new (text) token, and image positions do not recur",
            "a single swap layer is armed at a time; interactions between simultaneous "
            "layers are untested (x16 convention)",
            "the tiny backend is a random-weight fixture: it exercises the machinery, "
            "its flip rates do not transfer to the real model",
        ],
    }


# ------------------------------------------------------------ arm b: modulation


def run_topk_indicators(
    ws: Workspace,
    step_records: list[dict[int, torch.Tensor]],
    *,
    concept_id: int,
    object_ids: list[int],
) -> dict[int, dict[str, Any]]:
    """Per-layer, per-step top-k indicators along one traced generation.

    Returns ``{layer: {n_steps, concept_top10 (list[bool]), visual_top10
    (list[bool]), top1_ids (list[int])}}`` from the calibrated J-lens readout of each
    decoding step's cached residual (x14's step convention: step i predicts token i).
    """
    model = ws.model
    k = min(ws.top_k, int(model.unembed_weight().shape[0]))
    object_tensor = torch.tensor(sorted(set(object_ids)))
    out: dict[int, dict[str, Any]] = {}
    for layer in ws.layers:
        hidden = torch.stack(
            [step_records[step][layer].float() for step in range(len(step_records))]
        ).to(ws.device)  # [T, d]
        logits = _calibrated_lens_logits(
            model, ws.lens, hidden, layer, ws.bias, ws.scale, ws.temp, ws.logit_bias
        )  # [T, vocab] cpu float
        top = logits.topk(k, dim=-1).indices  # [T, k]
        if object_tensor.numel():
            visual_top10 = torch.isin(top, object_tensor).any(dim=-1).tolist()
        else:
            visual_top10 = [False] * top.shape[0]
        out[layer] = {
            "n_steps": int(top.shape[0]),
            "concept_top10": (top == concept_id).any(dim=-1).tolist(),
            "visual_top10": visual_top10,
            "top1_ids": [int(token) for token in top[:, 0].tolist()],
        }
    return out


def cooccur_stats(
    concept_mask: list[bool],
    visual_mask: list[bool],
    n_shuffles: int,
    rng: random.Random,
) -> dict[str, Any]:
    """The paper's co-occupancy statistic plus its shuffled control.

    Observed: P(visual | concept). Shuffled: the same conditional after re-drawing
    which caption steps carry the concept indicator (same count, random positions),
    ``n_shuffles`` times. z = (observed - shuffled_mean) / shuffled_std.
    """
    concept_steps = [i for i, flag in enumerate(concept_mask) if flag]
    if not concept_steps:
        return {
            "n_concept_steps": 0, "p_visual_given_concept": None,
            "p_visual_given_not_concept": None, "shuffled_mean": None,
            "shuffled_std": None, "z": None,
        }
    observed = _mean([float(visual_mask[i]) for i in concept_steps])
    not_concept = _mean(
        [float(visual_mask[i]) for i, flag in enumerate(concept_mask) if not flag]
    )
    total = len(concept_mask)
    shuffles = []
    for _ in range(max(n_shuffles, 1)):
        picked = rng.sample(range(total), len(concept_steps))
        shuffles.append(_mean([float(visual_mask[i]) for i in picked]) or 0.0)
    mean_s = _mean(shuffles) or 0.0
    std_s = (
        math.sqrt(sum((s - mean_s) ** 2 for s in shuffles) / len(shuffles))
        if len(shuffles) > 1 else 0.0
    )
    z = (observed - mean_s) / std_s if std_s > 1e-12 else None
    return {
        "n_concept_steps": len(concept_steps),
        "p_visual_given_concept": _round(observed),
        "p_visual_given_not_concept": _round(not_concept),
        "shuffled_mean": _round(mean_s),
        "shuffled_std": _round(std_s),
        "z": _round(z),
    }


def run_modulation(args: argparse.Namespace, ws: Workspace) -> dict[str, Any]:
    """Arm (b): modulation during captioning — concept-vs-visual co-occupancy."""
    model = ws.model
    concepts = [part.strip() for part in args.concepts.split(",") if part.strip()]
    if not concepts:
        raise ValueError("--concepts must name at least one concept")
    budget = max_new_tokens(args)
    layers = ws.layers
    print(
        f"== X23 arm=modulation: {len(ws.samples)} images x "
        f"({len(concepts)} concepts x 2 conditions), layers={layers}, "
        f"top_k={args.top_k}, max_new_tokens={budget} =="
    )

    # per (layer): observed/shuffled co-occupancy pools across all instructed runs
    pools: dict[str, dict[str, list[float]]] = {
        str(layer): {"cooc": [], "cooc_not": [], "shuffled": [], "z": [],
                     "concept_rate": [], "visual_rate": []}
        for layer in layers
    }
    per_concept_layer: dict[str, dict[str, dict[str, Any]]] = {
        concept: {
            str(layer): {"n_runs": 0, "cooc": [], "concept_rate": [], "visual_rate": []}
            for layer in layers
        }
        for concept in concepts
    }
    baseline_layer: dict[str, dict[str, list[float | None]]] = {
        str(layer): {"concept_rate": [], "visual_rate": []} for layer in layers
    }
    results: list[dict[str, Any]] = []

    for index, sample in enumerate(ws.samples):
        object_names = sorted(gt_name_set(sample, ws.name_by_id))
        object_ids = [ws.emission[name] for name in object_names if name in ws.emission]
        assigned_concept = concepts[index % len(concepts)]

        # ---- baseline condition (concept-independent prompt, once per image)
        baseline_prompt = registry.image_prompt(ws.kind, MODULATION_BASELINE_QUESTION)
        batch = registry.image_encode(
            model, ws.kind, baseline_prompt, sample.image, max_length=args.max_seq_len
        )
        new_tokens, step_records = generate_with_residual_traces(
            model, batch, layers, max_new_tokens=budget
        )
        baseline_text = model._decode(new_tokens)
        base_indicators = run_topk_indicators(
            ws, step_records,
            concept_id=ws.emission[assigned_concept], object_ids=object_ids,
        )
        for layer in layers:
            ind = base_indicators[layer]
            baseline_layer[str(layer)]["concept_rate"].append(
                _mean([float(v) for v in ind["concept_top10"]])
            )
            baseline_layer[str(layer)]["visual_rate"].append(
                _mean([float(v) for v in ind["visual_top10"]])
            )
        print(
            f"  [{index + 1}] {sample.name}: baseline {baseline_text[:60]!r} "
            f"(objects {object_names})"
        )
        results.append({
            "image": sample.name, "condition": "baseline",
            "concept": assigned_concept, "prompt": baseline_prompt,
            "text": baseline_text,
            "mentions_concept": category_in_caption(assigned_concept, baseline_text),
            "visual_mentions": [
                name for name in detect_mentions(ws.name_by_id, baseline_text)
                if name in set(object_names)
            ],
            "per_layer": {
                str(layer): {
                    "n_steps": base_indicators[layer]["n_steps"],
                    "concept_top10_rate": _round(_mean(
                        [float(v) for v in base_indicators[layer]["concept_top10"]]
                    )),
                    "visual_top10_rate": _round(_mean(
                        [float(v) for v in base_indicators[layer]["visual_top10"]]
                    )),
                }
                for layer in layers
            },
        })

        # ---- instructed conditions (one per concept)
        for concept_index, concept in enumerate(concepts):
            question = MODULATION_INSTRUCTED_TEMPLATE.format(concept=concept)
            prompt = registry.image_prompt(ws.kind, question)
            batch = registry.image_encode(model, ws.kind, prompt, sample.image, max_length=args.max_seq_len)
            new_tokens, step_records = generate_with_residual_traces(
                model, batch, layers, max_new_tokens=budget
            )
            text = model._decode(new_tokens)
            indicators = run_topk_indicators(
                ws, step_records,
                concept_id=ws.emission[concept], object_ids=object_ids,
            )
            per_layer_out: dict[str, Any] = {}
            for layer in layers:
                ind = indicators[layer]
                stats = cooccur_stats(
                    ind["concept_top10"], ind["visual_top10"], args.n_shuffles,
                    random.Random(args.seed * 7 + index * 131 + concept_index),
                )
                concept_rate = _mean([float(v) for v in ind["concept_top10"]])
                visual_rate = _mean([float(v) for v in ind["visual_top10"]])
                pool = pools[str(layer)]
                if stats["p_visual_given_concept"] is not None:
                    pool["cooc"].append(stats["p_visual_given_concept"])
                    pool["cooc_not"].append(stats["p_visual_given_not_concept"] or 0.0)
                    pool["shuffled"].append(stats["shuffled_mean"] or 0.0)
                    if stats["z"] is not None:
                        pool["z"].append(stats["z"])
                pool["concept_rate"].append(concept_rate or 0.0)
                pool["visual_rate"].append(visual_rate or 0.0)
                bucket = per_concept_layer[concept][str(layer)]
                bucket["n_runs"] += 1
                if stats["p_visual_given_concept"] is not None:
                    bucket["cooc"].append(stats["p_visual_given_concept"])
                bucket["concept_rate"].append(concept_rate or 0.0)
                bucket["visual_rate"].append(visual_rate or 0.0)
                per_layer_out[str(layer)] = {
                    **stats,
                    "concept_top10_rate": _round(concept_rate),
                    "visual_top10_rate": _round(visual_rate),
                }
            results.append({
                "image": sample.name, "condition": "instructed", "concept": concept,
                "prompt": prompt, "text": text,
                "mentions_concept": category_in_caption(concept, text),
                "visual_mentions": [
                    name for name in detect_mentions(ws.name_by_id, text)
                    if name in set(object_names)
                ],
                "per_layer": per_layer_out,
            })

    concept_layer_table: dict[str, dict[str, dict[str, Any]]] = {
        concept: {
            str(layer): {
                "n_runs": per_concept_layer[concept][str(layer)]["n_runs"],
                "p_visual_given_concept_mean": _round(
                    _mean(per_concept_layer[concept][str(layer)]["cooc"])
                ),
                "concept_top10_rate_mean": _round(
                    _mean(per_concept_layer[concept][str(layer)]["concept_rate"])
                ),
                "visual_top10_rate_mean": _round(
                    _mean(per_concept_layer[concept][str(layer)]["visual_rate"])
                ),
            }
            for layer in layers
        }
        for concept in concepts
    }
    layer_table: dict[str, dict[str, Any]] = {}
    for layer in layers:
        pool = pools[str(layer)]
        layer_table[str(layer)] = {
            "p_visual_given_concept_mean": _round(_mean(pool["cooc"])),
            "p_visual_given_not_concept_mean": _round(_mean(pool["cooc_not"])),
            "shuffled_mean": _round(_mean(pool["shuffled"])),
            "z_mean": _round(_mean(pool["z"])),
            "concept_top10_rate_mean": _round(_mean(pool["concept_rate"])),
            "visual_top10_rate_mean": _round(_mean(pool["visual_rate"])),
            "n_cooc_measurements": len(pool["cooc"]),
            "baseline_concept_top10_rate": _round(_mean(
                [v for v in baseline_layer[str(layer)]["concept_rate"] if v is not None]
            )),
            "baseline_visual_top10_rate": _round(_mean(
                [v for v in baseline_layer[str(layer)]["visual_rate"] if v is not None]
            )),
        }

    lines = [
        f"== X23 arm=modulation: {len(concepts)} concepts x 2 conditions, "
        f"caption-position co-occupancy ==",
        f"{'layer':<7}{'p(conc)':>9}{'p(vis|c)':>9}{'p(vis|~c)':>10}{'shuf':>9}"
        f"{'z':>7}{'base_p(c)':>10}{'base_vis':>9}",
    ]
    for layer in layers:
        row = layer_table[str(layer)]
        lines.append(
            f"L{layer:<6}{_fmt(row['concept_top10_rate_mean'], 9)}"
            f"{_fmt(row['p_visual_given_concept_mean'], 9)}"
            f"{_fmt(row['p_visual_given_not_concept_mean'], 10)}"
            f"{_fmt(row['shuffled_mean'], 9)}"
            f"{_fmt(row['z_mean'], 7, 2)}"
            f"{_fmt(row['baseline_concept_top10_rate'], 10)}"
            f"{_fmt(row['baseline_visual_top10_rate'], 9)}"
        )
    lines.append(
        "the VLM question: does visual content behave like a concurrent task "
        "(z << 0, exclusion) or like passive context (z ~ 0)?"
    )
    digest = "\n".join(lines)

    n_instructed = sum(1 for row in results if row["condition"] == "instructed")
    summary = {
        "n_images_processed": len({row["image"] for row in results}),
        "n_instructed_runs": n_instructed,
        "n_baseline_runs": len(results) - n_instructed,
        "concepts": concepts,
        "concept_mention_rate_instructed": _round(_mean([
            float(row["mentions_concept"])
            for row in results if row["condition"] == "instructed"
        ])),
    }
    return {
        "per_layer": layer_table,
        "per_concept_layer": concept_layer_table,
        "digest": digest,
        "summary": summary,
        "results": results,
        "notes": {
            "topk": TOPK_RULE,
            "cooccurrence": (
                "the paper's dual-task metric, transposed: at caption positions where "
                "the instructed concept is in the J-lens top-10, P(an image-object "
                "category is also in the top-10), vs a shuffled control that re-draws "
                "which caption steps carry the concept indicator (same count); the "
                "paper's text-LLM values: concept+concept 0.46 (interleave), "
                "concept+computed-answer 0.09 (exclude)"
            ),
            "handoff": HANDOFF_NOTE,
            "matcher": MATCHER_NOTE,
        },
        "interpretation_limits": [
            "the concept indicator uses the concept's emission token id only (last "
            "subword of ' ' + name); a caption paraphrase ('puppy') is invisible to "
            "the lens-side indicator and to the behavioral string match alike",
            "the shuffled control permutes the concept indicator WITHIN one caption, "
            "so it tests temporal alignment, not cross-caption base rates",
            "baseline runs use the concept assigned to the image "
            "(concepts[image_index % len(concepts)]), so the baseline concept rate is "
            "a spontaneous-rate contrast, not a per-concept controlled one",
            "image-object categories use the image's COCO ground truth; a mentioned "
            "object outside the annotation set counts as neither concept nor visual "
            "hit",
            "the tiny backend is a random-weight fixture: it exercises the machinery, "
            "its rates do not transfer to the real model",
        ],
    }


# ------------------------------------------------------------- arm c: ignition


def _projector_features(output: Any) -> torch.Tensor:
    """The projected-features tensor a hooked registry vision module returns.

    llava's ``multi_modal_projector`` returns the tensor directly. qwen2vl's
    vision tower (transformers 5.x) returns a ``BaseModelOutputWithPooling`` whose
    ``pooler_output`` is the merged embeds -- the exact tensor the LM scatters into
    the input embeddings (``last_hidden_state`` is the pre-merge patch states).
    """
    if isinstance(output, torch.Tensor):
        return output
    return output.pooler_output


def capture_projector_output(model: LensModel, kind: str, batch: MultimodalBatch) -> torch.Tensor:
    """The projected visual features for one image (no LM forward needed).

    A hook on the registry's vision module (llava: ``multi_modal_projector``;
    qwen2vl: the vision tower whose merged-embed output plays the same role) records
    the module output during a standalone vision-only call; this is the exact tensor
    the prefill merges into the input embeddings, so blending it IS blending the
    projected visual features at the image placeholder positions.
    """
    module = registry.vision_projector(model, kind)
    box: dict[str, torch.Tensor] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        box["out"] = _projector_features(output).detach().clone()
        return output

    handle = module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            registry.vision_only(model, kind, batch)
    finally:
        handle.remove()
    if "out" not in box:
        raise RuntimeError("projector capture hook never fired")
    return box["out"]


class ProjectorMixer:
    """Hook on the registry's vision module: out' = (1 - alpha) out + alpha donor.

    The blend replaces the projected visual features of the batch's own image before
    they merge into the input embeddings. With KV caching the module fires on the
    prefill forward only, so the mix persists through decode.
    """

    def __init__(self, model: LensModel, kind: str, donor: torch.Tensor, alpha: float) -> None:
        self._module = registry.vision_projector(model, kind)
        self._donor = donor
        self.alpha = float(alpha)
        self.n_fires = 0
        self._handle: Any = None

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        features = _projector_features(output)
        if tuple(features.shape) != tuple(self._donor.shape):
            raise RuntimeError(
                f"projector output {tuple(features.shape)} != donor "
                f"{tuple(self._donor.shape)}; both images must expand to the same "
                "number of placeholder tokens"
            )
        blended = (1.0 - self.alpha) * features.float() + self.alpha * self._donor.float()
        self.n_fires += 1
        blended = blended.to(features.dtype)
        if isinstance(output, torch.Tensor):
            return blended
        output.pooler_output = blended  # qwen2vl tower: the merged embeds live here
        return output

    def __enter__(self) -> ProjectorMixer:
        self._handle = self._module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc: Any) -> None:
        self._handle.remove()
        self._handle = None
        return None

    def assert_fired(self) -> None:
        if self.n_fires != 1:
            raise RuntimeError(
                f"expected exactly one projector fire, got {self.n_fires}"
            )


def alpha_grid(step: float) -> list[float]:
    """The sweep ``[0, step, ..., 1]`` (>= 2 points)."""
    if step <= 0 or step > 1:
        raise ValueError(f"alpha step must be in (0, 1], got {step}")
    n = max(int(round(1.0 / step)), 1)
    grid = [round(i / n, 10) for i in range(n + 1)]
    grid[-1] = 1.0
    return grid


def transition_width(
    alphas: list[float], shares: list[float]
) -> tuple[float | None, float | None, float | None]:
    """The alpha width of the share curve's 10%-90% transition (see SHARPNESS_RULE).

    Returns (width, alpha_lo, alpha_hi); (None, None, None) when the curve never
    crosses both levels in either direction.
    """
    if any(math.isnan(value) for value in shares):
        return None, None, None

    def crossing(level: float) -> float | None:
        for (a0, s0), (a1, s1) in zip(zip(alphas, shares), zip(alphas[1:], shares[1:])):
            if (s0 - level) * (s1 - level) <= 0 and s0 != s1:
                return a0 + (level - s0) / (s1 - s0) * (a1 - a0)
        return None

    lo, hi = crossing(0.1), crossing(0.9)
    if lo is None or hi is None:
        return None, None, None
    width = abs(hi - lo)
    if width <= 1e-12:  # degenerate: both crossings on one grid interval's edge
        return None, None, None
    return width, min(lo, hi), max(lo, hi)


def select_disjoint_pairs(
    samples: list[Sample], name_by_id: dict[int, str], n_trials: int
) -> list[tuple[Sample, Sample]]:
    """x12's greedy first-disjoint pairing over the annotated samples.

    Walks the name-sorted sample list; each not-yet-paired image A with a non-empty
    category set pairs with the first not-yet-paired later image B whose category name
    set is disjoint from A's. Returns at most ``n_trials`` (A, B) tuples.
    """
    def names_of(sample: Sample) -> set[str]:
        return gt_name_set(sample, name_by_id)

    ordered = sorted(samples, key=lambda sample: sample.name)
    used = [False] * len(ordered)
    pairs: list[tuple[Sample, Sample]] = []
    for index_a, sample_a in enumerate(ordered):
        if used[index_a]:
            continue
        names_a = names_of(sample_a)
        if not names_a:
            continue
        for index_b in range(index_a + 1, len(ordered)):
            if used[index_b]:
                continue
            sample_b = ordered[index_b]
            names_b = names_of(sample_b)
            if not names_b or not names_a.isdisjoint(names_b):
                continue
            pairs.append((sample_a, sample_b))
            used[index_a] = used[index_b] = True
            break
        if len(pairs) >= n_trials:
            break
    return pairs


def _first_match_pos(name: str, text: str) -> int | None:
    """Character offset of the first plural-tolerant word-boundary match."""
    match = re.search(rf"\b{re.escape(name)}(?:s|es)?\b", text.lower())
    return match.start() if match else None


def committed_class(text: str, name_a: str, name_b: str) -> str:
    """Which of the two objects the caption mentions first: 'A' / 'B' / 'neither'."""
    pos_a = _first_match_pos(name_a, text)
    pos_b = _first_match_pos(name_b, text)
    if pos_a is None and pos_b is None:
        return "neither"
    if pos_b is None:
        return "A"
    if pos_a is None:
        return "B"
    return "A" if pos_a <= pos_b else "B"


def caption_flip_alpha(captions: list[dict[str, Any]]) -> float | None:
    """Midpoint alpha of the committed class's first transition (None when none)."""
    singles = [
        (entry["alpha"], entry["committed_class"])
        for entry in captions
        if entry["committed_class"] in ("A", "B")
    ]
    if len(singles) < 2:
        return None
    first_class = singles[0][1]
    for position in range(1, len(singles)):
        if singles[position][1] != first_class:
            return round((singles[position - 1][0] + singles[position][0]) / 2, 6)
    return None


def _representative_categories(
    ws: Workspace,
    handoff_a: dict[int, torch.Tensor],
    handoff_b: dict[int, torch.Tensor],
    names_a: set[str],
    names_b: set[str],
    layer: int,
) -> tuple[str, str]:
    """Each image's committed-object category: the annotated category with the best
    (lowest) calibrated J-lens rank at the handoff position in its OWN pure run
    (alpha = 0 for A, alpha = 1 for B, at ``layer``); ties broken by name order."""
    def best(names: set[str], handoff: dict[int, torch.Tensor]) -> str:
        ordered = sorted(name for name in names if name in ws.emission)
        logits = _calibrated_lens_logits(
            ws.model, ws.lens, handoff[layer].to(ws.device), layer,
            ws.bias, ws.scale, ws.temp, ws.logit_bias,
        )
        ranked = sorted(
            (_rank_of_id(logits, ws.emission[name]), name) for name in ordered
        )
        return ranked[0][1]

    return best(names_a, handoff_a), best(names_b, handoff_b)


def run_ignition(args: argparse.Namespace, ws: Workspace) -> dict[str, Any]:
    """Arm (c): image-mixing ignition — alpha sweep, shares, sharpness, commitment."""
    model = ws.model
    layers = ws.layers
    # The readout layer this arm reads out at: the lens's own readout layer when the
    # sweep covers it, else the sweep's highest layer at/below it, else the sweep's
    # highest (the fitted lens's deepest layer need not be one of the sweep layers —
    # the box lens reads out at L30 while the ignition/band sweep steps 12..31).
    at_or_below = [layer for layer in layers if layer <= ws.readout_layer]
    readout_layer = max(at_or_below) if at_or_below else max(layers)
    budget = max_new_tokens(args)
    alphas = alpha_grid(args.alpha_step)
    caption_alphas = alpha_grid(args.caption_alpha_step)
    n_trials = 2 * args.n_pairs
    question = DEFAULT_QUESTIONS[0]
    prompt = registry.image_prompt(ws.kind, question)
    pairs = select_disjoint_pairs(ws.samples, ws.name_by_id, n_trials)
    print(
        f"== X23 arm=ignition: {len(pairs)} trials, {len(alphas)} mix alphas "
        f"(captions at {len(caption_alphas)}), layers={layers}, "
        f"max_new_tokens={budget} =="
    )

    per_layer_acc: dict[int, dict[str, list[float]]] = {
        layer: {"width_lens": [], "width_full": [], "thr_lens": [],
                "caption_flip": [], "thr_for_flip": [], "committed_share": []}
        for layer in layers
    }
    trial_rows: list[dict[str, Any]] = []

    for trial_index, (sample_a, sample_b) in enumerate(pairs):
        names_a = gt_name_set(sample_a, ws.name_by_id)
        names_b = gt_name_set(sample_b, ws.name_by_id)
        batch_a = registry.image_encode(model, ws.kind, prompt, sample_a.image, max_length=args.max_seq_len)
        batch_b = registry.image_encode(model, ws.kind, prompt, sample_b.image, max_length=args.max_seq_len)
        donor = capture_projector_output(model, ws.kind, batch_b)

        # ---- alpha sweep: one recorded prefill per alpha (the blend fires once)
        handoff: dict[float, dict[int, torch.Tensor]] = {}
        for alpha in alphas:
            mixer = ProjectorMixer(model, ws.kind, donor, alpha)
            with mixer, ActivationRecorder(model.layers, at=layers) as recorder:
                model.forward_mm(batch_a)
            mixer.assert_fired()
            handoff[alpha] = {
                layer: recorder.activations[layer][0, -1].detach().float().cpu()
                for layer in layers
            }

        rep_a, rep_b = _representative_categories(
            ws, handoff[alphas[0]], handoff[alphas[-1]], names_a, names_b,
            readout_layer,
        )
        vector_a = lens_vectors_ids(model, ws.lens, [ws.emission[rep_a]], layers)
        vector_b = lens_vectors_ids(model, ws.lens, [ws.emission[rep_b]], layers)

        # ---- per-layer share curves + per-alpha J-lens ranks at the handoff position
        per_layer_rows: dict[str, Any] = {}
        for layer in layers:
            u = (vector_a[layer][0] - vector_b[layer][0]).to(ws.device)  # [d]
            h_stack = torch.stack([handoff[a][layer] for a in alphas]).to(ws.device)
            h_pure_a = handoff[alphas[0]][layer].to(ws.device)
            h_pure_b = handoff[alphas[-1]][layer].to(ws.device)
            diff = h_pure_a - h_pure_b
            denom_full = float(diff.norm()) ** 2
            denom_lens = float(torch.dot(diff, u))
            lens_values = ((h_stack - h_pure_b) @ u).tolist()
            full_values = ((h_stack - h_pure_b) @ diff).tolist()
            share_lens = [
                value / denom_lens if abs(denom_lens) > 1e-9 else float("nan")
                for value in lens_values
            ]
            share_full = [
                value / denom_full if denom_full > 1e-12 else float("nan")
                for value in full_values
            ]
            width_l, thr_l, _ = transition_width(alphas, share_lens)
            width_f, thr_f, _ = transition_width(alphas, share_full)
            ranks = []
            for alpha in alphas:
                logits = _calibrated_lens_logits(
                    model, ws.lens, handoff[alpha][layer].to(ws.device), layer,
                    ws.bias, ws.scale, ws.temp, ws.logit_bias,
                )
                ranks.append({
                    "alpha": alpha,
                    "rank_A": _rank_of_id(logits, ws.emission[rep_a]),
                    "rank_B": _rank_of_id(logits, ws.emission[rep_b]),
                })
            per_layer_rows[str(layer)] = {
                "width_lens": _round(width_l),
                "width_full": _round(width_f),
                "threshold_alpha_lens": _round(thr_l),
                "threshold_alpha_full": _round(thr_f),
                "share_lens": [
                    None if math.isnan(v) else round(v, 4) for v in share_lens
                ],
                "share_full": [
                    None if math.isnan(v) else round(v, 4) for v in share_full
                ],
                "ranks": ranks,
            }
            if width_l is not None:
                per_layer_acc[layer]["width_lens"].append(width_l)
                per_layer_acc[layer]["thr_lens"].append(thr_l)
            if width_f is not None:
                per_layer_acc[layer]["width_full"].append(width_f)
            if thr_l is not None:
                nearest = min(range(len(alphas)), key=lambda i: abs(alphas[i] - thr_l))
                value = share_lens[nearest]
                if not math.isnan(value):
                    per_layer_acc[layer]["committed_share"].append(value)

        # ---- committed captions on the coarse grid
        captions: list[dict[str, Any]] = []
        for alpha in caption_alphas:
            mixer = ProjectorMixer(model, ws.kind, donor, alpha)
            with mixer:
                generated = model.hf_model.generate(
                    **batch_a.hf_kwargs(),
                    max_new_tokens=budget,
                    do_sample=False,
                    use_cache=True,
                )
            mixer.assert_fired()
            new_tokens = generated[0, batch_a.seq_len :].detach().cpu()
            text = model._decode(new_tokens)
            captions.append({
                "alpha": alpha,
                "text": text,
                "committed_class": committed_class(text, rep_a, rep_b),
            })
        flip = caption_flip_alpha(captions)
        thr_at_readout = per_layer_rows[str(readout_layer)]["threshold_alpha_lens"]
        if flip is not None and thr_at_readout is not None:
            per_layer_acc[readout_layer]["caption_flip"].append(flip)
            per_layer_acc[readout_layer]["thr_for_flip"].append(thr_at_readout)

        print(
            f"  [{trial_index + 1}] {sample_a.name} + {sample_b.name}: A={rep_a!r} "
            f"B={rep_b!r} width_lens(L{readout_layer})="
            f"{per_layer_rows[str(readout_layer)]['width_lens']} "
            f"caption_flip={flip}"
        )
        trial_rows.append({
            "trial": trial_index,
            "image_a": sample_a.name,
            "image_b": sample_b.name,
            "categories_a": sorted(names_a),
            "categories_b": sorted(names_b),
            "rep_a": rep_a,
            "rep_b": rep_b,
            "caption_flip_alpha": flip,
            "captions": captions,
            "per_layer": per_layer_rows,
        })

    layer_table: dict[str, dict[str, Any]] = {}
    for layer in layers:
        acc = per_layer_acc[layer]
        committed = acc["committed_share"]
        layer_table[str(layer)] = {
            "n_transitions_lens": len(acc["width_lens"]),
            "n_trials": len(trial_rows),
            "width_lens_mean": _round(_mean(acc["width_lens"])),
            "width_lens_median": _round(_median(acc["width_lens"])),
            "width_full_mean": _round(_mean(acc["width_full"])),
            "threshold_alpha_lens_mean": _round(_mean(acc["thr_lens"])),
            "caption_flip_alpha_mean": _round(_mean(acc["caption_flip"])),
            "threshold_vs_caption_pearson": _round(pearson(
                acc["thr_for_flip"], acc["caption_flip"]
            )),
            "share_at_threshold_mean": _round(_mean(committed)),
            "share_at_threshold_std": _round(_std(committed)),
            "share_at_threshold_frac_committed": _round(
                _mean([1.0 if abs(v - 0.5) > 0.25 else 0.0 for v in committed])
            ),
            "share_at_threshold_histogram": _histogram(committed, 10),
        }

    lines = [
        f"== X23 arm=ignition: {len(trial_rows)} trials, alpha sweep "
        f"{alphas[0]}..{alphas[-1]} =="
        if trial_rows else "== X23 arm=ignition: no trials ==",
        f"{'layer':<7}{'w_lens':>9}{'w_full':>9}{'thr_a':>8}{'cap_a':>8}"
        f"{'r_thr_cap':>10}{'committed':>10}{'n':>5}",
    ]
    for layer in layers:
        row = layer_table[str(layer)]
        lines.append(
            f"L{layer:<6}{_fmt(row['width_lens_mean'], 9)}"
            f"{_fmt(row['width_full_mean'], 9)}"
            f"{_fmt(row['threshold_alpha_lens_mean'], 8)}"
            f"{_fmt(row['caption_flip_alpha_mean'], 8)}"
            f"{_fmt(row['threshold_vs_caption_pearson'], 10, 2)}"
            f"{_fmt(row['share_at_threshold_frac_committed'], 10)}"
            f"{row['n_transitions_lens']:>5}"
        )
    lines.append(
        "expectation: width_lens shrinks and commitment sharpens across the workspace "
        "band (~L20-25); caption flip alphas align with the lens thresholds"
    )
    digest = "\n".join(lines)

    summary = {
        "n_trials": len(trial_rows),
        "n_alphas": len(alphas),
        "n_caption_alphas": len(caption_alphas),
    }
    return {
        "per_layer": layer_table,
        "digest": digest,
        "summary": summary,
        "results": trial_rows,
        "notes": {
            "share": SHARE_RULE,
            "sharpness": SHARPNESS_RULE,
            "handoff": HANDOFF_NOTE,
            "mixing": (
                "f_mix = (1 - alpha) f_A + alpha f_B replaces the projected visual "
                "features at the 576 placeholder positions (hook on the multi-modal "
                "projector output, the masked_scatter input); pixel values enter only "
                "at prefill, so the blend persists through decode via the KV cache"
            ),
            "matcher": MATCHER_NOTE,
        },
        "interpretation_limits": [
            "the committed class uses the REPRESENTATIVE category per image (the "
            "annotated category with the best J-lens rank in its own pure run at the "
            "readout layer); other annotated objects also influence the caption",
            "the sharpness width is interpolated on the measured alpha grid, so it is "
            "bounded below by the grid resolution (default 0.025)",
            "share_lens is undefined when the categories' J-lens vector difference is "
            "orthogonal to the A->B activation line (recorded as null widths)",
            "bimodality at the threshold pools each trial's share at ITS OWN "
            "threshold alpha (nearest grid point); frac_committed counts shares "
            "outside [0.25, 0.75] as committed per-trial",
            "greedy decoding makes the caption commitment deterministic per trial; "
            "bimodality here is ACROSS trials (image pairs), not across samples",
            "the tiny backend is a random-weight fixture: it exercises the machinery, "
            "its widths and thresholds do not transfer to the real model",
        ],
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _histogram(values: list[float], bins: int) -> list[int]:
    """Counts of ``values`` in ``bins`` equal-width bins over [0, 1] (clamped)."""
    counts = [0] * bins
    for value in values:
        index = min(max(int(value * bins), 0), bins - 1)
        counts[index] += 1
    return counts


# ------------------------------------------------------------------ arm d: band


@torch.no_grad()
def position_readout(
    ws: Workspace, residual: torch.Tensor, layer: int, *, chunk: int = 128
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-position calibrated J-lens readout over a full-sequence residual.

    ``residual`` is ``[seq, d]`` cpu float32 (one recorded prefill). Returns
    ``(top1_ids [seq], topk_ids [seq, k], excess_kurtosis [seq])`` on CPU; the
    [seq, vocab] logit tensor exists only per chunk (bounded memory).
    """
    model = ws.model
    k = min(ws.top_k, int(model.unembed_weight().shape[0]))
    seq = int(residual.shape[0])
    top1 = torch.empty(seq, dtype=torch.long)
    topk_ids = torch.empty(seq, k, dtype=torch.long)
    kurt = torch.empty(seq)
    for start in range(0, seq, chunk):
        block = residual[start : start + chunk].to(ws.device)
        logits = _calibrated_lens_logits(
            model, ws.lens, block, layer, ws.bias, ws.scale, ws.temp, ws.logit_bias
        )
        top = logits.topk(k, dim=-1)
        topk_ids[start : start + chunk] = top.indices.cpu()
        top1[start : start + chunk] = top.indices[:, 0].cpu()
        centered = logits - logits.mean(dim=-1, keepdim=True)
        variance = centered.pow(2).mean(dim=-1)
        kurt[start : start + chunk] = (
            centered.pow(4).mean(dim=-1) / variance.clamp_min(1e-12).pow(2) - 3
        ).cpu()
    return top1, topk_ids, kurt


def participation_ratio(vectors: torch.Tensor) -> float | None:
    """Effective dimensionality of a ``[n, d]`` vector set: the participation ratio
    ``(sum lambda)^2 / sum lambda^2`` of its covariance eigenvalues."""
    if vectors.shape[0] < 2:
        return None
    centered = vectors - vectors.mean(dim=0, keepdim=True)
    singulars = torch.linalg.svdvals(centered)
    lam = singulars**2
    total = float(lam.sum())
    if total <= 1e-12:
        return None
    return float(total**2 / float((lam**2).sum()))


def modality_stickiness(
    top1: torch.Tensor, mask: torch.Tensor
) -> tuple[float | None, torch.Tensor, torch.Tensor]:
    """Observed lag-1 stickiness over positionally consecutive same-modality pairs.

    Returns (observed, values, valid) where ``values`` is the modality's top-1 ids in
    position order and ``valid[i]`` marks value pairs (i, i+1) whose positions are
    consecutive — the exact pair set the null re-uses.
    """
    seq = int(top1.shape[0])
    mask = mask.detach().cpu()
    values = top1[mask]
    linked = torch.zeros(seq, dtype=torch.bool)
    linked[:-1] = mask[:-1] & mask[1:]
    valid = linked[mask][:-1]  # pair (i, i+1) exists iff position idx[i] is linked
    if values.numel() < 2 or not bool(valid.any()):
        return None, values, valid
    observed = float((values[1:][valid] == values[:-1][valid]).float().mean())
    return observed, values, valid


def stickiness_null(
    values: torch.Tensor, valid: torch.Tensor, n_shuffles: int, rng: random.Random
) -> tuple[float | None, float | None]:
    """The shuffled null of the modality stickiness (see BAND_STATS_RULE).

    The modality's top-1 ids are permuted over its positions (marginals preserved,
    temporal structure broken) and the observed statistic recomputed on the SAME pair
    set. Mean: the analytic permutation value ``(sum_c n_c (n_c - 1)) / (n (n - 1))``;
    std: empirical over ``n_shuffles`` permutations.
    """
    n = int(values.numel())
    if n < 2 or not bool(valid.any()):
        return None, None
    counts = torch.bincount(values).float()
    analytic = float((counts * (counts - 1)).sum() / (n * (n - 1)))
    pool = values.tolist()
    flags = valid.tolist()
    n_pairs = int(valid.sum())
    samples = []
    for _ in range(max(n_shuffles, 1)):
        rng.shuffle(pool)
        samples.append(
            sum(1 for i, flag in enumerate(flags) if flag and pool[i + 1] == pool[i])
            / n_pairs
        )
    return analytic, _std(samples)


def run_band(args: argparse.Namespace, ws: Workspace) -> dict[str, Any]:
    """Arm (d): the paper's four layer-band statistics, split by modality."""
    model = ws.model
    layers = ws.layers
    question = DEFAULT_QUESTIONS[0]
    prompt = registry.image_prompt(ws.kind, question)
    object_ids_all = sorted(set(ws.emission.values()))
    object_tensor = torch.tensor(object_ids_all)
    effective_dim = {
        str(layer): _round(participation_ratio(
            lens_vectors_ids(model, ws.lens, object_ids_all, [layer])[layer]
        ))
        for layer in layers
    }
    print(
        f"== X23 arm=band: {len(ws.samples)} images, layers={layers}, "
        f"top_k={ws.top_k} (position-wise readouts, no generation) =="
    )

    # per (layer, modality): pools over images
    pools: dict[tuple[int, str], dict[str, list[float]]] = {
        (layer, modality): {
            "topk_acc": [], "kurt": [], "stick": [], "null_mean": [],
            "null_std": [], "object_rate": [], "n_positions": [],
        }
        for layer in layers
        for modality in ("image", "text")
    }
    n_skipped = 0

    for index, sample in enumerate(ws.samples):
        if not sample.gt_ids:
            n_skipped += 1
            continue
        batch = registry.image_encode(model, ws.kind, prompt, sample.image, max_length=args.max_seq_len)
        masks = scope_masks(batch)
        image_mask = masks["image"]
        text_mask = masks["text"]
        with ActivationRecorder(model.layers, at=layers) as recorder:
            model.forward_mm(batch)
        residuals = {
            layer: recorder.activations[layer][0].detach().float().cpu()
            for layer in layers
        }
        input_ids = batch.input_ids[0]
        next_ids = input_ids[1:].cpu()  # position p's teacher-forced target (p < seq - 1); cpu: joined with the cpu top-k ids at the isin below
        object_ids = [
            ws.emission[name]
            for name in sorted(gt_name_set(sample, ws.name_by_id))
            if name in ws.emission
        ]
        if object_ids:
            sample_objects = torch.tensor(object_ids)
        else:
            sample_objects = object_tensor  # fall back to the full category pool

        for layer in layers:
            top1, topk_ids, kurt = position_readout(ws, residuals[layer], layer)
            seq = int(top1.shape[0])
            next_in_top = torch.isin(topk_ids[:-1], next_ids.unsqueeze(-1)).any(-1)
            object_hit = torch.isin(topk_ids, sample_objects).any(-1)
            for modality, mask in (("image", image_mask), ("text", text_mask)):
                pool = pools[(layer, modality)]
                positions = mask.detach().cpu().nonzero(as_tuple=True)[0]
                if positions.numel() == 0:
                    continue
                next_valid = positions[positions < seq - 1]
                if next_valid.numel():
                    pool["topk_acc"].append(
                        float(next_in_top[next_valid].float().mean())
                    )
                pool["kurt"].append(float(kurt[positions].mean()))
                stick, values, pair_valid = modality_stickiness(top1, mask)
                if stick is not None:
                    null_mean, null_std = stickiness_null(
                        values, pair_valid, args.n_null_shuffles,
                        random.Random(args.seed * 17 + index * 977 + layer),
                    )
                    pool["stick"].append(stick)
                    if null_mean is not None:
                        pool["null_mean"].append(null_mean)
                    if null_std is not None:
                        pool["null_std"].append(null_std)
                pool["object_rate"].append(float(object_hit[positions].float().mean()))
                pool["n_positions"].append(float(positions.numel()))
        if (index + 1) % 10 == 0 or index + 1 == len(ws.samples):
            print(f"  [{index + 1}/{len(ws.samples)}] {sample.name} read out", flush=True)

    layer_table: dict[str, dict[str, Any]] = {}
    for layer in layers:
        entry: dict[str, Any] = {"effective_dim_category_vectors": effective_dim[str(layer)]}
        for modality in ("image", "text"):
            pool = pools[(layer, modality)]
            null_means = pool["null_mean"]
            null_stds = pool["null_std"]
            z = None
            if pool["stick"] and null_means and null_stds:
                observed = _mean(pool["stick"])
                null = _mean(null_means)
                spread = _mean(null_stds) or 0.0
                if observed is not None and null is not None and spread > 1e-12:
                    z = (observed - null) / spread
            entry[modality] = {
                "n_images": len(pool["kurt"]),
                "n_positions_mean": _round(_mean(pool["n_positions"]), 1),
                "next_token_topk_acc": _round(_mean(pool["topk_acc"])),
                "readout_excess_kurtosis": _round(_mean(pool["kurt"])),
                "top1_stickiness": _round(_mean(pool["stick"])),
                "stickiness_null_mean": _round(_mean(null_means)),
                "stickiness_null_std": _round(_mean(null_stds)),
                "stickiness_z": _round(z),
                "object_in_top10_rate": _round(_mean(pool["object_rate"])),
            }
        image_row, text_row = entry["image"], entry["text"]
        asymmetry = None
        if (
            image_row["object_in_top10_rate"] is not None
            and text_row["object_in_top10_rate"] is not None
        ):
            asymmetry = text_row["object_in_top10_rate"] - image_row["object_in_top10_rate"]
        entry["object_rate_asymmetry_text_minus_image"] = _round(asymmetry)
        layer_table[str(layer)] = entry

    lines = [
        "== X23 arm=band: modality-split workspace statistics ==",
        f"{'layer':<7}{'acc_img':>9}{'acc_txt':>9}{'kurt_img':>10}{'kurt_txt':>10}"
        f"{'stick_img':>10}{'stick_txt':>10}{'z_img':>7}{'z_txt':>7}"
        f"{'obj_img':>9}{'obj_txt':>9}{'eff_dim':>9}",
    ]
    for layer in layers:
        row = layer_table[str(layer)]
        lines.append(
            f"L{layer:<6}{_fmt(row['image']['next_token_topk_acc'], 9)}"
            f"{_fmt(row['text']['next_token_topk_acc'], 9)}"
            f"{_fmt(row['image']['readout_excess_kurtosis'], 10, 1)}"
            f"{_fmt(row['text']['readout_excess_kurtosis'], 10, 1)}"
            f"{_fmt(row['image']['top1_stickiness'], 10)}"
            f"{_fmt(row['text']['top1_stickiness'], 10)}"
            f"{_fmt(row['image']['stickiness_z'], 7, 2)}"
            f"{_fmt(row['text']['stickiness_z'], 7, 2)}"
            f"{_fmt(row['image']['object_in_top10_rate'], 9)}"
            f"{_fmt(row['text']['object_in_top10_rate'], 9)}"
            f"{_fmt(row['effective_dim_category_vectors'], 9, 1)}"
        )
    lines.append(
        "novel claim: the workspace band for image-position content is delayed/absent "
        "relative to text positions — obj_img should stay ~0 at ALL layers while "
        "obj_txt rises through the band (prior evidence: image-position readouts stay "
        "object-free at all layers)"
    )
    digest = "\n".join(lines)

    summary = {
        "n_images_processed": sum(
            row["image"]["n_images"] for row in layer_table.values()
        ) // max(len(layers), 1),
        "n_images_skipped_no_annotations": n_skipped,
        "max_object_rate_asymmetry_layer": max(
            layers,
            key=lambda layer: layer_table[str(layer)][
                "object_rate_asymmetry_text_minus_image"
            ] or 0.0,
        ) if layers else None,
    }
    return {
        "per_layer": layer_table,
        "digest": digest,
        "summary": summary,
        "results": [],
        "notes": {
            "band_stats": BAND_STATS_RULE,
            "topk": TOPK_RULE,
            "modality_split": (
                "image = the 576 placeholder positions; text = all non-image non-bos "
                "prompt positions (x16's groups; the bos position belongs to neither "
                "split); all statistics use the calibrated J-lens readout at each "
                "position"
            ),
        },
        "interpretation_limits": [
            "next-token accuracy is teacher-forced over in-prompt successors: at image "
            "positions the target is almost always the next placeholder id, so the "
            "image-side accuracy measures placeholder predictability, not object "
            "knowledge — the object_in_top10_rate columns carry the modality claim",
            "the top-1 stickiness null is a permutation of the modality's own top-1 "
            "sequence, so it controls for the token marginals but not for position "
            "semantics",
            "effective dimensionality is position-independent by construction (the "
            "category J-lens vector matrix); it is reported per layer for completeness "
            "only",
            "the captioning prompt's text positions are few (~20 per image), so "
            "text-side pools are ~30x smaller than image-side pools per image",
            "the tiny backend is a random-weight fixture: it exercises the machinery, "
            "its rates do not transfer to the real model",
        ],
    }


# --------------------------------------------------------------------------- main


ARM_RUNNERS = {
    "report": run_report,
    "modulation": run_modulation,
    "ignition": run_ignition,
    "band": run_band,
}


def main() -> int:
    args = parse_args()
    ws = build_workspace(args)
    runner = ARM_RUNNERS[args.arm]
    result = runner(args, ws)
    report: dict[str, Any] = {
        "script": Path(__file__).name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "arm": args.arm,
        "backend": args.backend,
        "synthetic": args.synthetic,
        "images_dir": None if args.synthetic else str(args.images_dir),
        "annotations": None if args.synthetic else str(args.annotations),
        "n_images_requested": args.n_images,
        "n_samples": len(ws.samples),
        "seed": args.seed,
        "layers": ws.layers,
        "top_k": args.top_k,
        "max_new_tokens": max_new_tokens(args),
        "max_seq_len": args.max_seq_len,
        "prompt_template": registry.image_prompt_template(ws.kind),
        "prompt_template_hash": registry.prompt_template_sha256(ws.kind),
        "workspace_band": {
            "paper_band_32l": [12, 29],
            "handoff_completion_layers": [20, 25],
            "readout_layer": ws.readout_layer,
        },
        "lens": ws.lens_meta,
        "bias": ws.bias_meta,
        "calibrated": ws.calibrated,
        **result,
    }
    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + result["digest"])
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
