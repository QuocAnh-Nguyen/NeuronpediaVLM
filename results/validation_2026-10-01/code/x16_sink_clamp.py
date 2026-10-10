#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""X16: massive-activation census + selective sink clamping (hypothesis D44/H6).

Massive activations — the few residual dims whose |value| dwarfs every other dim —
are suspected to act as attention sinks that gate late-layer behavior. X16
quantifies them (PART 1) and tests whether they are a causal bottleneck or dead
weight (PART 2) by clamping them selectively during generation and reading out
caption change, hallucination/grounding, and the calibrated-lens handoff rank.

PART 1 — census (cheap: ONE recorded forward per image). For N images an
ActivationRecorder captures the block output of every fitted lens layer, every
``--layers`` arm layer, and the model's final layer (the pre-final-norm residual
stream; the final layer is always added so the readout is covered).
Per (layer, position group) the census accumulates, across images, the
mean |activation| of every dim over that group's positions:
    bos    position 0 (asserted to hold the tokenizer's bos id)
    image  the fused visual placeholder positions (``image_token_mask``)
    text   every other prompt position (bos and image excluded, so the three
           census groups are disjoint)
    all    every position (the union; also the "all" clamp group's ranking)
The massive-dims table reports, per (layer, group), the top ``--k`` dims by that
mean with their magnitudes, the dim's std over ALL positions, and the ratios
bos/std, image/std, text/std (mean |activation| of a group over the all-position
std). A massive dim shows a huge ratio: its value is a near-constant outlier.

PART 2 — clamp. A forward-hook clamper (x12's BlockPatcher pattern) zeroes or
median-replaces the census top-k dims of ONE (layer, group) arm during greedy
generation:
    zero    x[p, dim] := 0
    median  x[p, dim] := the arm's pooled median (below)
Arms = --layers x --positions; each arm runs alone (no other clamp armed), so
effects are attributable to that single (layer, group). Position semantics during
generation: the prefill forward carries every position, so every group fires
there; with KV caching each decode forward carries exactly the new token — a
text position — so ONLY the text and all groups also clamp decode steps (bos and
image positions do not recur; recorded as ``clamp_position_note``).

The median replacement value is the PER-POSITION-TYPE median: per image, the
median of the dim over the group's positions in that image's clean census pass,
then the median of those per-image medians across all census images. Pooling
across images is what makes median mode meaningful for bos — a within-image
median over one BOS position would return the value itself, a structural no-op
(recorded as ``median_note``).

Readouts per arm x image:
  - caption vs the unclamped baseline greedy caption: exact token-id change,
    token-length ratio, and the x13 mention matcher (``detect_mentions``) scored
    against the image's COCO ground truth: hallucination_removed_rate = the
    fraction of the baseline's hallucinated categories absent from the clamped
    caption (mean over images whose baseline hallucinated); grounded_retention =
    the fraction of the baseline's grounded categories still present (mean over
    images whose baseline grounded something).
  - the calibrated-lens handoff true-object rank (x14's --handoff-rank helper):
    at the LAST PROMPT position — the prefill decoding position whose
    distribution predicts generated token 0 — the calibrated lens distribution
    (``z = unembed(s_l * (h_l + b_l))`` + optional temp/logit_bias) is probed
    with the image's true category token ids (the last subword of ``' ' + name``);
    per image the best (lowest) rank among the categories is kept. The baseline
    rank comes from the clean census pass; the clamped rank comes from the SAME
    clamped generation pass (the clamper captures the clamped prefill residual
    at the readout layer). MEASUREMENT CHOICE (per the task): the lens measure
    clamps during a single forward — namely the generation pass's own prefill
    forward — rather than a separate recorded forward: the handoff position is a
    prefill position, so capturing it from the generation pass IS the
    single-forward measurement at zero extra cost. Readout layer: the highest
    FITTED lens layer (fully calibrated); without a lens dir, the raw final
    layer. Clamps at earlier layers propagate into the readout through
    attention — that propagation is the effect of interest; an arm whose layer
    equals the readout layer and whose group excludes the last-prompt position
    has a structurally zero lens delta (see ``interpretation_limits``).

Output (--json): {meta, census, per_layer_position, digest, summary,
results, interpretation_limits}. ``--backend tiny`` runs the whole pipeline on
the tiny CPU fixture (pass ``--layers 0,1,2`` — the fixture has 4 layers; with a
conftest-style fitted lens the census covers the fitted layers and the readout
is the highest fitted one).

Run (real, CUDA):
  python x16_sink_clamp.py --backend hf-llava --n-images 20 --json out/x16.json
Smoke (tiny, CPU, synthetic data):
  python x16_sink_clamp.py --backend tiny --layers 0,1,2 --n-images 3 \
      --images-dir <fixture>/images --annotations <fixture>/annotations.json \
      --lens-dir <fixture>/lens --bias-dir <fixture>/bias \
      --max-new-tokens 24 --k 4 --json /tmp/x16_smoke.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
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
    PROMPT_TEMPLATE,
    list_coco_images,
    prompt_template_hash,
    prompt_text,
    select_images,
)
from vlm_lens.interventions import generate_with_edits  # noqa: E402
from vlm_lens.models.llava import LlavaLensModel, MultimodalBatch  # noqa: E402

from x13_cure_test import detect_mentions, image_id_of, load_ground_truth  # noqa: E402
from x14_workspace_probe import (  # noqa: E402
    _calibrated_lens_logits,
    _category_token_ids,
    _rank_of_id,
    identity_lens,
    load_calibrated_params,
)

#: Hypothesis under test, recorded verbatim in the report.
HYPOTHESIS = (
    "D44/H6: massive activations (the few dims with enormous |value| that act as "
    "attention sinks) sit at specific positions and control late-layer behavior; "
    "clamping them selectively (zeroing or median-replacing the top-k dims by "
    "magnitude at chosen position types/layers) should change captions and/or "
    "hallucination rates non-trivially - a test of whether the sink is a causal "
    "bottleneck vs dead weight."
)

#: Position groups; bos/image/text are disjoint, ``all`` is their union.
POSITION_GROUPS: tuple[str, ...] = ("bos", "image", "text", "all")
#: Groups whose positions recur on the single-token decode forwards (the new
#: token is a text position; bos/image positions exist only in the prefill).
DECODE_GROUPS: frozenset[str] = frozenset({"text", "all"})
#: Default residual-layer sweep for the clamp arms.
DEFAULT_LAYERS: tuple[int, ...] = (16, 24, 28, 30)
#: Default top-k massive dims per (layer, group).
DEFAULT_K = 16
DEFAULT_LENS_DIR = "/data/anhnq/vlm-lens-out/validation/s2-merged/artifacts"
DEFAULT_BIAS_DIR = "/data/anhnq/vlm-lens-out/validation/step5e"
DEFAULT_IMAGES_DIR = "/data/baodq/coco2014/val2014"
DEFAULT_ANNOTATIONS = "/data/baodq/coco2014/annotations/instances_val2014.json"

CENSUS_NOTE = (
    "per (layer, group): top dims by the across-image mean of the per-image mean "
    "|activation| over the group's positions; bos/image/text are disjoint (text "
    "excludes bos), all is the union. std_all is the across-image mean of the "
    "dim's std over ALL positions of the sequence; bos_over_std / image_over_std "
    "/ text_over_std are the group's mean |activation| over std_all - the "
    "massive-activation signature is a huge ratio. clamp_median is the pooled "
    "median replacement value (see median_note). magnitudes are float32 casts of "
    "the block-output residual (pre-final-norm)"
)
CLAMP_POSITION_NOTE = (
    "the prefill forward carries every position, so every group clamps there; "
    "with KV caching each decode forward carries exactly the new token - a text "
    "position - so only the text and all groups also clamp decode steps (bos and "
    "image positions do not recur). decode clamps use the same dims and "
    "replacement values as the arm's prefill clamps"
)
MEDIAN_NOTE = (
    "median mode replaces the dim with the per-position-type median: per image "
    "the median of the dim over the group's positions in that image's clean "
    "census pass, then the median of those per-image medians across all census "
    "images (torch.median, lower median). pooling across images is what makes "
    "median mode meaningful for bos, where a within-image median over one "
    "position would return the value itself"
)
HANDOFF_MEASURE_NOTE = (
    "the lens measure clamps during a single forward - the clamped generation "
    "pass's own prefill forward - rather than a separate recorded forward: the "
    "handoff position (the last prompt position, whose distribution predicts "
    "generated token 0) is a prefill position, so capturing it from the "
    "generation pass is exactly the single-forward measurement at zero extra "
    "cost. baseline rank: the clean census pass (no clamp). clamped rank: the "
    "clamper's captured prefill residual at the readout layer. readout layer: "
    "the highest fitted lens layer (calibrated), else the raw final layer; rank "
    "rule: 1-based, ties count as better ranks (s2_eval._rank_of_row semantics, "
    "x14._rank_of_id); category token: the last subword of ' ' + name "
    "(x14.CATEGORY_TOKEN_RULE)"
)
MATCHER_NOTE = (
    "x13 detect_mentions: longest-first word-boundary match with optional plural "
    "suffix, each match masking its span ('hot dog' does not also count as 'dog'); "
    "irregular plurals ('people', 'mice') are a known miss"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("hf-llava", "tiny"), default="hf-llava",
        help="model backend: 'hf-llava' (default, CUDA) or the tiny CPU smoke fixture",
    )
    parser.add_argument("--lens-dir", default=DEFAULT_LENS_DIR,
                        help="J source lens dir: its fitted layers define the census "
                             "layers and the identity-transport handoff readout; if it "
                             "cannot be loaded the census covers ALL layers and the "
                             "handoff readout falls back to the raw final layer")
    parser.add_argument("--mask", default="text",
                        help="which lens-<mask>.pt and bias-<mask>.pt to use")
    parser.add_argument("--bias-dir", default=DEFAULT_BIAS_DIR,
                        help="directory holding bias-<mask>.pt (the step5e moment "
                             "payload: bias/scale, optional temp/logit_bias) applied "
                             "inside every calibrated lens readout")
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR)
    parser.add_argument(
        "--annotations", default=DEFAULT_ANNOTATIONS,
        help="instances_val2014.json: per-image ground-truth categories + the names",
    )
    parser.add_argument("--n-images", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="image-selection seed")
    parser.add_argument("--question", default="Describe this image in detail.")
    parser.add_argument(
        "--layers", default=",".join(str(layer) for layer in DEFAULT_LAYERS),
        help="comma-separated residual layers for the clamp arms",
    )
    parser.add_argument(
        "--positions", default=",".join(POSITION_GROUPS),
        help="comma-separated subset of: " + ",".join(POSITION_GROUPS),
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help="top-k massive dims per (layer, group), by the Part-1 census")
    parser.add_argument(
        "--clamp-mode", choices=("zero", "median"), default="zero",
        help="zero the top-k dims, or replace them with the per-position-type "
             "pooled median (see median_note)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--json", required=True)
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> LlavaLensModel:
    """The clamping model: the HF checkpoint (default, CUDA) or the tiny CPU fixture."""
    if args.backend == "tiny":
        from vlm_lens.models.tiny_llava import TinyLlavaConfig, build_tiny_llava

        hf_model, processor = build_tiny_llava(TinyLlavaConfig())
        return LlavaLensModel(hf_model, processor)
    return LlavaLensModel.from_pretrained(
        dtype=torch.bfloat16, device="cuda", local_files_only=True
    )


def load_lens_optional(
    args: argparse.Namespace,
) -> tuple[JacobianLens | None, dict[str, Any]]:
    """The identity-transport lens over ``--lens-dir``'s fitted layers, or None.

    A missing/unloadable lens dir is a documented fallback (not an error): the
    census then covers ALL layers and the handoff readout is the raw final layer.
    """
    try:
        lenses, _ = load_lens_set(args.lens_dir)
    except Exception as exc:  # noqa: BLE001 - any load failure takes the fallback
        print(f"note: no usable lens at {args.lens_dir} ({exc}); "
              "census over ALL layers, raw final-layer handoff readout")
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
        bias, scale, temp, logit_bias, meta = load_calibrated_params(args.bias_dir, args.mask)
    except Exception as exc:  # noqa: BLE001 - any load failure skips the corrections
        print(f"note: no usable bias payload at {args.bias_dir} ({exc}); "
              "calibrated readout runs without bias/scale/temp/logit_bias")
        return None, None, None, None, {"loaded": False, "dir": str(args.bias_dir),
                                        "error": str(exc)}
    return bias, scale, temp, logit_bias, {"loaded": True, "dir": str(args.bias_dir)}


def position_group_masks(model: LlavaLensModel, batch: MultimodalBatch) -> dict[str, torch.Tensor]:
    """Boolean ``[seq_len]`` selectors (on the batch's device) for the four groups.

    bos/image/text are disjoint; ``all`` is every position. Position 0 must hold
    the tokenizer's bos id (fail-closed: a shifted layout would misattribute the
    sink), mirroring the repo's placeholder guards.
    """
    bos_token_id = getattr(model.tokenizer, "bos_token_id", None)
    first = int(batch.input_ids[0, 0])
    if bos_token_id is not None and first != int(bos_token_id):
        raise ValueError(
            f"position 0 holds token {first}, not the bos id {int(bos_token_id)}; "
            "the bos group would be misattributed"
        )
    image = batch.image_token_mask[0].detach()
    bos = torch.zeros_like(image)
    bos[0] = True
    return {
        "bos": bos,
        "image": image,
        "text": ~(image | bos),
        "all": torch.ones_like(image),
    }


@torch.no_grad()
def census_pass(
    model: LlavaLensModel, batch: MultimodalBatch, layers: list[int]
) -> dict[int, torch.Tensor]:
    """One recorded no-grad forward: block-output residuals ``[seq, d]`` per layer.

    Tensors are detached, float-cast and moved to CPU so nothing downstream keeps
    the forward alive; the final-layer entry is the PRE-final-norm residual the
    lens is defined on.
    """
    with ActivationRecorder(model.layers, at=layers) as recorder:
        model.forward_mm(batch)
    return {layer: recorder.activations[layer][0].detach().float().cpu() for layer in layers}


class CensusAccumulator:
    """Across-image accumulation of the per-group dim statistics (Part 1).

    Keeps, per (layer, group): the running sum of the per-image mean |activation|
    vector, the position count, and each image's per-dim median vector (the raw
    material of the pooled median clamp values). Bounded memory: O(layers x
    groups x images x d_model) float32.
    """

    def __init__(self, layers: list[int], d_model: int) -> None:
        self.layers = layers
        self.d_model = d_model
        self.mean_abs_sum: dict[int, dict[str, torch.Tensor]] = {
            layer: {group: torch.zeros(d_model, dtype=torch.float64) for group in POSITION_GROUPS}
            for layer in layers
        }
        self.std_all_sum: dict[int, torch.Tensor] = {
            layer: torch.zeros(d_model, dtype=torch.float64) for layer in layers
        }
        self.n_positions: dict[int, dict[str, int]] = {
            layer: {group: 0 for group in POSITION_GROUPS} for layer in layers
        }
        self.image_medians: dict[int, dict[str, list[torch.Tensor]]] = {
            layer: {group: [] for group in POSITION_GROUPS} for layer in layers
        }
        self.n_images = 0

    def add_image(
        self, residuals: dict[int, torch.Tensor], masks: dict[str, torch.Tensor]
    ) -> None:
        """Fold one image's recorded residuals (``[seq, d]`` float32 CPU) in."""
        cpu_masks = {group: mask.detach().cpu() for group, mask in masks.items()}
        for layer, residual in residuals.items():
            self.std_all_sum[layer] += residual.std(dim=0).double()
            for group, mask in cpu_masks.items():
                values = residual[mask]
                if values.shape[0] == 0:
                    continue
                self.mean_abs_sum[layer][group] += values.abs().mean(dim=0).double()
                self.n_positions[layer][group] += int(values.shape[0])
                self.image_medians[layer][group].append(values.median(dim=0).values)
        self.n_images += 1

    def mean_abs(self, layer: int, group: str) -> torch.Tensor:
        """Across-image mean |activation| per dim, ``[d]`` float32."""
        return (self.mean_abs_sum[layer][group] / self.n_images).float()

    def std_all(self, layer: int) -> torch.Tensor:
        """Across-image mean per-dim std over all positions, ``[d]`` float32."""
        return (self.std_all_sum[layer] / self.n_images).float()

    def pooled_median(self, layer: int, group: str) -> torch.Tensor:
        """The median across images of the per-image group medians, ``[d]`` float32."""
        stack = torch.stack(self.image_medians[layer][group])
        return stack.median(dim=0).values.float()

    def top_dims(self, layer: int, group: str, k: int) -> torch.Tensor:
        """Top-k dims by mean |activation|, ties broken by ascending dim index."""
        order = torch.argsort(self.mean_abs(layer, group), descending=True, stable=True)
        return order[:k].clone()


@dataclass(frozen=True)
class LayerClamp:
    """One (layer, group) arm's clamp spec for the prefill/decode hooks."""

    mask: torch.Tensor  # [prompt_len] bool: prefill positions to clamp
    dims: torch.Tensor  # [k] long: the census top-k dims
    values: torch.Tensor  # [k] float: replacement values (zeros for zero mode)
    decode: bool  # also clamp the single-token decode forwards (text/all arms)


class SinkClamper:
    """Forward hooks that clamp census top-k dims at chosen positions.

    x12's BlockPatcher pattern: hooks on ``model.layers[l]`` rewrite the block
    output. The prefill forward (sequence length == the prompt length) clamps
    every position in the arm's mask; each single-token decode forward clamps the
    new token's row only for text/all arms (``LayerClamp.decode``). Layers listed
    in ``capture_layers`` (typically the handoff readout layer) additionally
    record their PREFILL residual at the last prompt position — with the arm's
    clamp upstream or at that layer, this is the clamped handoff residual
    (x14's ``step_records[0]`` convention). The prefill fire count is asserted
    == 1 per clamped layer, as in x12.
    """

    def __init__(
        self,
        model: LlavaLensModel,
        prompt_len: int,
        clamps: dict[int, LayerClamp],
        capture_layers: tuple[int, ...] = (),
    ) -> None:
        self._model = model
        self._prompt_len = int(prompt_len)
        self._clamps = clamps
        self._capture_layers = tuple(capture_layers)
        self._handles: list[Any] = []
        self.n_prefill_fires: dict[int, int] = {}
        self.n_decode_clamps = 0
        self.prefill_residuals: dict[int, torch.Tensor] = {}

    def _make_hook(self, layer: int, clamp: LayerClamp | None) -> Any:
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            tensor = output if torch.is_tensor(output) else output[0]
            is_prefill = tensor.shape[1] == self._prompt_len
            if is_prefill:
                self.n_prefill_fires[layer] = self.n_prefill_fires.get(layer, 0) + 1
            modified = tensor
            if clamp is not None and (is_prefill or clamp.decode):
                modified = tensor.clone()
                values = clamp.values.to(dtype=modified.dtype, device=modified.device)
                if is_prefill:
                    mask = clamp.mask.to(modified.device)
                    rows = modified[0, mask]
                    rows[:, clamp.dims.to(modified.device)] = values
                    modified[0, mask] = rows
                else:
                    modified[0, -1, clamp.dims.to(modified.device)] = values
                    self.n_decode_clamps += 1
            if is_prefill and layer in self._capture_layers:
                self.prefill_residuals[layer] = modified[0, -1].detach().float().cpu()
            if modified is tensor:
                return output
            if torch.is_tensor(output):
                return modified
            return (modified, *output[1:])

        return hook

    def __enter__(self) -> SinkClamper:
        for layer in sorted(set(self._clamps) | set(self._capture_layers)):
            self._handles.append(
                self._model.layers[layer].register_forward_hook(
                    self._make_hook(layer, self._clamps.get(layer))
                )
            )
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        return None

    def assert_fired(self) -> None:
        """Every clamped layer must have fired on exactly one prefill forward."""
        for layer in self._clamps:
            fires = self.n_prefill_fires.get(layer, 0)
            if fires != 1:
                raise RuntimeError(
                    f"expected exactly one prefill fire at layer {layer}, got {fires}"
                )
        for layer in self._capture_layers:
            if layer not in self.prefill_residuals:
                raise RuntimeError(
                    f"capture layer {layer} recorded no prefill residual"
                )


def generate_clamped(
    model: LlavaLensModel,
    batch: MultimodalBatch,
    clamps: dict[int, LayerClamp],
    capture_layers: tuple[int, ...],
    *,
    max_new_tokens: int,
) -> tuple[dict[str, Any], SinkClamper]:
    """Greedy caption under the arm's clamp (x12's ``generate_patched`` pattern)."""
    clamper = SinkClamper(model, batch.seq_len, clamps, capture_layers)
    with clamper:
        generated = model.hf_model.generate(
            **batch.hf_kwargs(),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    clamper.assert_fired()
    new_tokens = generated[0, batch.seq_len :].detach().cpu()
    return {
        "text": model._decode(new_tokens),
        "token_ids": [int(token) for token in new_tokens],
    }, clamper


def _mean(values: list[float | bool | int | None]) -> float | None:
    """Mean over the non-None entries, or None when there are none."""
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def build_arm_table(
    rows: list[dict[str, Any]], layers: list[int], groups: list[str]
) -> dict[str, dict[str, dict[str, Any]]]:
    """Per (layer x group) aggregates over the per-image arm rows."""
    table: dict[str, dict[str, dict[str, Any]]] = {}
    for layer in layers:
        per_group: dict[str, dict[str, Any]] = {}
        for group in groups:
            arm_rows = [
                row for row in rows if row["layer"] == layer and row["group"] == group
            ]
            if not arm_rows:
                continue
            with_halluc = [row for row in arm_rows if row["baseline_hallucinated"]]
            with_grounded = [row for row in arm_rows if row["baseline_grounded"]]
            per_group[group] = {
                "n_arms": len(arm_rows),
                "changed_rate": _mean([row["changed"] for row in arm_rows]),
                "n_with_baseline_halluc": len(with_halluc),
                "hallucination_removed_rate": _mean(
                    [row["hallucination_removed_rate"] for row in with_halluc]
                ),
                "n_with_baseline_grounded": len(with_grounded),
                "grounded_retention": _mean(
                    [row["grounded_retention"] for row in with_grounded]
                ),
                "token_length_ratio": _mean(
                    [row["token_length_ratio"] for row in arm_rows]
                ),
                "mean_handoff_rank_baseline": _mean(
                    [row["handoff_rank_baseline"] for row in arm_rows]
                ),
                "mean_handoff_rank_clamped": _mean(
                    [row["handoff_rank_clamped"] for row in arm_rows]
                ),
                "mean_handoff_rank_delta": _mean(
                    [row["handoff_rank_delta"] for row in arm_rows]
                ),
            }
        table[str(layer)] = per_group
    return table


def format_cell(value: float | None, width: int = 9, precision: int = 3) -> str:
    """Right-aligned fixed-precision cell, or a dash when empty."""
    if value is None:
        return f"{'-':>{width}}"
    return f"{value:>{width}.{precision}f}"


def print_arm_table(
    table: dict[str, dict[str, dict[str, Any]]], n_images: int
) -> str:
    """The human digest; also returned verbatim for the JSON ``digest`` field."""
    lines = [
        f"== X16 clamp digest: arms = layer x group, means over {n_images} images ==",
        f"{'layer':>5}  {'group':<6}  {'n':>3}  {'changed':>7}  {'hall_rem':>8}  "
        f"{'gnd_keep':>8}  {'len_rat':>7}  {'rank_base':>9}  {'rank_clmp':>9}  "
        f"{'rank_delta':>10}",
    ]
    for layer, per_group in table.items():
        for group, cell in per_group.items():
            delta = cell["mean_handoff_rank_delta"]
            delta_text = "-" if delta is None else f"{delta:>+10.1f}"
            base = cell["mean_handoff_rank_baseline"]
            clamped = cell["mean_handoff_rank_clamped"]
            lines.append(
                f"{layer:>5}  {group:<6}  {cell['n_arms']:>3}  "
                f"{format_cell(cell['changed_rate'], 7)}  "
                f"{format_cell(cell['hallucination_removed_rate'], 8)}  "
                f"{format_cell(cell['grounded_retention'], 8)}  "
                f"{format_cell(cell['token_length_ratio'], 7)}  "
                f"{format_cell(base, 9, 1)}  "
                f"{format_cell(clamped, 9, 1)}  "
                f"{delta_text:>10}"
            )
    return "\n".join(lines)


def print_census(
    census: dict[str, Any], layers: list[int], groups: list[str], max_show: int = 8
) -> str:
    """Compact one-line-per-(layer, group) view of the massive-dims table."""
    lines = [
        f"== X16 census: top-{census['k']} dims by mean |activation| over "
        f"{census['n_images']} images (full table in JSON.census) ==",
        f"{'layer':>5}  {'group':<6}  {'n_pos/img':>9}  {'top dims (first '
        f"{max_show})":<40}  {'max|a|':>10}  {'img/std@1':>10}",
    ]
    for layer in layers:
        for group in groups:
            cell = census["layers"][str(layer)]["groups"].get(group)
            if cell is None:
                continue
            dims = [entry["dim"] for entry in cell["top_dims"]]
            shown = ",".join(str(dim) for dim in dims[:max_show])
            if len(dims) > max_show:
                shown += f",+{len(dims) - max_show}"
            top1 = cell["top_dims"][0]
            lines.append(
                f"{layer:>5}  {group:<6}  {cell['n_positions_mean']:>9.1f}  "
                f"{shown:<40}  {top1['mean_abs']:>10.3f}  "
                f"{top1['image_over_std']:>10.3f}"
            )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    layers = [int(part) for part in args.layers.split(",") if part.strip()]
    groups = [part for part in args.positions.split(",") if part.strip()]
    unknown = [group for group in groups if group not in POSITION_GROUPS]
    if unknown:
        raise ValueError(f"unknown position groups {unknown}; choose from {list(POSITION_GROUPS)}")
    if args.k < 1:
        raise ValueError(f"--k must be >= 1, got {args.k}")
    if not layers:
        raise ValueError("--layers must name at least one layer")

    model = load_model(args)
    invalid = [layer for layer in layers if not 0 <= layer < model.n_layers]
    if invalid:
        raise ValueError(
            f"layers {invalid} outside 0..{model.n_layers - 1} (backend {args.backend!r}, "
            f"n_layers={model.n_layers}); pass --layers within range"
        )

    final_layer = model.n_layers - 1
    source_lens, lens_meta = load_lens_optional(args)
    bias, scale, temp, logit_bias, bias_meta = load_bias_optional(args)
    device = model.unembed_weight().device
    if source_lens is not None:
        lens = identity_lens(source_lens, device)
        census_layers = sorted(set(lens.source_layers) | {final_layer})
        readout_layer = max(lens.source_layers)
    else:
        # No lens dir: identity transport over ALL layers so x14's calibrated
        # readout helper keeps one code path; the handoff readout is then the
        # raw final layer unless a bias payload loads.
        lens = JacobianLens(
            jacobians={
                layer: torch.eye(model.d_model, dtype=torch.float32, device=device)
                for layer in range(model.n_layers)
            },
            n_prompts=0,
            d_model=model.d_model,
        )
        lens_meta = {**lens_meta, "identity_over_all_layers": True}
        census_layers = list(range(model.n_layers))
        readout_layer = final_layer
    bias = {layer: tensor.to(device) for layer, tensor in bias.items()} if bias else None
    logit_bias = (
        {layer: tensor.to(device) for layer, tensor in logit_bias.items()}
        if logit_bias
        else None
    )
    calibrated = bias is not None and readout_layer in bias
    # The census also covers the arm layers even when the lens did not fit
    # them: the |activation| ranking needs only the raw residual, so any
    # --layers value is clampable (calibration only affects the handoff
    # readout, which stays on the fitted/readout layer).
    census_layers = sorted(set(census_layers) | set(layers))
    if readout_layer not in census_layers:  # defensive; both branches guarantee it
        census_layers.append(readout_layer)
        census_layers.sort()

    k = min(args.k, model.d_model)
    if k < args.k:
        print(f"note: --k {args.k} capped to d_model={model.d_model}")

    names, gt_by_image = load_ground_truth(args.annotations)
    images = select_images(list_coco_images(args.images_dir), args.n_images, seed=args.seed)
    prompt = prompt_text(args.question)
    arms = [(layer, group) for layer in layers for group in groups]
    print(
        f"== X16 sink clamp: {len(images)} images, {len(arms)} arms "
        f"(layers={layers} x groups={groups}), k={k}, mode={args.clamp_mode} =="
    )
    print(f"model ready: layers={model.n_layers} d_model={model.d_model} backend={args.backend}")
    print(f"prompt: {prompt!r}")
    print(
        f"lens: {lens_meta.get('loaded', False)}; census layers={census_layers}; "
        f"handoff readout layer={readout_layer} calibrated={calibrated}"
    )

    # ------------------------------------------------------------------ Part 1
    print(f"\n-- Part 1: census over {len(images)} images "
          f"(one recorded forward per image) --")
    census_acc = CensusAccumulator(census_layers, model.d_model)
    baseline_handoff_rank: dict[str, int | None] = {}
    image_gt: dict[str, set[int]] = {}
    n_skipped = 0
    for index, path in enumerate(images):
        image_id = image_id_of(path)
        gt_ids = gt_by_image.get(image_id)
        if gt_ids is None:
            print(f"  [skip] {path.name} (id {image_id}) is in no annotation record")
            n_skipped += 1
            continue
        image_gt[path.name] = gt_ids
        batch = model.encode_mm(prompt, path, max_length=args.max_seq_len)
        masks = position_group_masks(model, batch)
        residuals = census_pass(model, batch, census_layers)
        census_acc.add_image(residuals, masks)

        # Baseline handoff rank from this clean pass (x14's step_records[0]
        # convention: the prefill forward's last-position residual).
        category_token_ids = _category_token_ids(
            model.tokenizer, [names[cid] for cid in gt_ids]
        )
        rank: int | None = None
        if category_token_ids:
            logits = _calibrated_lens_logits(
                # Census residuals live on CPU (bounded memory); the readout
                # hidden goes back to the lens device for transport+unembed.
                model, lens, residuals[readout_layer][-1].to(device), readout_layer,
                bias, scale, temp, logit_bias,
            )
            rank = min(_rank_of_id(logits, cid) for cid in category_token_ids)
        baseline_handoff_rank[path.name] = rank
        print(
            f"  [{index + 1 - n_skipped}/{len(images) - n_skipped}] {path.name}: "
            f"seq={batch.seq_len} img_tokens={batch.n_image_tokens} "
            f"gt={len(gt_ids)} handoff_rank={rank}"
        )

    if census_acc.n_images == 0:
        raise ValueError(
            "no images with an annotation record were processed; the census and "
            "clamp arms are undefined -- check --annotations against --images-dir"
        )

    # The massive-dims table + the clamp plan share one source: the census.
    census_layers_out: dict[str, Any] = {}
    plan_dims: dict[tuple[int, str], torch.Tensor] = {}
    plan_values: dict[tuple[int, str], torch.Tensor] = {}
    for layer in census_layers:
        groups_out: dict[str, Any] = {}
        for group in POSITION_GROUPS:
            if not census_acc.image_medians[layer][group]:
                continue  # group had no positions in any image
            mean_abs = {g: census_acc.mean_abs(layer, g) for g in POSITION_GROUPS}
            std_all = census_acc.std_all(layer)
            dims = census_acc.top_dims(layer, group, k)
            pooled = census_acc.pooled_median(layer, group)
            entries = []
            for rank_index, dim in enumerate(dims.tolist()):
                dim = int(dim)
                std = float(std_all[dim])
                entries.append({
                    "rank": rank_index,
                    "dim": dim,
                    "mean_abs": float(mean_abs[group][dim]),
                    "mean_abs_bos": float(mean_abs["bos"][dim]),
                    "mean_abs_image": float(mean_abs["image"][dim]),
                    "mean_abs_text": float(mean_abs["text"][dim]),
                    "mean_abs_all": float(mean_abs["all"][dim]),
                    "std_all": std,
                    "bos_over_std": float(mean_abs["bos"][dim]) / max(std, 1e-12),
                    "image_over_std": float(mean_abs["image"][dim]) / max(std, 1e-12),
                    "text_over_std": float(mean_abs["text"][dim]) / max(std, 1e-12),
                    "clamp_median": float(pooled[dim]),
                })
            groups_out[group] = {
                "n_positions_mean": census_acc.n_positions[layer][group]
                / census_acc.n_images,
                "top_dims": entries,
            }
            plan_dims[(layer, group)] = dims
            plan_values[(layer, group)] = (
                pooled[dims].clone() if args.clamp_mode == "median" else torch.zeros(len(dims))
            )
        census_layers_out[str(layer)] = {"groups": groups_out}
    census_out = {
        "n_images": census_acc.n_images,
        "k": k,
        "layers": census_layers_out,
        "note": CENSUS_NOTE,
    }
    print(print_census(census_out, census_layers, list(POSITION_GROUPS)))

    # ------------------------------------------------------------------ Part 2
    print(f"\n-- Part 2: {len(arms)} arms x {len(images) - n_skipped} images "
          f"(baseline + one clamped generation per arm) --")
    results: list[dict[str, Any]] = []
    for path in images:
        if path.name not in image_gt:
            continue
        gt_ids = image_gt[path.name]
        gt_names = {names[cid] for cid in gt_ids}
        batch = model.encode_mm(prompt, path, max_length=args.max_seq_len)
        masks = position_group_masks(model, batch)

        baseline = generate_with_edits(
            model, None, batch, max_new_tokens=args.max_new_tokens
        )
        baseline_ids = baseline["token_ids"]
        baseline_text = baseline["text"]
        baseline_mentioned = detect_mentions(names, baseline_text)
        baseline_halluc = [name for name in baseline_mentioned if name not in gt_names]
        baseline_grounded = [name for name in baseline_mentioned if name in gt_names]
        baseline_rank = baseline_handoff_rank[path.name]

        print(
            f"  {path.name}: baseline={baseline_text[:60]!r} "
            f"halluc={baseline_halluc or '-'} grounded={baseline_grounded or '-'}"
        )

        for layer, group in arms:
            mask = masks[group]
            if not bool(mask.any()):
                print(f"    [skip] L{layer}/{group}: no positions of this type")
                continue
            dims = plan_dims[(layer, group)]
            clamp = LayerClamp(
                mask=mask,
                dims=dims,
                values=plan_values[(layer, group)],
                decode=group in DECODE_GROUPS,
            )
            out, clamper = generate_clamped(
                model, batch, {layer: clamp}, (readout_layer,),
                max_new_tokens=args.max_new_tokens,
            )
            text = out["text"]
            token_ids = out["token_ids"]
            mentioned = detect_mentions(names, text)
            halluc = [name for name in mentioned if name not in gt_names]
            grounded = [name for name in mentioned if name in gt_names]
            removed_rate = (
                sum(1 for name in baseline_halluc if name not in set(halluc))
                / len(baseline_halluc)
                if baseline_halluc
                else None
            )
            retention = (
                sum(1 for name in baseline_grounded if name in set(grounded))
                / len(baseline_grounded)
                if baseline_grounded
                else None
            )
            clamped_rank: int | None = None
            if baseline_rank is not None:
                category_token_ids = _category_token_ids(
                    model.tokenizer, [names[cid] for cid in gt_ids]
                )
                logits = _calibrated_lens_logits(
                    model, lens, clamper.prefill_residuals[readout_layer].to(device),
                    readout_layer, bias, scale, temp, logit_bias,
                )
                clamped_rank = min(_rank_of_id(logits, cid) for cid in category_token_ids)
            results.append({
                "image": path.name,
                "image_id": image_id_of(path),
                "layer": layer,
                "group": group,
                "clamp_mode": args.clamp_mode,
                "k": k,
                "dims": [int(dim) for dim in dims.tolist()],
                "n_clamp_positions_prefill": int(mask.sum()),
                "n_decode_clamps": clamper.n_decode_clamps,
                "text": text,
                "n_tokens": len(token_ids),
                "baseline_text": baseline_text,
                "baseline_n_tokens": len(baseline_ids),
                "changed": token_ids != baseline_ids,
                "token_length_ratio": len(token_ids) / max(len(baseline_ids), 1),
                "mentioned": mentioned,
                "hallucinated": halluc,
                "grounded": grounded,
                "baseline_mentioned": baseline_mentioned,
                "baseline_hallucinated": baseline_halluc,
                "baseline_grounded": baseline_grounded,
                "hallucination_removed_rate": removed_rate,
                "grounded_retention": retention,
                "handoff_readout_layer": readout_layer,
                "handoff_rank_baseline": baseline_rank,
                "handoff_rank_clamped": clamped_rank,
                "handoff_rank_delta": (
                    None
                    if baseline_rank is None or clamped_rank is None
                    else clamped_rank - baseline_rank
                ),
            })

    n_images_processed = len({row["image"] for row in results})
    arm_table = build_arm_table(results, layers, groups)
    digest_text = print_arm_table(arm_table, n_images_processed)
    print("\n" + digest_text)

    # Per-image baseline accounting: every arm row repeats its image's baseline
    # mention lists, so the per-image view is taken once per image.
    image_names = sorted({row["image"] for row in results})
    first_row_of: dict[str, dict[str, Any]] = {}
    for row in results:
        first_row_of.setdefault(row["image"], row)
    summary = {
        "n_images_processed": n_images_processed,
        "n_arms": len(arms),
        "n_results": len(results),
        "n_images_with_baseline_hallucination": sum(
            1 for name in image_names if first_row_of[name]["baseline_hallucinated"]
        ),
        "n_baseline_hallucinated_mentions": sum(
            len(first_row_of[name]["baseline_hallucinated"]) for name in image_names
        ),
        "n_baseline_grounded_mentions": sum(
            len(first_row_of[name]["baseline_grounded"]) for name in image_names
        ),
        "mean_changed_rate": _mean(
            [
                cell["changed_rate"]
                for per_group in arm_table.values()
                for cell in per_group.values()
            ]
        ),
        "lens": lens_meta,
        "bias": bias_meta,
        "calibrated": calibrated,
    }

    interpretation_limits = [
        "single-arm causality: each arm clamps ONE (layer, group) with no other "
        "clamp armed; interactions between simultaneous clamps are untested",
        "census top-k uses the ACROSS-image mean |activation|; individual images "
        "can rank dims slightly differently, so a per-image clamp plan could "
        "shift a few dims",
        "median mode pools the per-image group medians across images (see "
        "median_note); a within-image median would be a structural no-op for the "
        "single-position bos group",
        "only text/all arms clamp decode steps: with KV caching the decode "
        "forward carries exactly the new (text) token, and bos/image positions "
        "do not recur; bos/image arms act on the prefill residual only, whose "
        "effect reaches the output through the KV cache",
        "handoff readout at the highest fitted lens layer: arms whose layer "
        "EQUALS the readout layer and whose group excludes the last-prompt "
        "position (bos, image) have a structurally zero lens delta at that "
        "layer - for those arms only the caption readout (and lens deltas of "
        "earlier-layer arms) is informative",
        "handoff_rank_delta > 0 means the image's true objects became LESS "
        "decodable at the handoff position under the clamp; the sign alone does "
        "not say whether caption quality improved - read it jointly with "
        "hallucination_removed_rate and grounded_retention",
        "detect_mentions matches whole words with an optional plural suffix; "
        "irregular plurals ('people', 'mice') are a known miss (x13)",
        "changed uses exact token-id equality: captions that differ only in "
        "tokenization count as changed (x12 convention)",
        "the tiny backend is a random-weight fixture: it exercises the census/"
        "clamp/scoring machinery, its magnitudes and rates do not transfer to "
        "the real model",
        "baseline vs clamped handoff residuals come from different forward "
        "shapes (use_cache=False census pass vs the generation prefill); causal "
        "greedy decoding makes them the same computation up to GEMM "
        "reassociation in low-order bits (x13's replay argument)",
    ]

    report: dict[str, Any] = {
        "script": Path(__file__).name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": args.backend,
        "hypothesis": HYPOTHESIS,
        "images_dir": str(args.images_dir),
        "annotations": str(args.annotations),
        "n_images_requested": args.n_images,
        "n_images_selected": len(images),
        "n_images_skipped_no_annotations": n_skipped,
        "seed": args.seed,
        "question": args.question,
        "prompt": prompt,
        "prompt_template": PROMPT_TEMPLATE,
        "prompt_template_hash": prompt_template_hash(),
        "layers": layers,
        "positions": groups,
        "k": k,
        "clamp_mode": args.clamp_mode,
        "max_new_tokens": args.max_new_tokens,
        "max_seq_len": args.max_seq_len,
        "census_layers": census_layers,
        "handoff_readout_layer": readout_layer,
        "lens": lens_meta,
        "bias": bias_meta,
        "calibrated": calibrated,
        "census_note": CENSUS_NOTE,
        "clamp_position_note": CLAMP_POSITION_NOTE,
        "median_note": MEDIAN_NOTE,
        "handoff_measure_note": HANDOFF_MEASURE_NOTE,
        "matcher_note": MATCHER_NOTE,
        "census": census_out,
        "per_layer_position": arm_table,
        "digest": digest_text,
        "summary": summary,
        "results": results,
        "interpretation_limits": interpretation_limits,
    }
    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.json} ({len(results)} arm rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
