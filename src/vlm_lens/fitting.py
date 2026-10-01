# SPDX-License-Identifier: Apache-2.0
"""Masked multimodal Jacobian fitting.

Extension of the vendored reference estimator (``jlens/fitting.py``, Apache-2.0). The
estimator itself is unchanged — one forward on the prompt replicated ``dim_batch`` times,
then ``ceil(d_model / dim_batch)`` backward passes, each planting a one-hot cotangent at
every valid *target* position, so that the gradient at source position ``p`` is
``sum_{p' >= p} dh_final[p'] / dh_l[p]``. What changes for VLMs:

* inputs are multimodal (``FitSample`` -> images + text through ``LlavaLensModel``);
* the *source*-position average is computed for several modality masks
  (``text`` / ``image`` / ``all``) **in the same backward passes** — the masks are
  different reductions of one gradient tensor, so the extra lenses are free;
* samples that contribute to no mask (too long, no valid positions) are skipped, and each
  mask accumulates its own ``n_prompts`` so merging across shards stays weighted correctly.

Per-sample cost is the reference's: ``d_model`` backward passes regardless of sequence
length; sequence length and ``dim_batch`` drive memory, not FLOPs.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from jlens.fitting import _check_layer_indices
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens

from vlm_lens._batch import as_batch
from vlm_lens.data.manifest import FitSample
from vlm_lens.models.llava import LlavaLensModel, MultimodalBatch
from vlm_lens.positions import DEFAULT_MASKS, build_position_masks

logger = logging.getLogger(__name__)

#: Text-only control fits mirror the paper's protocol.
TEXT_SKIP_FIRST = 16
#: Multimodal fits: only BOS is a pure sink; every image position carries content.
MM_SKIP_FIRST = 1

CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class FitInfo:
    """Per-sample diagnostics."""

    sample_id: str
    seq_len: int
    n_image_tokens: int
    mask_positions: dict[str, int]
    seconds: float
    target_mask: str = "all"
    target_positions: int = 0


@dataclass
class FitResult:
    """Masked lenses plus the per-sample history the fit script reports on."""

    lenses: dict[str, JacobianLens]
    history: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)

    @property
    def n_prompts(self) -> dict[str, int]:
        return {mask: lens.n_prompts for mask, lens in self.lenses.items()}


def model_fingerprint(model: Any) -> dict[str, Any]:
    """Identity of a model, for checkpoint/artifact validation (plain types only)."""
    hf_model = getattr(model, "hf_model", None)
    config = getattr(hf_model, "config", None)
    fingerprint = {
        "class": type(hf_model if hf_model is not None else model).__name__,
        "d_model": int(model.d_model),
        "n_layers": int(model.n_layers),
        "image_token_id": int(getattr(model, "image_token_id", -1)),
        "model_type": getattr(config, "model_type", None),
        "name_or_path": getattr(config, "_name_or_path", None)
        or getattr(config, "name_or_path", None),
    }
    vision = getattr(model, "vision_fingerprint", None)
    if callable(vision):
        fingerprint.update(vision())
    return fingerprint


def _atomic_save(obj: object, path: str | os.PathLike[str]) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def jacobian_for_sample(
    model: LlavaLensModel,
    sample: FitSample | MultimodalBatch | str,
    source_layers: Sequence[int],
    *,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 1536,
    skip_first: int = MM_SKIP_FIRST,
    masks: Sequence[str] = DEFAULT_MASKS,
    target_mask: str = "all",
) -> tuple[dict[str, dict[int, torch.Tensor]], FitInfo]:
    """Per-mask Jacobian estimators ``J_l`` for one multimodal sample.

    Returns ``({mask: {layer: [d_model, d_model] fp32 CPU tensor}}, info)``. Masks with no
    valid positions are omitted (e.g. ``image`` for a text-only sample).

    ``target_mask`` selects where the cotangent is seeded (X1): ``"all"`` matches the
    upstream estimator, ``"text"`` drops the 97 %-dominant patch-continuation targets so
    the rows measure influence on the verbalization positions instead. Source masks and
    target mask are independent; rows for sources that cannot causally reach a target
    (e.g. caption tokens after the image block) are bit-identical between the two.
    """
    n_layers, d_model = model.n_layers, model.d_model
    source_layers, target_layer = _check_layer_indices(source_layers, target_layer, n_layers)
    start = time.perf_counter()
    batch = as_batch(model, sample, max_seq_len)
    position_masks = build_position_masks(
        batch.input_ids, model.image_token_id, skip_first=skip_first, masks=masks
    )
    # Target positions are selected independently of the source reductions: the
    # cotangent is seeded at every position in ``target_mask``.
    target_positions = build_position_masks(
        batch.input_ids, model.image_token_id, skip_first=skip_first, masks=(target_mask,)
    )[target_mask]
    if not bool(target_positions.any()):
        raise ValueError(
            f"no valid target positions for target_mask={target_mask!r} "
            f"(seq_len={batch.seq_len}, skip_first={skip_first})"
        )

    active = {name: mask for name, mask in position_masks.items() if bool(mask.any())}
    if not active:
        raise ValueError(
            f"no valid source positions for masks {list(masks)} "
            f"(seq_len={batch.seq_len}, skip_first={skip_first})"
        )
    mask_positions = {name: int(mask.sum()) for name, mask in active.items()}

    jacobians: dict[str, dict[int, torch.Tensor]] = {
        name: {layer: torch.zeros(d_model, d_model, dtype=torch.float32) for layer in source_layers}
        for name in active
    }
    n_passes = math.ceil(d_model / dim_batch)

    with (
        ActivationRecorder(
            model.layers,
            at=[*source_layers, target_layer],
            start_graph_at=min(source_layers),
        ) as recorder,
        torch.enable_grad(),
    ):
        model.forward_mm(batch.expand(dim_batch))
        target_activation = recorder.activations[target_layer]  # [dim_batch, seq_len, d_model]
        source_activations = [recorder.activations[layer] for layer in source_layers]

        device = target_activation.device
        targets = target_positions.nonzero(as_tuple=True)[0].to(device)
        positions_by_mask = {
            name: mask.nonzero(as_tuple=True)[0].to(device) for name, mask in active.items()
        }
        batch_indices = torch.arange(dim_batch, device=device)
        cotangent = torch.zeros_like(target_activation)

        for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
            n_dims = min(dim_batch, d_model - dim_start)
            # One-hot cotangent at dim (dim_start + b) for batch element b, at every valid
            # target position: rows dim_start..dim_start+n of J_l.
            cotangent.zero_()
            cotangent[
                batch_indices[:n_dims][:, None],
                targets[None, :],
                (dim_start + batch_indices[:n_dims])[:, None],
            ] = 1.0
            grads = torch.autograd.grad(
                outputs=target_activation,
                inputs=source_activations,
                grad_outputs=cotangent,
                retain_graph=(pass_idx < n_passes - 1),
            )
            for layer, grad in zip(source_layers, grads, strict=True):
                for name, positions in positions_by_mask.items():
                    if positions.numel() == 0:
                        continue
                    rows = grad[:n_dims, positions, :].float().mean(dim=1)
                    jacobians[name][layer][dim_start : dim_start + n_dims, :] = rows.cpu()
            del grads

    seconds = time.perf_counter() - start
    return jacobians, FitInfo(
        sample_id=getattr(sample, "sample_id", "<batch>"),
        seq_len=batch.seq_len,
        n_image_tokens=batch.n_image_tokens,
        mask_positions=mask_positions,
        seconds=seconds,
        target_mask=target_mask,
        target_positions=int(target_positions.sum()),
    )


def fit_masked(
    model: LlavaLensModel,
    samples: Sequence[FitSample | MultimodalBatch | str],
    *,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 1536,
    skip_first: int = MM_SKIP_FIRST,
    masks: Sequence[str] = DEFAULT_MASKS,
    target_mask: str = "all",
    checkpoint_path: str | os.PathLike[str] | None = None,
    checkpoint_every: int | None = 1,
    resume: bool = True,
    limit: int | None = None,
    shard: tuple[int, int] = (0, 1),
    log_every: int = 1,
) -> FitResult:
    """Fit one lens per modality mask over a sample list, with resumable checkpoints.

    Args:
        samples: ``FitSample`` / ``MultimodalBatch`` / plain text (text-only).
        source_layers: Layers to fit at; defaults to every layer below ``target_layer``.
        target_layer: Gradient target layer; defaults to the final layer.
        dim_batch: Output dimensions per backward pass (memory knob; FLOPs invariant).
        max_seq_len: Reject longer samples instead of truncating them.
        skip_first: Leading source/target positions excluded (attention sinks).
        masks: Source reductions to fit (``text`` / ``image`` / ``all``).
        target_mask: Where the estimator's cotangent is seeded (see
            :func:`jacobian_for_sample`); part of the checkpoint fingerprint.
        checkpoint_path: Where to write resumable state (sums + counts, not means).
        checkpoint_every: Checkpoint cadence in samples; ``None`` = only at the end.
        resume: Resume from ``checkpoint_path`` when it exists and matches the config.
        limit: Use only the first ``limit`` samples (before sharding).
        shard: ``(index, count)`` — fit only ``samples[index::count]``; merge shards later
            with :meth:`jlens.lens.JacobianLens.merge`.
        log_every: Log cadence in samples.

    Returns:
        :class:`FitResult` with one :class:`JacobianLens` per mask that received at least
        one sample (``n_prompts`` is per mask).
    """
    n_layers, d_model = model.n_layers, model.d_model
    source_layers, target_layer = _check_layer_indices(source_layers, target_layer, n_layers)
    shard_index, shard_count = shard
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"invalid shard {shard!r}")

    sample_list = list(samples)
    if limit is not None:
        sample_list = sample_list[:limit]
    sample_list = sample_list[shard_index::shard_count]

    config: dict[str, Any] = {
        "source_layers": list(source_layers),
        "target_layer": int(target_layer),
        "dim_batch": int(dim_batch),
        "max_seq_len": int(max_seq_len),
        "skip_first": int(skip_first),
        "masks": list(masks),
        "target_mask": str(target_mask),
        "shard": [int(shard_index), int(shard_count)],
        "n_samples": len(sample_list),
    }
    # ``n_samples`` is a run parameter, not part of the estimand: a partial run must be
    # resumable by a longer one without tripping the fingerprint check.
    fingerprint = {
        **model_fingerprint(model),
        **{key: value for key, value in config.items() if key != "n_samples"},
    }

    jacobian_sum: dict[str, dict[int, torch.Tensor]] = {
        mask: {layer: torch.zeros(d_model, d_model, dtype=torch.float32) for layer in source_layers}
        for mask in masks
    }
    n_done: dict[str, int] = {mask: 0 for mask in masks}
    next_idx = 0

    if resume and checkpoint_path is not None and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if int(state.get("version", 0)) != CHECKPOINT_VERSION:
            raise ValueError(f"checkpoint {checkpoint_path} has an unsupported version")
        saved = state.get("fingerprint", {})
        mismatch = {k: (saved.get(k), v) for k, v in fingerprint.items() if saved.get(k) != v}
        if mismatch:
            raise ValueError(
                f"checkpoint at {checkpoint_path} was fitted with different settings: "
                f"{mismatch}; pass --no-resume to discard it"
            )
        jacobian_sum = {mask: dict(layers) for mask, layers in state["jacobian_sum"].items()}
        n_done = {mask: int(n) for mask, n in state["n_done"].items()}
        next_idx = int(state["next_idx"])
        logger.info("resuming from %s: %d/%d samples processed", checkpoint_path, next_idx, len(sample_list))

    def write_checkpoint() -> None:
        if checkpoint_path is None:
            return
        _atomic_save(
            {
                "version": CHECKPOINT_VERSION,
                "jacobian_sum": jacobian_sum,
                "n_done": n_done,
                "next_idx": next_idx,
                "fingerprint": fingerprint,
            },
            checkpoint_path,
        )

    sqrt_d = math.sqrt(d_model)
    history: list[dict[str, Any]] = []
    skipped: list[str] = []

    logger.info(
        "fit: model=%s, %d layers, fitting %d source layers (target=L%d) on %d samples, masks=%s",
        fingerprint["class"],
        n_layers,
        len(source_layers),
        target_layer,
        len(sample_list),
        list(masks),
    )

    for sample_idx, sample in enumerate(sample_list):
        if sample_idx < next_idx:
            continue
        sample_id = getattr(sample, "sample_id", f"<{sample_idx}>")
        try:
            per_mask_J, info = jacobian_for_sample(
                model,
                sample,
                source_layers,
                target_layer=target_layer,
                dim_batch=dim_batch,
                max_seq_len=max_seq_len,
                skip_first=skip_first,
                masks=masks,
                target_mask=target_mask,
            )
        except ValueError as exc:
            logger.warning("  skipping sample %d (%s): %s", sample_idx, sample_id, exc)
            skipped.append(f"{sample_id}: {exc}")
            next_idx = sample_idx + 1
            if checkpoint_every is not None and next_idx % checkpoint_every == 0:
                write_checkpoint()
            continue

        record: dict[str, Any] = {
            "sample_id": sample_id,
            "seq_len": info.seq_len,
            "n_image_tokens": info.n_image_tokens,
            "mask_positions": info.mask_positions,
            "seconds": round(info.seconds, 1),
            "prompt_norm": {},
            "mean_rel_change": {},
        }
        for mask, per_layer in per_mask_J.items():
            prompt_norm = max(per_layer[layer].norm().item() for layer in source_layers) / sqrt_d
            if n_done[mask] > 0:
                running = {layer: jacobian_sum[mask][layer] / n_done[mask] for layer in source_layers}
                mean_rel_change = max(
                    (
                        (per_layer[layer] - running[layer]).norm()
                        / ((n_done[mask] + 1) * running[layer].norm())
                    ).item()
                    for layer in source_layers
                )
            else:
                mean_rel_change = float("nan")
            record["prompt_norm"][mask] = round(prompt_norm, 3)
            record["mean_rel_change"][mask] = float(mean_rel_change)
            for layer in source_layers:
                jacobian_sum[mask][layer] += per_layer[layer]
            n_done[mask] += 1

        next_idx = sample_idx + 1
        history.append(record)

        if log_every and (next_idx % log_every == 0 or next_idx == len(sample_list)):
            change = ", ".join(
                f"{mask}:{record['mean_rel_change'][mask]:.1e}" for mask in record["mean_rel_change"]
            )
            logger.info(
                "  sample %d/%d %s seq=%d images=%d %s %.0fs  rel_change(%s)",
                next_idx,
                len(sample_list),
                sample_id,
                info.seq_len,
                info.n_image_tokens,
                "pos=" + str(info.mask_positions),
                info.seconds,
                change,
            )
        if checkpoint_every is not None and next_idx % checkpoint_every == 0:
            write_checkpoint()

    write_checkpoint()

    lenses: dict[str, JacobianLens] = {}
    for mask in masks:
        if n_done[mask] == 0:
            logger.warning("mask %r received no samples; no lens written for it", mask)
            continue
        mean = {layer: jacobian_sum[mask][layer] / n_done[mask] for layer in source_layers}
        lenses[mask] = JacobianLens(jacobians=mean, n_prompts=n_done[mask], d_model=d_model)

    if not lenses:
        raise ValueError("no samples were usable for any mask")

    logger.info(
        "fit: done; %s",
        ", ".join(f"{mask}={lens.n_prompts} prompts" for mask, lens in lenses.items()),
    )
    return FitResult(lenses=lenses, history=history, config=config, skipped=skipped)


def configure_tf32(enabled: bool = True) -> dict[str, bool]:
    """Turn TF32 matmul/cuDNN on (or off) and report the effective flag.

    ``torch`` defaults ``allow_tf32`` to False, so an fp32 fit on Hopper runs on CUDA cores
    at a fraction of tensor-core throughput (measured on the LLaVA pilot: ~10 min versus
    ~4.6 min per 660-token sample for bf16). TF32 keeps fp32 storage and accumulation with
    a 10-bit mantissa, which is the intended precision for the fp32 comparison fit
    (register D20 / experiment X6). Torch >= 2.9 exposes the same knob as
    ``fp32_precision``; both are set when present.
    """
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)
    try:  # newer API; setting it also flips allow_tf32 where both exist
        torch.backends.cuda.matmul.fp32_precision = "tf32" if enabled else "ieee"
    except (AttributeError, TypeError, RuntimeError):
        pass
    return {"allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32)}


def convergence_summary(history: Sequence[dict[str, Any]], *, window: int = 10) -> dict[str, float]:
    """Max ``mean_rel_change`` over the last ``window`` samples, per mask.

    The fit script reports this to decide whether to stop at the staged budget or keep
    going (''extend if needed''): the relative shift of the running mean falls like 1/n
    once the estimate has settled.
    """
    tail = history[-window:]
    summary: dict[str, float] = {}
    for record in tail:
        for mask, value in record.get("mean_rel_change", {}).items():
            if value != value:  # NaN
                continue
            summary[mask] = max(summary.get(mask, 0.0), float(value))
    return summary


def drop_stats(n_seen: int, skipped: Sequence[str], *, examples: int = 10) -> dict[str, Any]:
    """Drop-rate summary for provenance (V12).

    ``n_seen`` counts the samples this invocation considered (post ``limit``/``shard``);
    samples the corpus builder discarded for length are reported separately in the
    manifest header (``n_dropped_over_length``).
    """
    seen = int(n_seen)
    return {
        "n_samples_seen": seen,
        "n_skipped": len(skipped),
        "drop_rate": (len(skipped) / seen) if seen else 0.0,
        "skipped_examples": list(skipped[:examples]),
    }


__all__ = [
    "CHECKPOINT_VERSION",
    "FitInfo",
    "FitResult",
    "MM_SKIP_FIRST",
    "TEXT_SKIP_FIRST",
    "configure_tf32",
    "convergence_summary",
    "drop_stats",
    "fit_masked",
    "jacobian_for_sample",
    "model_fingerprint",
]
