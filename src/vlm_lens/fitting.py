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
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from jlens.fitting import _check_layer_indices
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens

from vlm_lens._batch import as_batch
from vlm_lens.data.manifest import FitSample, read_manifest
from vlm_lens.models.llava import LlavaLensModel, MultimodalBatch
from vlm_lens.positions import DEFAULT_MASKS, build_position_masks
from vlm_lens.readout import lens_readout

logger = logging.getLogger(__name__)

#: Text-only control fits mirror the paper's protocol.
TEXT_SKIP_FIRST = 16
#: Multimodal fits: only BOS is a pure sink; every image position carries content.
MM_SKIP_FIRST = 1

#: The online probe always scores with s2_eval's protocol: only the BOS sink dropped
#: (independent of the fit's ``skip_first``), one-step next-token targets, last position
#: and placeholder-continuation targets excluded.
PROBE_SKIP_FIRST = 1

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
    """Masked lenses plus the per-sample history the fit script reports on.

    ``probe_history`` holds the optional online-probe learning curve (see
    :func:`fit_masked`): one row per scored layer per probe.
    """

    lenses: dict[str, JacobianLens]
    history: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    probe_history: list[dict[str, Any]] = field(default_factory=list)

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
    """``torch.save`` via a tmp file + rename, with disk-full diagnostics.

    A full disk truncates ``torch.save``'s zip container mid-stream, which surfaces as an
    opaque ``unexpected pos X vs Y`` RuntimeError and leaves a large ``*.tmp.<pid>`` behind
    (2026-10-03: /data at 100 % left 4.3 GB of garbage). Free space is sampled before the
    write and the tmp file is removed on any failure, so a full disk fails fast, says so,
    and cleans up after itself.
    """
    path = os.fspath(path)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        free: int | None = shutil.disk_usage(os.path.dirname(os.path.abspath(path))).free
    except OSError:  # not a local filesystem path; torch.save will surface any real error
        free = None
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except Exception as error:  # noqa: BLE001 - re-raised with disk context below
        try:
            os.unlink(tmp)
        except OSError:
            pass
        hint = f"free={free / 2**20:.0f} MiB at save time" if free is not None else "free=unknown"
        raise RuntimeError(f"failed to write {path!r} ({hint}): {error}") from error


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


        heartbeat_every = max(1, n_passes // 10)
        pass_t0 = time.perf_counter()
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
            if (pass_idx + 1) % heartbeat_every == 0 or pass_idx in (0, n_passes - 1):
                logger.info(
                    "    pass %d/%d (%.0f%%) elapsed=%.0fs gpu=%.1fGiB",
                    pass_idx + 1,
                    n_passes,
                    100.0 * (pass_idx + 1) / n_passes,
                    time.perf_counter() - pass_t0,
                    torch.cuda.memory_allocated() / 2**30,
                )

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


def _probe_positions(
    model: LlavaLensModel,
    batch: MultimodalBatch,
    tag: str,
) -> list[int] | None:
    """Scored positions for one probe sample (s2_eval's default mode), or ``None``.

    The ``tag`` mask selects the positions (``skip_first=PROBE_SKIP_FIRST``); each is
    scored against its *next* token, so the final position (no target) and
    placeholder-continuation targets are excluded, exactly like s2_eval's text-tag
    scoring.
    """
    mask = build_position_masks(
        batch.input_ids, model.image_token_id, skip_first=PROBE_SKIP_FIRST, masks=(tag,)
    )[tag]
    if not bool(mask.any()):
        return None
    input_ids = batch.input_ids.detach().cpu().reshape(-1)
    seq_len = int(input_ids.shape[0])
    positions = [
        int(p)
        for p in mask.cpu().nonzero(as_tuple=True)[0]
        if int(p) + 1 < seq_len and int(input_ids[int(p) + 1]) != model.image_token_id
    ]
    return positions or None


@torch.no_grad()
def _probe_eval(
    model: LlavaLensModel,
    lens: JacobianLens,
    samples: Sequence[MultimodalBatch | FitSample | str],
    tag: str,
    *,
    max_seq_len: int,
) -> list[dict[str, float]]:
    """Score one (running) lens on held-out ``samples``; one dict per scored layer.

    Positions and targets follow :func:`_probe_positions`; every metric is a mean over
    the scored positions of all samples: ``kl`` is the mean true-token
    ``KL(lens_softmax || model_softmax)``, ``rank`` the mean 1-based rank of the true
    token in the lens's logit row (ties count as better ranks), ``top1`` the fraction of
    positions where the lens's argmax matches the model's. The final layer is always
    scored (it reads out the model's own logits unless it was itself fitted). Returns
    ``[]`` when no sample has a valid position.
    """
    final_layer = model.n_layers - 1
    score_layers = sorted(set(lens.source_layers) | {final_layer})
    sums = {layer: {"kl": 0.0, "rank": 0.0, "top1": 0.0} for layer in score_layers}
    n_positions = 0
    for sample in samples:
        batch = as_batch(model, sample, max_seq_len)
        positions = _probe_positions(model, batch, tag)
        if positions is None:
            continue
        readout = lens_readout(
            model,
            lens,
            batch,
            layers=score_layers,
            positions=positions,
            max_seq_len=max_seq_len,
        )
        targets = readout.input_ids[torch.tensor([p + 1 for p in positions], dtype=torch.long)]
        model_top1 = readout.model_logits.argmax(dim=1)
        model_log_probs = torch.log_softmax(readout.model_logits, dim=-1)
        for layer in score_layers:
            lens_rows = readout.lens_logits[layer]
            target_logit = lens_rows.gather(1, targets[:, None]).squeeze(1)
            rank = (lens_rows > target_logit[:, None]).sum(dim=1) + 1
            sums[layer]["kl"] += float(
                F.kl_div(
                    model_log_probs,
                    torch.log_softmax(lens_rows, dim=-1),
                    reduction="none",
                    log_target=True,
                ).sum()
            )
            sums[layer]["rank"] += float(rank.sum())
            sums[layer]["top1"] += float((lens_rows.argmax(dim=1) == model_top1).sum())
        n_positions += len(positions)
    if n_positions == 0:
        return []
    return [
        {
            "layer": int(layer),
            "kl": sums[layer]["kl"] / n_positions,
            "rank": sums[layer]["rank"] / n_positions,
            "top1": sums[layer]["top1"] / n_positions,
        }
        for layer in score_layers
    ]


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
    probe_every: int | None = None,
    probe_manifest: str | Path | None = None,
    probe_n: int = 16,
    probe_tag: str = "text",
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
        probe_every: Online-probe cadence in *used* samples (skipped samples do not
            count); ``None`` disables probing entirely.
        probe_manifest: Held-out manifest the running lens is scored on; probing
            requires both this and ``probe_every``.
        probe_n: Score the running lens on the first ``probe_n`` probe-manifest samples.
        probe_tag: Which position tag the probe scores (a mask name, e.g. ``text``).

    Returns:
        :class:`FitResult` with one :class:`JacobianLens` per mask that received at least
        one sample (``n_prompts`` is per mask). With the online probe active,
        ``probe_history`` holds one row per scored layer per probe —
        ``{"n_samples", "mask", "layer", "kl", "rank", "top1", "elapsed_s"}`` — recording
        the running lens's held-out performance while fitting.
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

    probe_history: list[dict[str, Any]] = []
    probe_active = False
    probe_stride = 0
    probe_batches: list[MultimodalBatch] = []
    probe_started = 0.0
    if probe_every is not None:
        if probe_manifest is None:
            logger.info("online probe requested but no probe manifest given; probe disabled")
        elif probe_every < 1:
            raise ValueError(f"probe_every must be >= 1, got {probe_every}")
        elif probe_n < 1:
            raise ValueError(f"probe_n must be >= 1, got {probe_n}")
        else:
            probe_stride = int(probe_every)
            for probe_sample in read_manifest(probe_manifest)[: int(probe_n)]:
                try:
                    probe_batches.append(as_batch(model, probe_sample, max_seq_len))
                except ValueError as exc:
                    logger.warning(
                        "  probe: skipping probe sample %s (%s)", probe_sample.sample_id, exc
                    )
            if not probe_batches or not any(
                _probe_positions(model, batch, probe_tag) for batch in probe_batches
            ):
                logger.info(
                    "online probe disabled: no valid %r positions in %d probe sample(s)",
                    probe_tag,
                    len(probe_batches),
                )
            else:
                probe_active = True
                probe_started = time.perf_counter()

    def run_probe(n_seen: int) -> list[dict[str, Any]]:
        """One online probe: score each mask's running lens on the held-out samples.

        Read-only w.r.t. the estimator: the per-layer means are new tensors (exactly the
        ones the final write builds), wrapped in a throwaway :class:`JacobianLens` and
        scored under ``torch.no_grad``; ``jacobian_sum``/``n_done`` are never touched.
        """
        elapsed = time.perf_counter() - probe_started
        rows: list[dict[str, Any]] = []
        for mask in masks:
            if n_done[mask] == 0:
                continue
            mean = {layer: jacobian_sum[mask][layer] / n_done[mask] for layer in source_layers}
            running = JacobianLens(jacobians=mean, n_prompts=n_done[mask], d_model=d_model)
            for score in _probe_eval(
                model, running, probe_batches, probe_tag, max_seq_len=max_seq_len
            ):
                rows.append(
                    {
                        "n_samples": n_seen,
                        "mask": mask,
                        "layer": score["layer"],
                        "kl": score["kl"],
                        "rank": score["rank"],
                        "top1": score["top1"],
                        "elapsed_s": round(elapsed, 2),
                    }
                )
        if rows:
            kl_by_mask: dict[str, list[float]] = {}
            for row in rows:
                kl_by_mask.setdefault(str(row["mask"]), []).append(float(row["kl"]))
            logger.info(
                "  probe n=%d %.1fs  kl(%s)",
                n_seen,
                elapsed,
                ", ".join(f"{mask}={sum(kls) / len(kls):.4f}" for mask, kls in kl_by_mask.items()),
            )
        return rows

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

        if probe_active and len(history) % probe_stride == 0:
            probe_history.extend(run_probe(len(history)))

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

    if probe_active:
        # The final probe fires even when the last used sample is not a multiple of
        # probe_every (or when a resume processed no new samples): the curve must end
        # at the final state.
        probe_history.extend(run_probe(len(history)))

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
    return FitResult(
        lenses=lenses,
        history=history,
        config=config,
        skipped=skipped,
        probe_history=probe_history,
    )


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
