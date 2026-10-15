# SPDX-License-Identifier: Apache-2.0
"""Per-layer affine calibration of the J lens: the tuned-lens completion.

Why
---
The Jacobian transport maps a mid-block residual into the final-layer basis with the
right *directions* but the wrong *calibration*: against the model's own next-token
distribution the lens softmax is systematically offset and mis-scaled. A per-layer
affine on the lens *logits* - ``z = scale * x + bias`` with a full-vocabulary
``bias`` - is the smallest fix that preserves the transport's geometry (the
tuned-lens-style head for the J transport), fitted by distillation: cross-entropy
between ``softmax(scale * x + bias)`` and the model's teacher-forced next-token
distribution at the mask's source positions, with a sample-level hold-out so the
early-stop measures generalization to unseen prompts.

The lens logits ``x`` are produced by exactly the readout path
(``readout.lens_readout``): the same ``ActivationRecorder`` capture of pre-norm block
outputs, the same fp32 residual cast, the same ``lens.transport`` call, and the model's
final norm inside ``unembed`` - so a calibrated readout is the plain one with
``logit_scale``/``logit_bias`` applied after ``unembed`` (see :func:`bias_payload` and
:func:`affine_readout_kwargs`). A residual-space multiplier (readout's ``scale`` kwarg)
is a no-op under the final RMSNorm, which is why the fitted scale is a *logit-space*
``logit_scale`` instead.

Identification: ``softmax`` is invariant to adding a constant to every logit, so
``bias`` is identified only up to ``+ c * 1``; the mean-residual initialization
(``bias0 = log mean_target - log mean_softmax(x)``) pins the constant, and the fitted
``scale`` is unaffected. Memory: positions are streamed in mini-batches - nothing
materializes ``[n_positions, vocab]`` for all positions, each chunk's logits live only
for its backward pass, and the fitted parameters are one vocab-sized fp32 vector per
layer.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import torch
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens

from vlm_lens._batch import as_batch
from vlm_lens.data.manifest import FitSample
from vlm_lens.fitting import model_fingerprint
from vlm_lens.models.llava import LlavaLensModel, MultimodalBatch
from vlm_lens.positions import ALL_MASKS, build_position_masks

logger = logging.getLogger(__name__)

#: Bumped when the payload layout changes; recorded in ``meta`` (additive evolution only).
CALIBRATION_VERSION = 1

#: What ``meta.base_lens.fingerprint`` means (the x21 convention, same hash).
FINGERPRINT_NOTE = (
    "sha256 over the loaded lens's fitted J_l values (layer tag + raw float32 bytes, "
    "layer order): equal report metrics with equal fingerprints = the same fitted lens "
    "was read; equal metrics with different fingerprints = a lens-blind readout"
)


def lens_fingerprint(lens: JacobianLens) -> str:
    """SHA-256 over the fitted ``J_l`` values: the loaded lens's *content* identity.

    The x21 pattern: each layer's J is hashed as raw float32 bytes tagged with its
    layer index, in ``source_layers`` order. Recorded in the calibration provenance so
    a bias file stays verifiably bound to the lens it was fitted against.
    """
    digest = hashlib.sha256()
    for layer in lens.source_layers:
        J = lens.jacobians[layer].detach().to(device="cpu", dtype=torch.float32).contiguous()
        digest.update(f"L{layer}:".encode("ascii"))
        digest.update(J.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class LayerCalibration:
    """Fitted affine for one layer: calibrated lens logits are ``scale * x + bias``.

    ``bias`` is a fp32 ``[vocab]`` CPU tensor. ``pre``/``post`` hold the hold-out
    metrics before/after calibration (train metrics when no hold-out was requested):
    ``n``, ``top1_agreement``, ``mean_kl``, ``holdout_ce`` and, when next-token ids
    were available, ``mean_rank_true`` - the same definitions as
    :mod:`vlm_lens.evaluate`.
    """

    layer: int
    bias: torch.Tensor
    scale: float
    pre: dict[str, float]
    post: dict[str, float]


@dataclass(frozen=True)
class CalibrationResult:
    """Per-layer affine calibrations plus the provenance binding them to a lens."""

    layers: dict[int, LayerCalibration]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class _Captured:
    """One sample's captured residuals and its calibratable positions."""

    input_ids: torch.Tensor  # [seq_len] long, CPU
    positions: torch.Tensor  # [n_positions] long, CPU: mask source positions
    residuals: dict[int, torch.Tensor]  # layer -> [seq_len, d_model] fp32 CPU


def _check_layers(model: LlavaLensModel, lens: JacobianLens, layers: Sequence[int]) -> list[int]:
    """Validated sorted layer list: fitted, in range, never the (identity) final layer."""
    final_layer = model.n_layers - 1
    unknown = [layer for layer in layers if layer not in lens.jacobians]
    if unknown:
        raise ValueError(f"layers {unknown} are not fitted; fitted layers are {lens.source_layers}")
    if any(not 0 <= layer < model.n_layers for layer in layers):
        raise ValueError(f"layers {layers} out of range for a {model.n_layers}-layer model")
    if final_layer in layers:
        raise ValueError(
            f"layer {final_layer} is the final layer: its lens readout is the model's "
            "own logits, so there is nothing to calibrate"
        )
    return sorted(set(layers))


def _split_samples(n_samples: int, *, holdout_frac: float, seed: int) -> tuple[list[int], list[int]]:
    """Seeded sample-level split: hold-out is ``round(frac * n)`` of a shuffled order."""
    if not 0.0 <= holdout_frac < 1.0:
        raise ValueError(f"holdout_frac must be in [0, 1), got {holdout_frac}")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n_samples, generator=generator).tolist()
    n_holdout = int(round(holdout_frac * n_samples))
    if holdout_frac > 0.0 and n_samples > 1 and n_holdout == 0:
        n_holdout = 1
    return order[n_holdout:], order[:n_holdout]


def _capture_sample(
    model: LlavaLensModel,
    sample: FitSample | MultimodalBatch | str,
    *,
    layers: Sequence[int],
    mask: str,
    skip_first: int,
    max_seq_len: int,
    residual_dtype: torch.dtype,
) -> _Captured:
    """One teacher-forced pass: block outputs at ``layers`` + final, and mask positions.

    Mirrors ``readout.lens_readout``'s capture exactly (same recorder, same forward,
    fp32 residual cast) so the fitted affine applies to the readout's own logits.
    """
    batch = as_batch(model, sample, max_seq_len)
    final_layer = model.n_layers - 1
    record_at = sorted(set(layers) | {final_layer})
    with torch.no_grad():
        with ActivationRecorder(model.layers, at=record_at) as recorder:
            model.forward_mm(batch)
        residuals = {
            layer: recorder.activations[layer][0].detach().to("cpu", residual_dtype)
            for layer in record_at
        }
        source_mask = build_position_masks(
            batch.input_ids,
            model.image_token_id,
            skip_first=skip_first,
            exclude_last=True,
            masks=(mask,),
        )[mask]
        positions = source_mask.nonzero(as_tuple=True)[0].to("cpu")
    return _Captured(
        input_ids=batch.input_ids[0].detach().cpu(),
        positions=positions,
        residuals=residuals,
    )


def _affine_metrics(x: torch.Tensor, target: torch.Tensor, *, token_ids: torch.Tensor | None) -> dict[str, float]:
    """Metrics of ``x`` (already scaled/shifted) against the model's distribution.

    Same definitions as :mod:`vlm_lens.evaluate`: rank of the true token (ties count
    as better ranks), top-1 agreement with the model's argmax, KL(model || lens), and
    the distillation cross-entropy.
    """
    log_p = torch.log_softmax(x, dim=-1)
    log_q = torch.log_softmax(target, dim=-1)
    n = int(x.shape[0])
    metrics: dict[str, float] = {"n": float(n)}
    if token_ids is not None:
        target_logit = x.gather(1, token_ids[:, None]).squeeze(1)
        rank = (x > target_logit[:, None]).sum(dim=1).to(torch.float32) + 1.0
        metrics["mean_rank_true"] = float(rank.mean())
    metrics["top1_agreement"] = float((x.argmax(dim=1) == target.argmax(dim=1)).float().mean())
    metrics["mean_kl"] = float((log_q.exp() * (log_q - log_p)).sum(dim=-1).mean())
    metrics["holdout_ce"] = float(-(log_q.exp() * log_p).sum(dim=-1).mean())
    return metrics


def fit_logit_affines(
    collect: Callable[[torch.Tensor, Sequence[int]], tuple[dict[int, torch.Tensor], torch.Tensor]],
    *,
    layers: Sequence[int],
    train_rows: torch.Tensor,
    holdout_rows: torch.Tensor,
    token_ids: torch.Tensor | None = None,
    epochs: int,
    lr: float,
    batch_positions: int,
    patience: int = 2,
    device: torch.device | str | None = None,
    seed: int = 0,
) -> dict[int, LayerCalibration]:
    """Fit ``z = scale * x + bias`` per layer by distillation against ``collect``'s target.

    ``collect(rows, layers)`` returns ``({layer: x_chunk [c, vocab] fp32}, target_chunk
    [c, vocab] fp32)`` for the flat row indices ``rows`` - the lens logits and the
    model's teacher-forced logits, both treated as constants. Positions are streamed in
    mini-batches of ``batch_positions`` rows; the per-layer Adam parameters are one
    vocab-sized fp32 bias plus a scalar scale. Bias starts from the mean-residual
    trick (``log mean_target - log mean_softmax(x)`` over the train rows), scale starts
    at 1.0. Early stopping freezes a layer when its hold-out CE stops improving for
    ``patience`` epochs (``patience=0`` or an empty ``holdout_rows`` disables it) and
    the best state is restored before the post metrics are taken.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if lr <= 0.0:
        raise ValueError(f"lr must be > 0, got {lr}")
    if batch_positions < 1:
        raise ValueError(f"batch_positions must be >= 1, got {batch_positions}")
    if patience < 0:
        raise ValueError(f"patience must be >= 0, got {patience}")
    if int(train_rows.numel()) == 0:
        raise ValueError("no trainable positions: the mask selected no source positions")
    layers = sorted(set(layers))
    device = torch.device("cpu") if device is None else torch.device(device)
    score_rows = holdout_rows if int(holdout_rows.numel()) else train_rows

    # Initialization sweep: the mean-residual trick for each layer's bias, scale 1.0.
    generator = torch.Generator().manual_seed(seed)
    order = train_rows[torch.randperm(int(train_rows.numel()), generator=generator)]
    mean_q: torch.Tensor | None = None
    mean_p: dict[int, torch.Tensor | None] = {layer: None for layer in layers}
    seen = 0
    with torch.no_grad():
        for start in range(0, int(order.numel()), batch_positions):
            chunk_x, target = collect(order[start : start + batch_positions], layers)
            target = target.to(device).float()
            probs = torch.softmax(target, dim=-1).sum(dim=0)
            mean_q = probs if mean_q is None else mean_q + probs
            for layer in layers:
                chunk_probs = torch.softmax(chunk_x[layer].to(device).float(), dim=-1).sum(dim=0)
                mean_p[layer] = chunk_probs if mean_p[layer] is None else mean_p[layer] + chunk_probs
            seen += int(target.shape[0])
    params: dict[int, dict[str, Any]] = {}
    for layer in layers:
        bias0 = (mean_q / seen).clamp_min(1e-12).log() - (mean_p[layer] / seen).clamp_min(1e-12).log()
        bias_param = torch.nn.Parameter(bias0.to(device))
        scale_param = torch.nn.Parameter(torch.ones((), device=device))
        params[layer] = {
            "bias": bias_param,
            "scale": scale_param,
            "optimizer": torch.optim.Adam([bias_param, scale_param], lr=lr),
            "best_ce": float("inf"),
            "best_bias": bias0.to(device),
            "best_scale": 1.0,
            "wait": 0,
            "frozen": False,
        }

    def calibrated_metrics(rows: torch.Tensor, *, calibrated: bool) -> dict[int, dict[str, float]]:
        """Metrics over ``rows``; ``calibrated=False`` scores the raw lens logits."""
        sums: dict[int, dict[str, float]] = {layer: {} for layer in layers}
        counts = 0
        with torch.no_grad():
            for start in range(0, int(rows.numel()), batch_positions):
                chunk_rows = rows[start : start + batch_positions]
                chunk_x, target = collect(chunk_rows, layers)
                target = target.to(device).float()
                chunk_token_ids = None if token_ids is None else token_ids[chunk_rows.cpu()].to(device)
                for layer in layers:
                    x_chunk = chunk_x[layer].to(device).float()
                    if calibrated:
                        x_chunk = params[layer]["scale"] * x_chunk + params[layer]["bias"]
                    metrics = _affine_metrics(x_chunk, target, token_ids=chunk_token_ids)
                    for key, value in metrics.items():
                        if key == "n":
                            continue
                        sums[layer][key] = sums[layer].get(key, 0.0) + value * metrics["n"]
                counts += int(target.shape[0])
        return {
            layer: {**{key: value / counts for key, value in accumulator.items()}, "n": float(counts)}
            for layer, accumulator in sums.items()
        }

    def train_pass() -> None:
        """One Adam pass over the train rows (frozen layers are skipped)."""
        active = [layer for layer in layers if not params[layer]["frozen"]]
        if not active:
            return
        permutation = train_rows[torch.randperm(int(train_rows.numel()), generator=generator)]
        for start in range(0, int(permutation.numel()), batch_positions):
            chunk_rows = permutation[start : start + batch_positions]
            active = [layer for layer in layers if not params[layer]["frozen"]]
            if not active:
                return
            chunk_x, target = collect(chunk_rows, active)
            with torch.no_grad():
                q = torch.softmax(target.to(device).float(), dim=-1)
            for layer in active:
                state = params[layer]
                x_chunk = chunk_x[layer].to(device).float()
                log_p = torch.log_softmax(state["scale"] * x_chunk + state["bias"], dim=-1)
                loss = -(q * log_p).sum(dim=-1).mean()
                state["optimizer"].zero_grad()
                loss.backward()
                state["optimizer"].step()

    def holdout_pass() -> dict[int, float]:
        """Per-layer hold-out CE with the current parameters (inf without a hold-out)."""
        active = [layer for layer in layers if not params[layer]["frozen"]]
        if int(holdout_rows.numel()) == 0 or not active:
            return {layer: float("inf") for layer in active}
        ces = {layer: 0.0 for layer in active}
        counts = 0
        with torch.no_grad():
            for start in range(0, int(holdout_rows.numel()), batch_positions):
                chunk_rows = holdout_rows[start : start + batch_positions]
                chunk_x, target = collect(chunk_rows, active)
                log_q = torch.log_softmax(target.to(device).float(), dim=-1)
                for layer in active:
                    state = params[layer]
                    x_chunk = chunk_x[layer].to(device).float()
                    log_p = torch.log_softmax(state["scale"] * x_chunk + state["bias"], dim=-1)
                    ces[layer] += float(-(log_q.exp() * log_p).sum())
                counts += int(target.shape[0])
        return {layer: ces[layer] / counts for layer in active}

    pre_metrics = calibrated_metrics(score_rows, calibrated=False)
    for epoch in range(1, epochs + 1):
        train_pass()
        holdout_ce = holdout_pass()
        for layer, ce in holdout_ce.items():
            state = params[layer]
            if ce < state["best_ce"] - 1e-6:
                state["best_ce"] = ce
                state["best_bias"] = state["bias"].detach().clone()
                state["best_scale"] = float(state["scale"].detach())
                state["wait"] = 0
            else:
                state["wait"] += 1
                if patience > 0 and state["wait"] >= patience:
                    state["frozen"] = True
                    logger.info(
                        "calibrate: layer %d early-stopped after epoch %d "
                        "(hold-out CE %.4f, best %.4f)",
                        layer,
                        epoch,
                        ce,
                        state["best_ce"],
                    )
        if all(params[layer]["frozen"] for layer in layers):
            break
    # Restore each layer's best hold-out state before the post metrics.
    for layer in layers:
        state = params[layer]
        with torch.no_grad():
            state["bias"].copy_(state["best_bias"])
            state["scale"].fill_(state["best_scale"])
    post_metrics = calibrated_metrics(score_rows, calibrated=True)

    fitted: dict[int, LayerCalibration] = {}
    for layer in layers:
        fitted[layer] = LayerCalibration(
            layer=layer,
            bias=params[layer]["best_bias"].detach().to("cpu", torch.float32).clone(),
            scale=float(params[layer]["best_scale"]),
            pre=dict(pre_metrics[layer]),
            post=dict(post_metrics[layer]),
        )
    return fitted


def fit_calibration(
    model: LlavaLensModel,
    lens: JacobianLens,
    samples: Sequence[FitSample | MultimodalBatch | str],
    *,
    mask: str = "text",
    layers: Sequence[int] | None = None,
    holdout_frac: float = 0.2,
    epochs: int,
    lr: float,
    batch_positions: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    seed: int = 0,
    patience: int = 2,
    skip_first: int = 1,
    max_seq_len: int = 1536,
) -> CalibrationResult:
    """Calibrate ``lens`` per layer against the model's own next-token distribution.

    Args:
        mask: Source-position mask (see :mod:`vlm_lens.positions`); only positions of
            this mask feed the fit and the hold-out metrics.
        layers: Layers to calibrate; defaults to all fitted ``source_layers``. Every
            layer must be fitted (the final layer's readout is the model itself).
        holdout_frac: Sample-level hold-out fraction (split by sample, not position).
        epochs / lr / batch_positions: Adam budget and the streamed position
            mini-batch size.
        device: Where the transport/unembed matmuls and Adam run; default the model's
            unembedding device. The lens's J matrices are moved there in place.
        dtype: Residual-storage dtype; default fp32 (readout-identical). Fitting
            itself always accumulates fp32.
        patience / skip_first / max_seq_len: Early-stop patience, leading-position
            exclusion (multimodal default 1) and the encode truncation length.

    The returned :class:`CalibrationResult` records the base-lens content fingerprint
    (x21 pattern) so a written bias file stays bound to the lens it was fitted against.
    """
    if not samples:
        raise ValueError("no samples to calibrate on")
    if mask not in ALL_MASKS:
        raise ValueError(f"unknown mask {mask!r}; expected one of {list(ALL_MASKS)}")
    fitted_layers = _check_layers(model, lens, lens.source_layers if layers is None else layers)
    device = torch.device(device) if device is not None else model.unembed_weight().device
    residual_dtype = torch.float32 if dtype is None else dtype
    final_layer = model.n_layers - 1
    vocab_size = int(model.unembed_weight().shape[0])
    # Production readouts run on a device-resident lens; move J once so per-chunk
    # transport does not re-copy the [d_model, d_model] matrices.
    for layer in fitted_layers:
        lens.jacobians[layer] = lens.jacobians[layer].to(device)

    logger.info(
        "calibrate: %d samples, mask=%s, layers=%s, holdout_frac=%.2f, device=%s",
        len(samples),
        mask,
        fitted_layers,
        holdout_frac,
        device,
    )
    captured: list[_Captured] = []
    capture_started = time.monotonic()
    for index, sample in enumerate(samples):
        captured.append(
            _capture_sample(
                model,
                sample,
                layers=fitted_layers,
                mask=mask,
                skip_first=skip_first,
                max_seq_len=max_seq_len,
                residual_dtype=residual_dtype,
            )
        )
        if (index + 1) % 10 == 0 or index + 1 == len(samples):
            logger.info(
                "calibrate: captured %d/%d samples (%.1fs elapsed)",
                index + 1,
                len(samples),
                time.monotonic() - capture_started,
            )

    sample_parts: list[torch.Tensor] = []
    position_parts: list[torch.Tensor] = []
    token_parts: list[torch.Tensor] = []
    for index, item in enumerate(captured):
        n_pos = int(item.positions.numel())
        sample_parts.append(torch.full((n_pos,), index, dtype=torch.long))
        position_parts.append(item.positions.to(torch.long))
        token_parts.append(item.input_ids[item.positions.to(torch.long) + 1])
    sample_of_row = torch.cat(sample_parts) if sample_parts else torch.empty(0, dtype=torch.long)
    position_of_row = torch.cat(position_parts) if position_parts else torch.empty(0, dtype=torch.long)
    token_ids = torch.cat(token_parts) if token_parts else torch.empty(0, dtype=torch.long)
    if int(sample_of_row.numel()) == 0:
        raise ValueError(f"mask {mask!r} selected no source positions in any sample")

    train_samples, holdout_samples = _split_samples(len(captured), holdout_frac=holdout_frac, seed=seed)
    train_rows = torch.isin(sample_of_row, torch.tensor(train_samples, dtype=torch.long)).nonzero(as_tuple=True)[0]
    holdout_rows = torch.isin(sample_of_row, torch.tensor(holdout_samples, dtype=torch.long)).nonzero(as_tuple=True)[0]
    if int(holdout_rows.numel()) == 0:
        logger.warning(
            "calibrate: empty hold-out (%d samples, frac %.2f); metrics come from the "
            "train rows and early stopping is disabled",
            len(captured),
            holdout_frac,
        )

    def collect(rows: torch.Tensor, requested: Sequence[int]) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """Lens + model logits for the flat ``rows`` (constants; streamed, no_grad)."""
        rows = rows.detach().to("cpu").long()
        out_x: dict[int, torch.Tensor] = {}
        out_t = torch.empty(int(rows.numel()), vocab_size, dtype=torch.float32)
        with torch.no_grad():
            for layer in requested:
                out_x[layer] = torch.empty(int(rows.numel()), vocab_size, dtype=torch.float32)
            for sample_index in torch.unique(sample_of_row[rows], sorted=True).tolist():
                selector = sample_of_row[rows] == sample_index
                positions = position_of_row[rows[selector]]
                item = captured[int(sample_index)]
                hidden = item.residuals[final_layer][positions].to(device)
                out_t[selector] = model.unembed(hidden).float().cpu()
                for layer in requested:
                    hidden = item.residuals[layer][positions].to(device)
                    out_x[layer][selector] = model.unembed(lens.transport(hidden, layer)).float().cpu()
        return out_x, out_t

    fitted = fit_logit_affines(
        collect,
        layers=fitted_layers,
        train_rows=train_rows,
        holdout_rows=holdout_rows,
        token_ids=token_ids,
        epochs=epochs,
        lr=lr,
        batch_positions=batch_positions,
        patience=patience,
        device=device,
        seed=seed,
    )
    for layer, calibration in sorted(fitted.items()):
        logger.info(
            "calibrate: layer %d scale=%.4f hold-out CE %.4f -> %.4f",
            layer,
            calibration.scale,
            calibration.pre["holdout_ce"],
            calibration.post["holdout_ce"],
        )

    provenance: dict[str, Any] = {
        "calibration_version": CALIBRATION_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mask": mask,
        "layers": fitted_layers,
        "holdout_frac": holdout_frac,
        "epochs": epochs,
        "lr": lr,
        "batch_positions": batch_positions,
        "patience": patience,
        "skip_first": skip_first,
        "max_seq_len": max_seq_len,
        "seed": seed,
        "device": str(device),
        "residual_dtype": str(residual_dtype),
        "n_samples": len(samples),
        "n_train_samples": len(train_samples),
        "n_holdout_samples": len(holdout_samples),
        "n_train_positions": int(train_rows.numel()),
        "n_holdout_positions": int(holdout_rows.numel()),
        "d_model": int(model.d_model),
        "model": model_fingerprint(model),
        "base_lens": {
            "fingerprint": lens_fingerprint(lens),
            "fingerprint_note": FINGERPRINT_NOTE,
            "source_layers": list(lens.source_layers),
            "n_prompts": int(lens.n_prompts),
            "d_model": int(lens.d_model),
        },
    }
    return CalibrationResult(layers=fitted, provenance=provenance)


def bias_payload(result: CalibrationResult) -> dict[str, Any]:
    """The ``bias-<mask>.pt`` payload for :func:`vlm_lens.artifacts.save_bias`.

    Reuses the existing affine-correction schema: the schema-required residual-space
    keys are inert (``bias`` zeros, ``scale`` 1.0 - a residual multiplier is a no-op
    under the final norm), and the fitted affine rides the additive logit-space keys
    ``logit_bias`` (fp32 ``[vocab]``) and ``logit_scale`` (float), applied by
    ``readout.lens_readout`` as ``z = z / temp * logit_scale + logit_bias``. The
    calibration provenance (including the base-lens fingerprint) goes to ``meta``.
    """
    d_model = int(result.provenance.get("d_model") or 0)
    if d_model <= 0:
        raise ValueError("calibration provenance lacks a usable 'd_model'")
    payload: dict[str, Any] = {}
    for layer, calibration in sorted(result.layers.items()):
        payload[str(layer)] = {
            "bias": torch.zeros(d_model, dtype=torch.float32),
            "scale": 1.0,
            "logit_bias": calibration.bias.detach().to("cpu", torch.float32).clone(),
            "logit_scale": float(calibration.scale),
        }
    payload["meta"] = {
        **result.provenance,
        "kind": "calibration",
        "calibration_version": CALIBRATION_VERSION,
    }
    return payload


def affine_readout_kwargs(payload: Mapping[str, Any]) -> dict[str, dict[int, Any]]:
    """``lens_readout(...)`` kwargs for a loaded bias payload (both schema variants).

    Returns ``{"logit_bias": {layer: Tensor[vocab]}, "logit_scale": {layer: float}}``
    from the calibration schema's additive keys; entries without a ``logit_bias``
    (the older residual-space affine-correction files) are omitted, and a missing
    ``logit_scale`` defaults to 1.0.
    """
    logit_bias: dict[int, torch.Tensor] = {}
    logit_scale: dict[int, float] = {}
    for key, entry in payload.items():
        if key == "meta" or not isinstance(entry, Mapping):
            continue
        bias = entry.get("logit_bias")
        if not isinstance(bias, torch.Tensor):
            continue
        layer = int(key)
        logit_bias[layer] = bias.detach().to(device="cpu", dtype=torch.float32)
        logit_scale[layer] = float(entry.get("logit_scale", 1.0))
    return {"logit_bias": logit_bias, "logit_scale": logit_scale}


__all__ = [
    "CALIBRATION_VERSION",
    "CalibrationResult",
    "FINGERPRINT_NOTE",
    "LayerCalibration",
    "affine_readout_kwargs",
    "bias_payload",
    "fit_calibration",
    "fit_logit_affines",
    "lens_fingerprint",
]
