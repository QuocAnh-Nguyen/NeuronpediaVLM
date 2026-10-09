#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Tuned-lens-style LOW-RANK translator: per-layer rank-r A_l + bias b_l by KL distillation.

The model-class ladder's next rung above the diagonal census bias+scale: the census
corrects the J-lens readout with one affine map per layer; the translator instead fits a
LOW-RANK linear map,

    lens_l(h) = unembed(A_l x_l + b_l),   A_l = U_l V_l^T  (rank r),

per layer, by KL distillation to the model's own final logits on a FIT split:

    mean_p KL(softmax(z_model) || softmax(unembed(A_l x_l + b_l))),

with ``z_model = unembed(h_final)`` - the model's final-layer logits, never corrected.
The target is the model's SOFT distribution rather than the ground-truth next token:
distilling to soft labels avoids a ground-truth probe learning extra information beyond
what the model itself reads out (Hewitt & Liang 2019). The translator input ``x_l`` is
switchable: ``jacobian`` (default) ``x_l = J_l h_l`` - the transported vector, our
J-lens class extended; ``raw`` ``x_l = h_l`` - the pure tuned lens, the literature's
ceiling for our readout. The source Jacobian ``J_l`` stays FIXED (no new Jacobian
fits); the fit readout is the exact one the zoo applies to the composed artifact,
``unembed(v) = W_U final_norm(v)`` (the final norm included), with ``W_U`` from
``--unembed`` (the saved ``[vocab, d]`` .pt) or, when absent, the live model's
``unembed_weight()`` (the tiny fixture's own ``W_U`` on the tiny backend).

Fit details. ``A_l`` initializes to the rank-r PCA projector of the fit-split ``x`` (the
best rank-r approximation of the tuned-lens identity start under the data covariance;
``torch.svd_lowrank`` - ``torch.svds`` no longer exists), ``b_l`` to zero; Adam on
minibatches of 256 positions for ``--steps`` steps, with 10% of positions held out as a
fit-side val set. ``fit_report.json`` records per-layer train/val KL (first/last/best)
AND the ``||A_l||_F`` Frobenius norms - the tuned-lens v6 undertraining diagnostic (SGD
fits were "severely undertrained" with small weight norms; the norm is the visible tell).

Three passes: (1) cache - one forward pass per fit sample, residuals at the lens layers
(plus the final, for the target) at s2_eval's exact text-tag scoring positions; (2) fit -
the per-layer seeded Adam distillation; (3) write - the composed lens as a synthetic lens
dir plus a bias payload, evaluable by the standard zoo (``--lens name=OUT/artifacts
--bias-dir OUT``).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402, I001  # vlm_lens must precede jlens: it installs the path
import torch  # noqa: E402
import vlm_lens  # noqa: E402, F401
from jlens.hooks import ActivationRecorder  # noqa: E402
from jlens.lens import JacobianLens  # noqa: E402

from vlm_lens._batch import as_batch  # noqa: E402
from vlm_lens.artifacts import MASK_FILENAME, load_lens_set, save_bias, save_lens_set  # noqa: E402
from vlm_lens.data.manifest import read_manifest  # noqa: E402
from vlm_lens.models.llava import LlavaLensModel  # noqa: E402
from vlm_lens.positions import build_position_masks  # noqa: E402

#: The distilled readout, per layer (the assignment's ``W_U (A x + b)``: the zoo's
#: ``unembed`` includes the final norm, so the fit's must too).
ESTIMATOR = (
    "A_l = U_l V_l^T (rank r), fitted per layer by KL distillation to the model's final "
    "logits: mean_p KL(softmax(z_model) || softmax(W_U final_norm(A_l x_l + b_l))), "
    "Adam minibatches; x_l = J_l h_l (jacobian) or h_l (raw); J_l fixed"
)
#: Adam minibatch in positions, also the eval-chunk size (calib_readout's chunk_size).
BATCH_SIZE = 256
#: Eval-checkpoint cadence in steps: the first, every Nth, and the last step record
#: full-train and full-val KL (the trajectory the undertraining diagnostic reads).
EVAL_EVERY = 100
#: Power iterations for the init PCA (``torch.svd_lowrank``); seven land within ~0.2% of
#: the exact rank-k optimum on flat spectra, 16 stays seconds on the fit split.
PCA_NITER = 16
#: Fraction of cached positions held out as the fit-side val set.
VAL_FRACTION = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lens-dir", required=True,
        help="the J source (lens directory or single lens-<mask>.pt)",
    )
    parser.add_argument("--manifest", required=True, help="the FIT split (a corpus manifest)")
    parser.add_argument(
        "--mask", default="text", help="which lens file (and which positions) to translate"
    )
    parser.add_argument("--limit", type=int, default=40, help="fit on only the first N samples")
    parser.add_argument("--rank", type=int, default=32, help="translator rank r (A_l = U V^T)")
    parser.add_argument("--steps", type=int, default=800, help="Adam steps per layer")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument(
        "--seed", type=int, default=0, help="torch+numpy seed, recorded in the report"
    )
    parser.add_argument(
        "--unembed", default=None,
        help="saved W_U [vocab, d] .pt for the fit readout; absent -> the live model's "
        "unembed weight (the tiny fixture's own W_U on the tiny backend)",
    )
    parser.add_argument(
        "--translator-input", choices=("raw", "jacobian"), default="jacobian",
        help="x_l = J_l h_l (default, our class extended) or x_l = h_l (the pure tuned lens)",
    )
    parser.add_argument(
        "--out-root", required=True,
        help="writes OUT/artifacts/, OUT/bias-<mask>.pt and OUT/fit_report.json",
    )
    parser.add_argument(
        "--backend", choices=("hf-llava", "tiny"), default="hf-llava",
        help="model backend: 'hf-llava' (default, CUDA) or the tiny CPU smoke fixture",
    )
    parser.add_argument(
        "--dtype", default="bfloat16", help="torch dtype name for the model weights"
    )
    parser.add_argument("--skip-first", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=1536)
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> LlavaLensModel:
    """The distillation model: the HF checkpoint (default, CUDA) or the tiny CPU fixture."""
    if args.backend == "tiny":
        from vlm_lens.models.tiny_llava import TinyLlavaConfig, build_tiny_llava

        hf_model, processor = build_tiny_llava(TinyLlavaConfig())
        return LlavaLensModel(hf_model, processor)
    return LlavaLensModel.from_pretrained(
        dtype=getattr(torch, args.dtype), device="cuda", local_files_only=True
    )


def seed_everything(seed: int) -> None:
    """Seed torch (+CUDA) and numpy; the seed is recorded in the report and provenance."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_unembed(
    args: argparse.Namespace, model: LlavaLensModel, device: torch.device
) -> torch.Tensor:
    """The fit readout's ``W_U`` as ``[vocab, d]`` float32 on ``device``.

    The ``--unembed`` .pt when given, else the live model's ``unembed_weight()`` (the
    tiny fixture's own ``W_U`` on the tiny backend). A vocabulary mismatch with the live
    model is a hard error: the distillation target and the readout directions the zoo
    applies must share a vocabulary.
    """
    if args.unembed is not None:
        w_u = torch.load(args.unembed, map_location="cpu", weights_only=True)
        if not isinstance(w_u, torch.Tensor) or w_u.dim() != 2:
            raise ValueError(f"{args.unembed} is not a [vocab, d] W_U tensor")
        live_vocab = int(model.unembed_weight().shape[0])
        if int(w_u.shape[0]) != live_vocab:
            raise ValueError(
                f"{args.unembed} has vocab {w_u.shape[0]} but the live model has "
                f"{live_vocab}; the fit target and the zoo readout must share a vocabulary"
            )
    else:
        w_u = model.unembed_weight().detach()
    w_u = w_u.to(torch.float32).to(device)
    if w_u.shape[1] != model.d_model:
        raise ValueError(
            f"W_U is [vocab, {w_u.shape[1]}] but the model d_model is {model.d_model}"
        )
    return w_u


def split_positions(n_pos: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The fit-side split: a seeded permutation, 10% held out as the val set.

    Returns ``(train_idx, val_idx)`` CPU long tensors. The generator is fresh per call
    and main() calls it once, so every layer sees the SAME split.
    """
    if n_pos < 2:
        raise ValueError(f"{n_pos} positions cannot split into train/val")
    order = torch.randperm(n_pos, generator=torch.Generator().manual_seed(seed))
    n_val = max(1, int(round(VAL_FRACTION * n_pos)))
    return order[n_val:], order[:n_val]


@dataclass
class _Cache:
    """Pass-1 cache: per-layer translator inputs plus the shared distillation target."""

    x: dict[int, torch.Tensor]  # layer -> [n_positions, d] float32 on the fit device
    model_log_probs: torch.Tensor  # [n_positions, vocab] float32, log_softmax(z_model)
    targets: torch.Tensor  # [n_positions] long CPU: true next-token ids (reporting only)
    model_top1_agreement: float  # the model's own next-token top-1 rate (the reference)
    n_used: int  # samples that contributed
    n_skipped: int  # samples skipped (no valid scoring positions)


def cache_translator_inputs(
    model: LlavaLensModel,
    lens: JacobianLens,
    samples: list,
    *,
    mask: str,
    translator_input: str,
    skip_first: int,
    max_seq_len: int,
    device: torch.device,
) -> _Cache:
    """Pass 1: one forward pass per sample, translator inputs at every lens layer.

    Positions are exactly the text-tag scoring positions of ``s2_eval.score_table_fast``:
    mask nonzero, a next token exists, and (the default mode, not
    ``include_placeholders``) the next token is not the image placeholder. Per layer the
    cache keeps ``x_l`` ([n_positions, d] float32 on ``device``; one batched matmul per
    (sample, layer)); the final logits ``z_model`` and the true next-token ids (reporting
    only) are shared across layers. Residuals are recorded at the lens layers plus the
    final (the readout's capture pattern: block outputs, each block's pre-norm residual).
    """
    final_layer = model.n_layers - 1
    # The zoo scores the final-layer row as the model's own logits, so no translator is
    # fitted there (calib_readout's convention); a final-layer source passes through.
    fit_layers = [layer for layer in sorted(set(lens.source_layers)) if layer != final_layer]
    if not fit_layers:
        raise ValueError("no lens layers below the final layer to translate")
    if translator_input == "jacobian":
        missing = [layer for layer in fit_layers if layer not in lens.jacobians]
        if missing:
            raise ValueError(
                f"jacobian input needs J at {missing}; the lens has {lens.source_layers}"
            )
    # One J upload per layer: numerically identical to lens.transport (residual @ J_bar.T)
    # but without re-uploading the [d, d] J per (sample, layer).
    j_dev = (
        {
            layer: lens.jacobians[layer].to(device=device, dtype=torch.float32)
            for layer in fit_layers
        }
        if translator_input == "jacobian"
        else {}
    )
    record_at = sorted(set(fit_layers) | {final_layer})
    per_layer: dict[int, list[torch.Tensor]] = {layer: [] for layer in fit_layers}
    z_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    n_used = 0
    n_skipped = 0

    with torch.no_grad():
        for index, sample in enumerate(samples):
            batch = as_batch(model, sample, max_seq_len)
            masks = build_position_masks(
                batch.input_ids, model.image_token_id, skip_first=skip_first, masks=(mask,)
            )
            mask_positions = masks[mask]
            if not bool(mask_positions.any()):
                n_skipped += 1
                continue
            # s2_eval's text-tag scoring positions: mask nonzero, a next token exists,
            # and the next token is not the image placeholder (the V5 default mode).
            input_ids = batch.input_ids[0].detach().cpu()
            seq_len = int(input_ids.numel())
            positions = [
                int(p) for p in mask_positions.nonzero(as_tuple=True)[0] if int(p) + 1 < seq_len
            ]
            positions = [p for p in positions if int(input_ids[p + 1]) != model.image_token_id]
            if not positions:
                n_skipped += 1
                continue
            pos_idx = torch.tensor(positions, dtype=torch.long)
            # The readout's capture pattern: block outputs at the lens layers plus the
            # final layer, each block's pre-norm residual.
            with ActivationRecorder(model.layers, at=record_at) as recorder:
                model.forward_mm(batch)
                activations = {layer: recorder.activations[layer].detach() for layer in record_at}
            final_act = activations[final_layer][0]  # [seq_len, d_model]
            hf = final_act[pos_idx.to(final_act.device)].float()
            z_rows.append(model.unembed(hf).float())  # [n_pos, vocab]; the model's own logits
            target_rows.append(pos_idx)
            for layer in fit_layers:
                act = activations[layer][0]
                h = act[pos_idx.to(act.device)].float()
                x = h @ j_dev[layer].T if translator_input == "jacobian" else h
                per_layer[layer].append(x.to(device=device, dtype=torch.float32))
            n_used += 1
            if (index + 1) % 10 == 0:
                print(f"  cached {index + 1}/{len(samples)} samples", flush=True)

    if n_used == 0:
        raise ValueError(f"no samples with valid {mask!r} scoring positions in the fit manifest")
    cache_x = {layer: torch.cat(rows, dim=0).to(device) for layer, rows in per_layer.items()}
    z_model = torch.cat(z_rows, dim=0).to(device)
    model_log_probs = torch.log_softmax(z_model, dim=-1)
    targets = torch.cat(target_rows, dim=0)  # CPU long, reporting only
    model_top1 = float(
        (z_model.argmax(dim=-1) == targets.to(z_model.device)).float().sum() / z_model.shape[0]
    )
    return _Cache(
        x=cache_x,
        model_log_probs=model_log_probs,
        targets=targets,
        model_top1_agreement=model_top1,
        n_used=n_used,
        n_skipped=n_skipped,
    )


def _kl_mean(
    x: torch.Tensor, logits_fn, log_probs: torch.Tensor, chunk_size: int
) -> float:
    """Mean KL(softmax(z_model) || softmax(lens(x))) over positions, chunked to bound memory.

    ``logits_fn`` maps an ``x`` chunk to lens logits; ``log_probs`` is the model's
    ``log_softmax(z_model)`` (precomputed once, shared by every layer). Eval only.
    """
    total = 0.0
    for start in range(0, int(x.shape[0]), chunk_size):
        target = log_probs[start : start + chunk_size]
        lens_log_probs = torch.log_softmax(logits_fn(x[start : start + chunk_size]), dim=-1)
        total += float(
            torch.nn.functional.kl_div(
                lens_log_probs, target, reduction="none", log_target=True
            ).sum()
        )
    return total / float(x.shape[0])


@dataclass
class _LayerFit:
    """One layer's fitted translator: factorized params plus the KL trajectory."""

    layer: int
    u: torch.Tensor  # [d, r] float32 CPU; A_l = U V^T
    v: torch.Tensor  # [d, r] float32 CPU
    b: torch.Tensor  # [d] float32 CPU
    fro_norm: float  # ||A_l||_F - the tuned-lens v6 undertraining diagnostic
    history: list[dict[str, float | int]]  # {step, train_kl, val_kl} per checkpoint


def fit_layer(
    layer: int,
    x_all: torch.Tensor,
    model_log_probs: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    *,
    final_norm,
    w_u: torch.Tensor,
    rank: int,
    steps: int,
    lr: float,
    chunk_size: int,
) -> _LayerFit:
    """The per-layer seeded Adam distillation of ``A = U V^T`` + ``b`` to the final logits.

    Init: ``A_0 = V_r V_r^T``, the rank-r PCA projector of the fit-split ``x`` (both
    factors start at the top-r right singular vectors, so the initial readout is the
    rank-r truncated ``x`` through the model's unembed - the tuned-lens identity start's
    best rank-r approximation under the data covariance); ``b_0 = 0``. 10% of positions
    are the fit-side val set; the KL is evaluated at the first, every ``EVAL_EVERY``-th,
    and the last step.
    """
    device = x_all.device
    d = int(x_all.shape[1])
    x_val = x_all[val_idx.to(device)]
    x_train = x_all[train_idx.to(device)]
    lp_val = model_log_probs[val_idx.to(model_log_probs.device)]
    lp_train = model_log_probs[train_idx.to(model_log_probs.device)]
    n_train = int(x_train.shape[0])

    r = min(rank, d, n_train)
    _, _, v_r = torch.svd_lowrank(x_train, q=r, niter=PCA_NITER)
    u = v_r.detach().clone().requires_grad_(True)  # [d, r]
    v = v_r.detach().clone().requires_grad_(True)  # [d, r]
    b = torch.zeros(d, device=device, dtype=torch.float32).requires_grad_(True)
    optimizer = torch.optim.Adam([u, v, b], lr=lr)

    def logits(x: torch.Tensor) -> torch.Tensor:
        # unembed(A x + b), the exact zoo readout: A x = U (V^T x) = (x @ V) @ U^T, then
        # the final norm, then W_U - never materializing the [d, d] A during training.
        return final_norm((x @ v) @ u.T + b) @ w_u.T

    history: list[dict[str, float | int]] = []

    def evaluate(step: int) -> None:
        with torch.no_grad():
            train_kl = _kl_mean(x_train, logits, lp_train, chunk_size)
            val_kl = _kl_mean(x_val, logits, lp_val, chunk_size)
        history.append(
            {"step": step, "train_kl": round(train_kl, 6), "val_kl": round(val_kl, 6)}
        )

    evaluate(0)  # the init: the PCA-projector readout before any update
    batch = min(BATCH_SIZE, n_train)
    cursor = n_train  # forces a fresh shuffle on the first step
    perm = torch.empty(0, dtype=torch.long)
    for step in range(1, steps + 1):
        if cursor + batch > n_train:
            perm = torch.randperm(n_train)  # global seeded RNG: the run is deterministic
            cursor = 0
        idx = perm[cursor : cursor + batch].to(device)
        lens_log_probs = torch.log_softmax(logits(x_train[idx]), dim=-1)
        loss = torch.nn.functional.kl_div(
            lens_log_probs, lp_train[idx], reduction="none", log_target=True
        ).sum(dim=-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cursor += batch
        if step % EVAL_EVERY == 0 or step == steps:
            evaluate(step)

    with torch.no_grad():
        fro_norm = float(torch.linalg.matrix_norm((u @ v.T).detach(), ord="fro"))
    return _LayerFit(
        layer=layer,
        u=u.detach().to(torch.float32).cpu(),
        v=v.detach().to(torch.float32).cpu(),
        b=b.detach().to(torch.float32).cpu(),
        fro_norm=fro_norm,
        history=history,
    )


def compose_lens(
    source: JacobianLens, fit_results: dict[int, _LayerFit], *, translator_input: str
) -> dict[int, torch.Tensor]:
    """The composed ``J'_l`` float32 contiguous CPU dict: ``A_l J_l`` (jacobian) or ``A_l`` (raw).

    Only fitted layers are translated; any other source layer (only a final layer can
    occur) passes through unchanged - the zoo's final-layer row reads the model's own
    logits and never consults it.
    """
    composed: dict[int, torch.Tensor] = {}
    for layer, fit in fit_results.items():
        a_l = fit.u @ fit.v.T  # [d, d] float32 CPU
        if translator_input == "jacobian":
            composed[layer] = (a_l @ source.jacobians[layer].to(torch.float32)).contiguous()
        else:
            composed[layer] = a_l.contiguous()
    for layer in source.source_layers:
        if layer not in composed:  # a final-layer source: pass through, never fitted
            composed[layer] = source.jacobians[layer].to(torch.float32).contiguous()
    return composed


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    samples = read_manifest(args.manifest)
    if args.limit:
        samples = samples[: args.limit]
    src_dir = Path(args.lens_dir)
    lenses, src_prov = load_lens_set(src_dir)
    if args.mask not in lenses:
        raise ValueError(f"mask {args.mask!r} not in {src_dir} (found {sorted(lenses)})")
    lens = lenses[args.mask]
    model = load_model(args)
    print(
        f"model ready: layers={model.n_layers} d_model={model.d_model} "
        f"image_seq_length={model.image_seq_length}; fitted J at {sorted(lens.source_layers)}; "
        f"translator input={args.translator_input} rank={args.rank}"
    )
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and model.unembed_weight().is_cuda) else "cpu"
    )
    # The vendored unembed's final norm (a private attr): the only public readout path
    # hard-wires the model's own lm_head, but the fit reads W_U from --unembed.
    final_norm = model._final_norm
    w_u = load_unembed(args, model, device)
    print(f"pass 1: caching translator inputs ({args.translator_input}) on {device}")

    cache = cache_translator_inputs(
        model, lens, samples,
        mask=args.mask, translator_input=args.translator_input,
        skip_first=args.skip_first, max_seq_len=args.max_seq_len, device=device,
    )
    n_pos = int(cache.model_log_probs.shape[0])
    print(
        f"  {cache.n_used} samples used, {cache.n_skipped} skipped, {n_pos} positions, "
        f"model top-1 {cache.model_top1_agreement:.3f}"
    )

    print(
        f"pass 2: fitting rank-{args.rank} translators "
        f"({args.steps} steps, lr={args.lr}, seed={args.seed})"
    )
    train_idx, val_idx = split_positions(n_pos, args.seed)
    n_train, n_val = int(train_idx.numel()), int(val_idx.numel())
    fit_results: dict[int, _LayerFit] = {}
    for layer in sorted(cache.x):
        fit = fit_layer(
            layer, cache.x[layer], cache.model_log_probs, train_idx, val_idx,
            final_norm=final_norm, w_u=w_u, rank=args.rank, steps=args.steps, lr=args.lr,
            chunk_size=BATCH_SIZE,
        )
        fit_results[layer] = fit
        del cache.x[layer]  # the fitted layer's cache is dead weight from here
        best = min(fit.history, key=lambda row: row["val_kl"])
        print(
            f"  L{layer}: val_kl {fit.history[0]['val_kl']:.3f} -> "
            f"{fit.history[-1]['val_kl']:.3f} (best {best['val_kl']:.3f} @ step "
            f"{best['step']}), ||A_l||_F={fit.fro_norm:.3f}",
            flush=True,
        )

    print("pass 3: writing the composed lens + bias payload (CPU float32)")
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    composed = compose_lens(lens, fit_results, translator_input=args.translator_input)
    composed_lens = JacobianLens(
        jacobians=composed, n_prompts=lens.n_prompts, d_model=lens.d_model,
    )
    src_sha = ((src_prov.get("artifacts") or {}).get(args.mask) or {}).get("sha256")
    # run_merge/synth_lenses provenance pattern: the source identity is echoed,
    # created_utc is fresh, and the stale source files-meta is dropped (save_lens_set
    # writes its own artifacts key); everything stays JSON-plain for weights_only.
    provenance = {
        **{key: value for key, value in src_prov.items() if key != "artifacts"},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_prompts": {args.mask: int(lens.n_prompts)},
        "derived_from": {"lens_dir": str(src_dir), "mask": args.mask, "sha256": src_sha},
        "posthoc_distill": {
            "rank": int(args.rank),
            "steps": int(args.steps),
            "lr": float(args.lr),
            "seed": int(args.seed),
            "translator_input": args.translator_input,
            # the FINAL val KL: the number the saved (last-step) weights actually score
            "val_kl_per_layer": {
                str(layer): fit_results[layer].history[-1]["val_kl"] for layer in fit_results
            },
        },
    }
    artifacts_dir = out_root / "artifacts"
    # float32 on disk (save_lens_set defaults to float16): the schema stays
    # upstream-compatible, so the zoo reads it with no special casing.
    save_lens_set(
        artifacts_dir, {args.mask: composed_lens}, provenance=provenance, dtype=torch.float32,
    )
    meta = {
        "lens_dir": str(args.lens_dir),
        "mask": args.mask,
        "manifest": args.manifest,
        "n_samples_used": cache.n_used,
        "n_positions": n_pos,
        "skip_first": args.skip_first,
        "rank": int(args.rank),
        "steps": int(args.steps),
        "lr": float(args.lr),
        "seed": int(args.seed),
        "translator_input": args.translator_input,
        "created_utc": provenance["created_utc"],
        "estimator": ESTIMATOR,
    }
    bias_path = out_root / f"bias-{args.mask}.pt"
    save_bias(
        bias_path,
        {**{str(layer): {"bias": fit.b, "scale": 1.0} for layer, fit in fit_results.items()},
         "meta": meta},
    )

    layer_rows = {
        str(layer): {
            "fit": {
                "first": fit_results[layer].history[0],
                "last": fit_results[layer].history[-1],
                "best": min(fit_results[layer].history, key=lambda row: row["val_kl"]),
            },
            "fro_norm": round(fit_results[layer].fro_norm, 6),
        }
        for layer in fit_results
    }
    report = {
        "lens_dir": str(args.lens_dir),
        "mask": args.mask,
        "manifest": args.manifest,
        "translator_input": args.translator_input,
        "rank": int(args.rank),
        "steps": int(args.steps),
        "lr": float(args.lr),
        "seed": int(args.seed),
        "device": str(device),
        "unembed": args.unembed or "model.unembed_weight()",
        "n_samples_used": cache.n_used,
        "n_samples_skipped": cache.n_skipped,
        "n_positions": n_pos,
        "n_train": n_train,
        "n_val": n_val,
        "model_top1_agreement": cache.model_top1_agreement,
        "layers": layer_rows,
        "meta": meta,
    }
    report_path = out_root / "fit_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"\ntranslator: {cache.n_used} samples used, {cache.n_skipped} skipped, "
        f"{n_pos} positions ({n_train} train / {n_val} val)"
    )
    print("layer   val_kl_first  val_kl_last  val_kl_best  ||A_l||_F")
    for layer in sorted(fit_results):
        fit = fit_results[layer]
        best = min(fit.history, key=lambda row: row["val_kl"])
        print(
            f"{layer:<6}  {fit.history[0]['val_kl']:>12.3f}  "
            f"{fit.history[-1]['val_kl']:>11.3f}  {best['val_kl']:>11.3f}  "
            f"{fit.fro_norm:>10.3f}"
        )
    print(f"\nwrote {artifacts_dir / MASK_FILENAME.format(mask=args.mask)}")
    print(f"wrote {bias_path}")
    print(f"wrote {report_path}")
    print(f"zoo: --lens <name>={artifacts_dir} --bias-dir {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
