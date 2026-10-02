#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""S1 go/no-go: score the text-only lens on a held-out WikiText shard and run the checks.

Reported per layer: rank_true, KL, top1_agree, next to the model ceiling (the model's own
logits), the chance level (uniform) and the plain logit lens (J = I, ``use_jacobian=False``).

Gate (register/task S1): PASS iff
  (i)   J=I reproduces the logit lens exactly,
  (ii)  finite-difference rows of J_l match the estimator at ~1-2 % relative error,
  (iii) J-lens and logit lens agree closely in the last layers,
  (iv)  fidelity improves with depth,
  and J-lens is not worse than the logit lens in the middle layers.
Noise in roughly the first third of layers is expected and is not a failure.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402, I001  # the vlm_lens import must precede jlens: it installs the path
import vlm_lens  # noqa: E402, F401
from jlens.lens import JacobianLens  # noqa: E402

from vlm_lens._batch import as_batch  # noqa: E402
from vlm_lens.artifacts import load_lens_set  # noqa: E402
from vlm_lens.data.manifest import read_manifest  # noqa: E402
from vlm_lens.evaluate import format_scores, frequency_control, score_lens  # noqa: E402
from vlm_lens.fitting import jacobian_for_sample  # noqa: E402
from vlm_lens.models.llava import LlavaLensModel  # noqa: E402
from vlm_lens.positions import build_position_masks  # noqa: E402

SKIP_FIRST = 16  # paper protocol for text-only control fits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lens-dir", required=True, help="run dir containing artifacts/")
    parser.add_argument("--heldout-manifest", required=True)
    parser.add_argument("--fit-manifest", required=True, help="fitting prompts, for the unigram prior")
    parser.add_argument("--json", required=True)
    parser.add_argument("--tag", default="text")
    parser.add_argument("--mask", default="text", help="fitted lens to score")
    parser.add_argument("--fd-layers", default="0,8,16,24,30")
    parser.add_argument("--fd-rows", type=int, default=4)
    parser.add_argument("--fd-samples", type=int, default=1)
    parser.add_argument("--fd-dtype", default="float32", help="FD dtype (bf16 cannot resolve eps=1e-2)")
    parser.add_argument("--fd-device", default="cuda", help="FD device; falls back to cpu on CUDA OOM")
    parser.add_argument(
        "--fd-eps",
        type=float,
        default=1e-2,
        help="absolute FD step; keep small against the model's residual RMS",
    )
    parser.add_argument(
        "--fd-min-free-mib",
        type=int,
        default=32000,
        help="free GPU memory the FD model waits for before falling back to CPU",
    )
    parser.add_argument(
        "--fd-wait-minutes",
        type=int,
        default=60,
        help="how long to wait for that window before the CPU fallback",
    )
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--skip-fd", action="store_true")
    return parser.parse_args()


def jacobian_lens_identity(d_model: int, layers: list[int]) -> JacobianLens:
    eye = torch.eye(d_model)
    return JacobianLens(jacobians={layer: eye.clone() for layer in layers}, n_prompts=1, d_model=d_model)


def finite_difference_check(
    model, samples, *, layers, n_rows, max_seq_len, skip_first=SKIP_FIRST, eps=1e-2
):
    """Check (ii): compare estimator columns against a central-difference of the same functional.

    ``J_l`` is stored output-dim-major (``J_l[j, i] = d h_final[target, j] / d h_l[src, i]``,
    the layout ``unembed(J @ h)`` and the intervention directions need). The estimator's
    entry ``[j, i]`` at layer ``l`` is the mean over source positions ``p`` of
    ``d( sum_{p' in targets} h_final[p', j] ) / d h_l[p, i]``. Perturbing ``h_l[p, i] += eps``
    along one *source* dimension ``i`` at all source positions and differencing the summed
    final residuals therefore reproduces a *column*: ``n_src * J_l[:, i]``. Perturbing
    several source dimensions at once would difference the sum of their columns and cannot
    be compared against one column.

    ``eps`` is an absolute step on the residual: it must stay small against the residual
    RMS of the model under test (the real checkpoint's pre-norm residuals are O(10) at
    layer 0; the tiny CPU fixture's are O(0.01) and needs ~1e-5 for the same relative step).
    """
    layer_cols: dict[str, list[float]] = {}
    stats: dict[str, dict[str, float]] = {}
    for sample in samples:
        batch = as_batch(model, sample, max_seq_len)
        masks = build_position_masks(
            batch.input_ids, model.image_token_id, skip_first=skip_first, masks=("text", "all")
        )
        # The FD perturbation must cover exactly the source set the estimator averages
        # over ("text"); the target sum matches the estimator's target mask ("all").
        source_positions = masks["text"].nonzero(as_tuple=True)[0]
        target_positions = masks["all"].nonzero(as_tuple=True)[0]
        n_src = int(source_positions.numel())
        estimated, _ = jacobian_for_sample(
            model, batch, source_layers=layers, dim_batch=8, skip_first=skip_first, masks=("text",)
        )
        for layer in layers:
            # Highest-norm columns = the most influential source dimensions.
            cols = sorted(
                set(torch.topk(estimated["text"][layer].norm(dim=0), k=n_rows).indices.tolist())
            )
            handle = None

            def perturb(
                module, inputs, output, *, dim=0, positions=source_positions, delta=0.0
            ):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden = hidden.clone()
                hidden[:, positions.to(hidden.device), dim] += delta
                return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

            for dim in cols:
                deltas = {}
                for sign, key in ((1.0, "plus"), (-1.0, "minus")):
                    delta = eps * sign
                    handle = model.layers[layer].register_forward_hook(
                        lambda module, inputs, output, e=delta, d=dim: perturb(
                            module, inputs, output, dim=d, delta=e
                        )
                    )
                    try:
                        with torch.no_grad():
                            # [0]: batch 1, drop the leading axis -> [seq, d_model]
                            residual = model.forward_residual(batch)[0].float()
                    finally:
                        handle.remove()
                    deltas[key] = residual[target_positions.to(residual.device)].sum(dim=0)
                fd_col = (deltas["plus"] - deltas["minus"]) / (2 * eps * n_src)
                reference = estimated["text"][layer][:, dim]
                relative = float((fd_col.cpu() - reference).norm() / (reference.norm() + 1e-30))
                stats[f"L{layer}_col{dim}"] = {
                    "relative_error": round(relative, 5),
                    "n_src": n_src,
                }
                layer_cols.setdefault(f"L{layer}", []).append(round(relative, 5))
    return {"per_col": stats, "per_layer_mean": {k: sum(v) / len(v) for k, v in layer_cols.items()}}



def depth_checks(comparison: list[dict], n_layers: int) -> dict[str, object]:
    """Check (iv): does fidelity improve with depth?

    Gated as a trend - the last layer must beat the depth midpoint - because strict
    per-layer monotonicity wobbles by fractions of a rank on a finite held-out shard and
    would fail the gate spuriously. Monotonicity is still reported (as a diagnostic) so a
    genuinely non-improving lens is visible in the JSON.
    """
    rank_by_layer = {row["layer"]: row["rank_j"] for row in comparison}
    last_layer = max(rank_by_layer)
    mid_layer = min(rank_by_layer, key=lambda layer: abs(layer - n_layers // 2))
    nonmonotone = [
        (comparison[index - 1]["layer"], comparison[index]["layer"])
        for index in range(1, len(comparison))
        if comparison[index]["layer"] >= mid_layer
        and comparison[index]["rank_j"] > comparison[index - 1]["rank_j"]
    ]
    return {
        "iv_rank_improves_with_depth": rank_by_layer[last_layer] < rank_by_layer[mid_layer],
        "iv_rank_last": rank_by_layer[last_layer],
        "iv_rank_mid": rank_by_layer[mid_layer],
        "iv_rank_monotone_last_half": not nonmonotone,
        "iv_n_nonmonotone_steps_last_half": len(nonmonotone),
    }


def main() -> int:
    args = parse_args()
    heldout = read_manifest(args.heldout_manifest)
    fit_texts = [sample.text for sample in read_manifest(args.fit_manifest)]
    lenses, provenance = load_lens_set(args.lens_dir)
    if args.mask not in lenses:
        raise SystemExit(f"lens {args.mask!r} not in {sorted(lenses)}")
    lens = lenses[args.mask]
    model = LlavaLensModel.from_pretrained(dtype=torch.bfloat16, device="cuda", local_files_only=True)
    layers = sorted(lens.source_layers)
    report: dict[str, object] = {
        "lens_dir": args.lens_dir,
        "mask": args.mask,
        "n_prompts": int(lens.n_prompts),
        "layers": layers,
        "heldout_manifest": args.heldout_manifest,
        "n_heldout": len(heldout),
        "skip_first": SKIP_FIRST,
        "vocab_size": int(model.unembed_weight().shape[0]),
        "chance_mean_rank": (int(model.unembed_weight().shape[0]) + 1) / 2,
        "uniform_mean_kl": math.log(int(model.unembed_weight().shape[0])),
        "fit_provenance": {
            key: provenance.get(key)
            for key in ("created_utc", "estimator", "corpus", "mask_positions", "fit_config")
        },
    }

    # (a) rank/KL/agreement vs the model ceiling.
    scores = score_lens(model, lens, heldout, tags=(args.tag,), skip_first=SKIP_FIRST)
    report["j_lens"] = [score.to_json() for score in scores]
    print("== J-lens (held-out) ==")
    print(format_scores(scores))

    # (b) logit-lens baseline and check (i): J=I must reproduce it exactly.
    logit_scores = score_lens(
        model, lens, heldout, tags=(args.tag,), skip_first=SKIP_FIRST, use_jacobian=False
    )
    identity = jacobian_lens_identity(model.d_model, layers)
    identity_scores = score_lens(
        model, identity, heldout, tags=(args.tag,), skip_first=SKIP_FIRST, layers=layers
    )
    report["logit_lens"] = [score.to_json() for score in logit_scores]
    by_key = {(row.layer, row.tag): row for row in logit_scores}
    identity_delta = 0.0
    for row in identity_scores:
        reference = by_key[(row.layer, row.tag)]
        identity_delta = max(
            identity_delta,
            abs(row.mean_rank_true - reference.mean_rank_true),
            abs(row.mean_kl - reference.mean_kl),
            abs(row.top1_agreement - reference.top1_agreement),
        )
    report["check_i_identity_max_abs_diff"] = identity_delta
    print("\n== logit lens (J = I) ==")
    print(format_scores(logit_scores))

    # Per-layer comparison, J-lens vs logit lens.
    comparison = []
    for row in scores:
        base = by_key[(row.layer, row.tag)]
        comparison.append(
            {
                "layer": row.layer,
                "rank_j": round(row.mean_rank_true, 2),
                "rank_logit": round(base.mean_rank_true, 2),
                "kl_j": round(row.mean_kl, 4),
                "kl_logit": round(base.mean_kl, 4),
                "agree_j": round(row.top1_agreement, 4),
                "agree_logit": round(base.top1_agreement, 4),
                "j_better_rank": row.mean_rank_true < base.mean_rank_true,
            }
        )
    report["comparison"] = comparison

    # (iii) last-layer agreement between J-lens and logit lens; (iv) depth trend.
    final_j = next(row for row in scores if row.layer == max(row2.layer for row2 in scores))
    middle = [entry for entry in comparison if entry["layer"] < model.n_layers - 1]
    report["checks"] = {
        "iii_last_layer_rank_diff": round(
            abs(final_j.mean_rank_true - by_key[(final_j.layer, final_j.tag)].mean_rank_true), 3
        ),
        **depth_checks(comparison, model.n_layers),
        "j_not_worse_in_middle": all(entry["rank_j"] <= entry["rank_logit"] for entry in middle),
        "n_middle_layers_worse": sum(
            1 for entry in middle if entry["rank_j"] > entry["rank_logit"]
        ),
    }

    # (v) frequency control against the fitting corpus' unigram prior.
    control = frequency_control(
        model,
        lens,
        heldout,
        fit_texts,
        layers=[0, model.n_layers // 2, model.n_layers - 1],
        tags=(args.tag,),
        skip_first=SKIP_FIRST,
        top_k=50,
    )
    report["frequency_control"] = [row.to_json() for row in control]
    print("\n== frequency control (top-50 unigram) ==")
    for row in control:
        print(
            f"L{row.layer:<3} n={row.n:<6} true_in_top50={row.true_in_top_k:.3f} "
            f"lens_top1_in_top50={row.lens_top1_in_top_k:.3f} model_top1_in_top50={row.model_top1_in_top_k:.3f} "
            f"unigram_rank={row.unigram_mean_rank_true:.0f} lens_rank={row.lens_mean_rank_true:.0f} "
            f"model_rank={row.model_mean_rank_true:.0f} unigram_agree={row.unigram_top1_agreement:.3f}"
        )
    # Snapshot the scoring rows before the FD model loads: the FD pass is the last step and
    # the only one that can still be killed (OOM/SIGKILL) after nearly all the work is done,
    # and the gate must not lose the fidelity rows to an environmental failure. The final
    # write below overwrites this with the complete report.
    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")


    # Free the bf16 scoring model before the FD model loads: they would otherwise coexist on
    # the GPU (15 + 29 GiB) and every FD check would fall back to CPU.
    del model
    gc.collect()
    torch.cuda.empty_cache()

    if not args.skip_fd:
        print("\n== finite-difference check (ii) ==")
        fd_dtype = torch.float32 if args.fd_dtype == "float32" else torch.bfloat16
        fd_layers = [int(part) for part in args.fd_layers.split(",") if part.strip()]
        # The fp32 FD model (~29 GiB) needs a real window on this shared GPU. Waiting beats
        # the CPU fallback: the estimator's ceil(d_model/dim_batch) chunk-backwards on a 7B
        # fp32 CPU model take hours, not the minutes a forward-only check would (measured
        # 2026-10-02: the CPU path was entered under a 90 MiB window and had to be killed).
        fd_device = args.fd_device
        if fd_device == "cuda":
            waited = 0
            while True:
                free_mib = torch.cuda.mem_get_info()[0] // (1024 * 1024)
                if free_mib >= args.fd_min_free_mib:
                    break
                if waited >= args.fd_wait_minutes * 60:
                    print(
                        f"no {args.fd_min_free_mib} MiB window after {waited // 60} min "
                        f"(free={free_mib} MiB) - falling back to CPU"
                    )
                    fd_device = "cpu"
                    break
                print(
                    f"waiting for GPU memory: free={free_mib} MiB < "
                    f"{args.fd_min_free_mib} MiB ({waited // 60}/{args.fd_wait_minutes} min)"
                )
                time.sleep(30)
                waited += 30
        try:
            fd_model = LlavaLensModel.from_pretrained(
                dtype=fd_dtype, device=fd_device, local_files_only=True
            )
            fd = finite_difference_check(
                fd_model,
                heldout[: args.fd_samples],
                layers=fd_layers,
                n_rows=args.fd_rows,
                max_seq_len=args.max_seq_len,
                eps=args.fd_eps,
            )
        except torch.OutOfMemoryError:
            print(f"CUDA OOM for the {args.fd_dtype} FD model - retrying on CPU")
            torch.cuda.empty_cache()
            fd_device = "cpu"
            fd_model = LlavaLensModel.from_pretrained(
                dtype=fd_dtype, device="cpu", local_files_only=True
            )
            fd = finite_difference_check(
                fd_model,
                heldout[: args.fd_samples],
                layers=fd_layers,
                n_rows=args.fd_rows,
                max_seq_len=args.max_seq_len,
                eps=args.fd_eps,
            )
        report["check_ii_finite_difference"] = fd
        report["check_ii_device"] = fd_device
        for layer, value in fd["per_layer_mean"].items():
            print(f"{layer}: mean relative error over rows = {value:.4f}")

    gate = {
        "i_identity_exact": report["check_i_identity_max_abs_diff"] == 0.0,
        "ii_finite_difference_ok": (
            None
            if "check_ii_finite_difference" not in report
            else max(report["check_ii_finite_difference"]["per_layer_mean"].values()) <= 0.05
        ),
        "iii_last_layers_agree": report["checks"]["iii_last_layer_rank_diff"] <= 2.0,
        "iv_depth_improves": report["checks"]["iv_rank_improves_with_depth"],
        "j_not_worse_in_middle": report["checks"]["j_not_worse_in_middle"],
    }
    # A None criterion means "not measured", not "failed": with --skip-fd the row (ii) is
    # re-run out-of-band (D20), and a gate that silently read that as False would misreport it.
    unmeasured = sorted(key for key, value in gate.items() if value is None)
    gate["not_measured"] = unmeasured
    gate["PASS"] = all(value for value in gate.values() if value is not None)
    gate["PASS_complete"] = not unmeasured and gate["PASS"]
    report["gate"] = gate
    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\ngate: {json.dumps(gate)}")
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
