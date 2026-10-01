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
import json
import math
import sys
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
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--skip-fd", action="store_true")
    return parser.parse_args()


def jacobian_lens_identity(d_model: int, layers: list[int]) -> JacobianLens:
    eye = torch.eye(d_model)
    return JacobianLens(jacobians={layer: eye.clone() for layer in layers}, n_prompts=1, d_model=d_model)


def finite_difference_check(model, samples, *, layers, n_rows, max_seq_len, skip_first=SKIP_FIRST):
    """Check (ii): compare estimator rows against a central-difference of the same functional.

    The estimator's row ``i`` at layer ``l`` is
    ``(1/n_src) * d( sum_{p' in targets} h_final[p', j] ) / d h_l[p, i]`` summed over sources.
    Perturbing ``h_l[p, i] += eps`` at *all* source positions and differencing the summed
    final residuals reproduces ``n_src * J_l[i, :]``.
    """
    layer_rows: dict[str, list[float]] = {}
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
            rows = sorted(
                set(torch.topk(estimated["text"][layer].norm(dim=1), k=n_rows).indices.tolist())
            )
            handle = None

            def perturb(
                module, inputs, output, *, dims=rows, positions=source_positions, delta=0.0
            ):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden = hidden.clone()
                hidden[:, positions.to(hidden.device), dims] += delta
                return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

            for dim in rows:
                deltas = {}
                for sign, key in ((1.0, "plus"), (-1.0, "minus")):
                    eps = 1e-2 * sign
                    handle = model.layers[layer].register_forward_hook(
                        lambda module, inputs, output, s=sign, e=eps: perturb(
                            module, inputs, output, delta=e
                        )
                    )
                    try:
                        with torch.no_grad():
                            residual = model.forward_residual(batch)[0].float()
                    finally:
                        handle.remove()
                    deltas[key] = residual[target_positions.to(residual.device)].sum(dim=0)
                fd_row = (deltas["plus"] - deltas["minus"]) / (2 * 1e-2 * n_src)
                reference = estimated["text"][layer][dim]
                relative = float((fd_row.cpu() - reference).norm() / (reference.norm() + 1e-30))
                stats[f"L{layer}_row{dim}"] = {
                    "relative_error": round(relative, 5),
                    "n_src": n_src,
                }
                layer_rows.setdefault(f"L{layer}", []).append(round(relative, 5))
    return {"per_row": stats, "per_layer_mean": {k: sum(v) / len(v) for k, v in layer_rows.items()}}


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

    # (iii) last-layer agreement between J-lens and logit lens; (iv) depth improvement.
    final_j = next(row for row in scores if row.layer == max(row2.layer for row2 in scores))
    middle = [entry for entry in comparison if entry["layer"] < model.n_layers - 1]
    report["checks"] = {
        "iii_last_layer_rank_diff": round(
            abs(final_j.mean_rank_true - by_key[(final_j.layer, final_j.tag)].mean_rank_true), 3
        ),
        "iv_rank_monotone_last_half": all(
            comparison[index]["rank_j"] <= comparison[index - 1]["rank_j"]
            for index in range(1, len(comparison))
            if comparison[index]["layer"] >= model.n_layers // 2
        ),
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

    if not args.skip_fd:
        print("\n== finite-difference check (ii) ==")
        fd_dtype = torch.float32 if args.fd_dtype == "float32" else torch.bfloat16
        fd_layers = [int(part) for part in args.fd_layers.split(",") if part.strip()]
        # The fp32 FD model (~29 GiB) does not fit while the box's co-tenant holds most of
        # the H100. Falling back to CPU keeps the gate honest instead of failing it on
        # environmental OOM: 1 sample, 4 rows, 5 layers - minutes on the 263 GiB host.
        fd_device = args.fd_device
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
            )
        report["check_ii_finite_difference"] = fd
        report["check_ii_device"] = fd_device
        for layer, value in fd["per_layer_mean"].items():
            print(f"{layer}: mean relative error over rows = {value:.4f}")

    gate = {
        "i_identity_exact": report["check_i_identity_max_abs_diff"] == 0.0,
        "ii_finite_difference_ok": (
            "check_ii_finite_difference" in report
            and max(report["check_ii_finite_difference"]["per_layer_mean"].values()) <= 0.05
        ),
        "iii_last_layers_agree": report["checks"]["iii_last_layer_rank_diff"] <= 2.0,
        "iv_depth_improves": report["checks"]["iv_rank_monotone_last_half"],
        "j_not_worse_in_middle": report["checks"]["j_not_worse_in_middle"],
    }
    gate["PASS"] = all(gate.values())
    report["gate"] = gate
    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\ngate: {json.dumps(gate)}")
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
