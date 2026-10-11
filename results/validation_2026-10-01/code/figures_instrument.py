#!/usr/bin/env python
"""
figures_instrument.py - instrument-family figures (figA1..figA5) for the
2026-10-01 validation round (fitting precision, depth fidelity, masks/census,
model-class ladder, data scaling).

Deterministic: fixed palette, fixed layout, no timestamps inside figures;
re-running regenerates byte-identical PNGs.

Outputs (relative to this script's parent dir, results/validation_2026-10-01/):
  figures/figA1_dtype_precision.png
  figures/figA2_depth_fidelity.png
  figures/figA3_census_masks.png
  figures/figA4_model_class_ladder.png
  figures/figA5_data_scaling.png
  summary/metrics_instrument.json

Data sources:
  raw/step1/x6_dtype.json   (X6: bf16-vs-fp32 rel. Frobenius by layer x mask)
  raw/step2/s1_score.json   (S1: J-lens vs logit-lens mean rank + KL by layer)
  raw/step1/x3_norms.json   (X3: residual-norm census by position group x layer)
  DECISIONS.md D41/D42/D43  (ladder + scaling numbers; no lens_zoo JSON under raw/)

Run: <python> results/validation_2026-10-01/code/figures_instrument.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "raw"
FIGS = BASE / "figures"
SUMMARY = BASE / "summary"
FIGS.mkdir(parents=True, exist_ok=True)
SUMMARY.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colorblind-safe palette
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
BLUE = "#0072B2"
VERMIL = "#D55E00"
PURPLE = "#CC79A7"
GREY = "0.35"

MASK_COLOR = {"all": BLUE, "image": GREEN, "text": VERMIL}

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.framealpha": 0.9,
        "legend.fontsize": 9,
    }
)

SAVE_KW = dict(bbox_inches="tight", metadata={"Software": "figures_instrument.py"})


def load(name):
    with open(RAW / name) as f:
        return json.load(f)


# ---------------------------------------------------------------- figA1: X6
def figA1():
    x6 = load("step1/x6_dtype.json")
    layers = sorted(int(k) for k in x6["masks"]["all"]["per_layer"])
    masks = ["all", "image", "text"]
    med = {m: x6["masks"][m]["median_rel"] * 100 for m in masks}

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for m in masks:
        ys = [x6["masks"][m]["per_layer"][str(L)]["rel_frobenius"] * 100 for L in layers]
        ax.plot(layers, ys, "o-", lw=1.8, ms=5, color=MASK_COLOR[m],
                label=f"{m} mask (median {med[m]:.2f}%)")

    # X6 verdict rule thresholds
    ax.axhline(10, ls="--", lw=1, color=GREY)
    ax.text(30.2, 10, "max rule 10%", va="center", fontsize=8.5, color=GREY)
    ax.axhline(3, ls=":", lw=1, color=GREY)
    ax.text(30.2, 3, "median rule 3%", va="center", fontsize=8.5, color=GREY)

    # worst cell: text / L0
    worst = x6["masks"]["text"]["per_layer"]["0"]["rel_frobenius"] * 100
    ax.annotate(f"worst cell: text/L0 = {worst:.1f}%",
                xy=(0, worst), xytext=(3.2, 33),
                arrowprops=dict(arrowstyle="->", color=VERMIL), color=VERMIL,
                fontsize=9.5, fontweight="bold")
    ax.annotate("per-mask medians:\nall 1.20% | image 1.26% | text 1.48%",
                xy=(0.03, 0.05), xycoords="axes fraction", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.35", fc="0.95", ec="0.7"))

    ax.set_yscale("log")
    ax.set_xticks(layers)
    ax.set_xlim(-1.2, 35.5)
    ax.set_ylim(0.12, 90)
    ax.set_xlabel("layer")
    ax.set_ylabel("bf16 rel. Frobenius error vs fp32 (%)")
    ax.set_title("X6 fitting precision: bf16 error stays <1.5% by L16 but hits 49.2% at text/L0\n"
                 "verdict (X6): production = fp32 + TF32")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGS / "figA1_dtype_precision.png", **SAVE_KW)
    plt.close(fig)

    return {
        "text_L0_rel_frobenius_pct": round(worst, 3),
        "all_L0_rel_frobenius_pct": round(x6["masks"]["all"]["per_layer"]["0"]["rel_frobenius"] * 100, 3),
        "median_rel_pct": {m: round(med[m], 3) for m in masks},
        "max_rel_pct": {m: round(x6["masks"][m]["max_rel"] * 100, 3) for m in masks},
        "rule": x6["verdict"]["rule"],
        "use_fp32": x6["verdict"]["use_fp32"],
    }


# ---------------------------------------------------------------- figA2: S1
def figA2():
    s1 = load("step2/s1_score.json")
    j = {e["layer"]: e for e in s1["j_lens"]}
    ll = {e["layer"]: e for e in s1["logit_lens"]}
    Ls = sorted(j)
    jr = [j[L]["mean_rank_true"] for L in Ls]
    lr = [ll[L]["mean_rank_true"] for L in Ls]
    chance = s1["chance_mean_rank"]
    model_rank = j[31]["model_mean_rank_true"]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                  gridspec_kw=dict(height_ratios=[1.35, 1]))

    for a in (ax, ax2):
        a.axvspan(16.5, 24.5, color="0.88", alpha=0.55, zorder=0)

    ax.plot(Ls, jr, "o-", lw=1.8, ms=4.5, color=BLUE, label="J-lens (fitted)")
    ax.plot(Ls, lr, "s-", lw=1.8, ms=4.5, color=ORANGE, label="logit lens (untrained)")
    ax.axhline(chance, ls=":", lw=1, color=GREY)
    ax.text(0.2, chance * 1.12, f"chance = {chance:,.1f}", fontsize=8.5, color=GREY)
    ax.axhline(model_rank, ls="--", lw=1, color=GREEN)
    ax.text(0.2, model_rank * 1.35, f"model itself = {model_rank:.2f}", fontsize=8.5, color=GREEN)
    ax.annotate("L31: exact - rank 27.31 = model,\nKL = 0, top-1 agreement = 1.00",
                xy=(31, model_rank), xytext=(21.5, 260),
                arrowprops=dict(arrowstyle="->", color=GREEN), color=GREEN, fontsize=9)
    ax.text(20.5, 9000, "L17-24 mid-layer:\nJ trails the logit lens\n(A2 averaged-Jacobian limit)",
            ha="center", fontsize=9, color="0.25",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9))
    ax.set_yscale("log")
    ax.set_ylim(12, 45000)
    ax.set_ylabel("mean rank of true token (lower = better)")
    ax.set_title("S1 depth fidelity (30 held-out prompts, 6,930 text positions):\n"
                 "J-lens beats the untrained logit lens at every layer except the L17-24 bump - and is exact at L31")
    ax.legend(loc="lower left")

    ax2.plot(Ls, [j[L]["mean_kl"] for L in Ls], "o-", lw=1.8, ms=4.5, color=BLUE, label="J-lens")
    ax2.plot(Ls, [ll[L]["mean_kl"] for L in Ls], "s-", lw=1.8, ms=4.5, color=ORANGE, label="logit lens")
    ax2.axhline(s1["uniform_mean_kl"], ls=":", lw=1, color=GREY)
    ax2.text(0.2, s1["uniform_mean_kl"] + 0.15, f"uniform KL = {s1['uniform_mean_kl']:.2f}",
             fontsize=8.5, color=GREY)
    ax2.set_ylabel("mean KL (nats)")
    ax2.set_xlabel("layer")
    ax2.set_title("mean KL by layer (mid-layer band shaded)", fontsize=10)
    ax2.legend(loc="lower left")

    ax.set_xticks(np.arange(0, 31, 5))
    ax.set_xlim(-1.5, 32.5)
    fig.tight_layout()
    fig.savefig(FIGS / "figA2_depth_fidelity.png", **SAVE_KW)
    plt.close(fig)

    mid = list(range(17, 25))
    jmr = float(np.mean([j[L]["mean_rank_true"] for L in mid]))
    lmr = float(np.mean([ll[L]["mean_rank_true"] for L in mid]))
    return {
        "j_lens_L0_mean_rank": round(j[0]["mean_rank_true"], 1),
        "logit_lens_L0_mean_rank": round(ll[0]["mean_rank_true"], 1),
        "j_lens_L16_mean_rank": round(j[16]["mean_rank_true"], 1),
        "logit_lens_L16_mean_rank": round(ll[16]["mean_rank_true"], 1),
        "mid_L17_24_mean_rank_j_vs_lens": [round(jmr, 1), round(lmr, 1)],
        "mid_layers_j_worse_count": int(sum(j[L]["mean_rank_true"] > ll[L]["mean_rank_true"] for L in mid)),
        "L30_mean_rank_j_vs_lens": [round(j[30]["mean_rank_true"], 1), round(ll[30]["mean_rank_true"], 1)],
        "L31_exact_rank_kl_top1": [round(model_rank, 2), round(j[31]["mean_kl"], 3), round(j[31]["top1_agreement"], 3)],
        "chance_mean_rank": chance,
    }


# ---------------------------------------------------------------- figA3: X3
def figA3():
    x3 = load("step1/x3_norms.json")
    lay = sorted(x3["layers"], key=int)
    groups = list(x3["layers"]["0"].keys())
    pretty = {"bos": "BOS", "pos_1_16": "pos 1-16", "text_pre": "text (pre)",
              "patch_1_16": "patch 1-16", "patch_17_end": "patch 17-end",
              "text_post": "text (post)"}
    lay_color = {"0": ORANGE, "16": BLUE, "24": GREEN, "31": PURPLE}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.6), width_ratios=[2.1, 1])

    x = np.arange(len(groups))
    w = 0.2
    for i, L in enumerate(lay):
        vals = [x3["layers"][L][g]["mean_norm"] for g in groups]
        ax.bar(x + (i - 1.5) * w, vals, w, label=f"L{L}", color=lay_color[L])
    ax.set_yscale("log")
    ax.set_ylim(1.2, 5000)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty[g] for g in groups])
    ax.set_ylabel("mean residual norm")
    ax.legend(title="layer", ncol=2)
    bos16 = x3["layers"]["16"]["bos"]["mean_norm"]
    bos0 = x3["layers"]["0"]["bos"]["mean_norm"]
    ax.annotate(f"BOS sink:\n{bos0:.1f} at L0 -> {bos16:,.1f} at L16 ({bos16 / bos0:.0f}x)",
                xy=(0 + 1.5 * w, bos16), xytext=(1.15, 700),
                arrowprops=dict(arrowstyle="->", color=GREY), fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="0.95", ec="0.7"))
    ax.set_title("residual norm by position group x layer (log scale):\nBOS is the outlier - smallest at L0, ~42x text norms by L16")

    # right panel: per-mask position counts from the same 50-sample census
    ps = x3["per_sample"]
    n = x3["n_samples"]
    all_pos = sum(s["seq_len"] for s in ps)
    img_pos = sum(s["n_image_tokens"] for s in ps)
    txt_pos = sum(s["seq_len"] - 1 - s["n_image_tokens"] for s in ps)
    bos_pos = n
    counts = [txt_pos, img_pos, all_pos]
    bars = ax2.bar(["text", "image", "all"], counts,
                   color=[MASK_COLOR["text"], MASK_COLOR["image"], MASK_COLOR["all"]])
    for b, c in zip(bars, counts, strict=False):
        ax2.text(b.get_x() + b.get_width() / 2, c * 1.01, f"{c:,}",
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, all_pos * 1.18)
    ax2.set_ylabel("positions (50-sample census)")
    ax2.set_title("positions per mask:\nimage patches are 88.7% of the sequence")
    ax2.text(0.02, 0.95, f"BOS = {bos_pos} (1/sample)\ntext = seq_len - 1 - 576 patches",
             transform=ax2.transAxes, va="top", fontsize=8.5, color="0.25")

    fig.tight_layout()
    fig.savefig(FIGS / "figA3_census_masks.png", **SAVE_KW)
    plt.close(fig)

    return {
        "n_samples": n,
        "bos_mean_norm": {L: x3["layers"][L]["bos"]["mean_norm"] for L in lay},
        "bos_growth_L0_to_L16_x": round(bos16 / bos0, 1),
        "bos_over_text_post_at_L16_x": round(bos16 / x3["layers"]["16"]["text_post"]["mean_norm"], 1),
        "text_post_mean_norm_L16": x3["layers"]["16"]["text_post"]["mean_norm"],
        "positions_per_mask": {"text": txt_pos, "image": img_pos, "all": all_pos, "bos": bos_pos},
        "image_position_share_pct": round(100 * img_pos / all_pos, 1),
    }


# ---------------------------------------------------------------- figA4: ladder
def figA4():
    # LQS ladder: DECISIONS.md D41 table (lines 859-871); best J readout from D42.
    rows = [
        # (label, LQS, class, cite)
        ("rank-64 truncated J", -0.587, "j", "D41"),
        ("rank-256 truncated J", -0.284, "j", "D41"),
        ("full J, uncorrected", -0.102, "j", "D41"),
        ("alpha-I (scaled identity, no bias)", 0.000, "logit", "D41"),
        ("untrained logit lens (baseline)", 0.000, "logit", "D41"),
        ("diag(J) only, no bias", 0.374, "j", "D41"),
        ("shift-P4 (J_l+4 on h_l, no bias)", 0.524, "j", "D41"),
        ("ident + bias (calibrated logit lens)", 0.647, "logit", "D41"),
        ("full J + census bias+scale (D40)", 0.766, "j", "D41"),
        ("full J + bias + logit_bias (merged, P5e)", 0.922, "j", "D41"),
        ("best J readout: tmtext50 + bias+scale+logit_bias", 0.947, "j", "D42"),
        ("lowrankJ (J + rank-32 KL translator)", 1.714, "tr", "D41"),
        ("lowrankRaw (pure tuned lens, rank-32)", 1.718, "tr", "D41"),
    ]
    cls_color = {"logit": SKY, "j": BLUE, "tr": VERMIL}
    cls_label = {"logit": "logit-lens class (no transport)",
                 "j": "J-lens class (Jacobian transport)",
                 "tr": "translator class (rank-32 KL)"}

    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    y = np.arange(len(rows))
    for i, (name, v, c, _cite) in enumerate(rows):
        best = name.startswith("best J")
        ax.barh(i, v, color=cls_color[c], edgecolor="black" if best else "none",
                linewidth=1.8 if best else 0)
        if name.startswith("best J"):
            ax.text(v + 0.05, i, "best J readout +0.947\n(DECISIONS.md D42)",
                    va="center", ha="left", fontsize=9.5, fontweight="bold", color=BLUE)
        elif v >= 0:
            ax.text(v + 0.03, i, f"{v:+.3f}", va="center", ha="left", fontsize=8.5)
        else:
            ax.text(v - 0.03, i, f"{v:+.3f}", va="center", ha="right", fontsize=8.5)
    ax.axvline(0, color="0.2", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(-1.05, 2.25)
    ax.set_xticks(np.arange(-1.0, 2.01, 0.5))
    ax.set_xlabel("LQS (composite lens quality score, 300-sample held-out, tag=text; higher = better)")

    ax.text(0.975, 0.03,
            "affine does early work, transport does late work (D41):\n"
            "bias+scale lifts LQS -0.102 -> +0.766; transport adds +0.119 (16% of the fix);\n"
            "L30 rank ratio: J 1.30-1.79 vs translator 5.62-5.82 (D42/D43)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.8,
            bbox=dict(boxstyle="round,pad=0.4", fc="0.95", ec="0.7"))

    handles = [plt.Rectangle((0, 0), 1, 1, color=cls_color[c]) for c in ("logit", "j", "tr")]
    ax.legend(handles, [cls_label[c] for c in ("logit", "j", "tr")], loc="center right",
              title="lens class")
    ax.set_title("Model-class ladder (D41/D42): translators lead at LQS 1.71-1.72,\n"
                 "best J readout +0.947 - the J's value is the late-layer transport")
    fig.tight_layout()
    fig.savefig(FIGS / "figA4_model_class_ladder.png", **SAVE_KW)
    plt.close(fig)

    return {
        "untrained_logit_lens_LQS": 0.000,
        "full_J_uncorrected_LQS": -0.102,
        "full_J_census_bias_scale_LQS": 0.766,
        "full_J_bias_logit_bias_merged_LQS": 0.922,
        "best_J_readout_LQS": 0.947,
        "lowrankJ_LQS": 1.714,
        "lowrankRaw_LQS": 1.718,
        "transport_gain_over_ident_bias_LQS": 0.119,
        "L30_rank_ratio_J_vs_translator": "1.30-1.79 vs 5.62-5.82",
        "citation": "DECISIONS.md D41 (table, lines 859-871) and D42; no lens_zoo JSON under raw/",
    }


# ---------------------------------------------------------------- figA5: scaling
def figA5():
    # D43: translator LQS vs n; J-class flat band 0.92-0.95.
    ns = np.array([40, 100, 500, 1000])
    lqs = np.array([1.714, 2.043, 2.691, 3.009])
    deltas = np.diff(lqs)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.axhspan(0.92, 0.95, color=GREEN, alpha=0.25, zorder=0)
    ax.axhline(0.935, color=GREEN, lw=1.2, ls="--")
    ax.text(42, 0.965, "J-class flat band 0.92-0.95 (saturates by n~100)",
            fontsize=9, color=GREEN)

    ax.plot(ns, lqs, "o-", lw=2, ms=6, color=BLUE, label="rank-32 KL translator (held-out LQS)")
    for n, v in zip(ns, lqs, strict=False):
        ax.annotate(f"{v:.3f}", xy=(n, v), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=9, color=BLUE)
    for n0, v0, n1, v1, d in zip(ns[:-1], lqs[:-1], ns[1:], lqs[1:], deltas, strict=False):
        ax.annotate(f"+{d:.2f}", xy=(np.sqrt(n0 * n1), (v0 + v1) / 2),
                    xytext=(-4, -14), textcoords="offset points", fontsize=8.5, color="0.3")
    ax.text(0.03, 0.86,
            "not saturated at n=1000 - recommend n=2000-5000 (D43);\n"
            "held-out LQS rises ~linearly in log n (~0.4 per e-fold) while\n"
            "fit-side KL decouples: 1.424 (n=40) -> 0.888 (n=1000)",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="0.95", ec="0.7"))

    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlim(33, 1300)
    ax.set_ylim(0.5, 3.5)
    ax.set_xlabel("translator fit samples n (log scale)")
    ax.set_ylabel("held-out LQS (higher = better)")
    ax.set_title("J saturates at ~100; translator does not (D43 data scaling)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "figA5_data_scaling.png", **SAVE_KW)
    plt.close(fig)

    return {
        "translator_LQS_by_n": {str(n): float(v) for n, v in zip(ns, lqs, strict=False)},
        "rung_deltas": [round(float(d), 3) for d in deltas],
        "lqs_per_log_fold_approx": 0.4,
        "J_class_flat_band": [0.92, 0.95],
        "fit_side_val_kl_40_to_1000": [1.424, 0.888],
        "citation": "DECISIONS.md D43; no step8/lens_zoo_scaling.json under raw/",
    }


def main():
    metrics = {
        "round": "validation_2026-10-01",
        "family": "instrument (fitting precision, depth fidelity, masks/census, "
                  "model-class ladder, data scaling)",
        "script": "results/validation_2026-10-01/code/figures_instrument.py",
        "figures": [
            {
                "file": "figA1_dtype_precision.png",
                "sources": ["results/validation_2026-10-01/raw/step1/x6_dtype.json"],
                "key_numbers": figA1(),
                "finding": "bf16-vs-fp32 relative error falls with depth to 1.2-1.5% "
                           "at the per-mask medians by L16, but the text mask shows 49.2% "
                           "relative error at L0 - X6 verdict: production = fp32 + TF32.",
            },
            {
                "file": "figA2_depth_fidelity.png",
                "sources": ["results/validation_2026-10-01/raw/step2/s1_score.json"],
                "key_numbers": figA2(),
                "gap": "S2 depth-curve JSON absent from raw/ (raw/step3/ holds only fit "
                       "manifests); figure uses S1 only, as instructed.",
                "finding": "The J-lens beats the untrained logit lens on mean rank at every "
                           "layer except the L17-24 mid-layer bump (worse at 5/8 of those "
                           "layers; mean rank 2228.8 vs 2119.3), and is exact at L31 "
                           "(rank 27.31 = the model's own, KL = 0, top-1 = 1.00).",
            },
            {
                "file": "figA3_census_masks.png",
                "sources": ["results/validation_2026-10-01/raw/step1/x3_norms.json"],
                "key_numbers": figA3(),
                "finding": "Position census (50 samples): image patches are 88.7% of the "
                           "32,454 positions, and the BOS sink is the outlier - the smallest "
                           "group norm at L0 (8.3) but 1,569 by L16 (188x), ~42x the text "
                           "norms, decaying to 634 by L31.",
            },
            {
                "file": "figA4_model_class_ladder.png",
                "sources": [
                    "results/validation_2026-10-01/DECISIONS.md D41 (ladder table, lines 859-871)",
                    "results/validation_2026-10-01/DECISIONS.md D42 (best J readout +0.947)",
                ],
                "key_numbers": figA4(),
                "finding": "Affine correction does the early work, transport the late work: "
                           "bias+scale lifts LQS from -0.102 to +0.766, logit_bias to +0.922, "
                           "and the tmtext50 refit to +0.947; rank-32 translators lead overall "
                           "(1.714-1.718) but are 3x worse at L30 (5.62-5.82 vs 1.30-1.79).",
            },
            {
                "file": "figA5_data_scaling.png",
                "sources": ["results/validation_2026-10-01/DECISIONS.md D43"],
                "key_numbers": figA5(),
                "finding": "The J-class is data-saturated (flat LQS 0.92-0.95 by n~100) while "
                           "the translator's held-out LQS keeps rising ~linearly in log n: "
                           "1.714@40 -> 2.043@100 -> 2.691@500 -> 3.009@1000 - the production "
                           "recommendation flips to the translator (n>=1000, keep scaling).",
            },
        ],
        "notes": [
            "All JSON-sourced numbers are read from the raw files at run time; ladder and "
            "scaling numbers exist only in DECISIONS.md (no step4/lens_zoo_*.json or "
            "step8/lens_zoo_scaling.json under raw/), so they are cited as DECISIONS.md D41/D42/D43.",
            "figA2 S2-depth-curve gap: no step3/step4 result JSON under raw/; S1 used alone.",
            "figA3 right-panel mask counts are derived from x3_norms.json per_sample "
            "(text = seq_len - 1 - 576 image patches; BOS = 1 per sample).",
        ],
    }
    out = SUMMARY / "metrics_instrument.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"wrote {out}")
    for entry in metrics["figures"]:
        print(f"  {entry['file']}")


if __name__ == "__main__":
    main()
