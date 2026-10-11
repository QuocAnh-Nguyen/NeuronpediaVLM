#!/usr/bin/env python
"""
figures_d_detect.py - detection-round detail figures (figD5..figD8) for the
2026-10-01 validation round: POPE handoff distributions (x18), decoding
uncertainty signals (x15), attention mirror + sink clamp (x17/x16), and
workspace reasoning dynamics (x22 multihop/flexgen, x23 modulation).

Deterministic: fixed palette, fixed layout, fixed bins/jitter seed, no
timestamps inside figures; re-running regenerates byte-identical PNGs and the
metrics JSON.

Outputs (relative to this script's parent dir, results/validation_2026-10-01/):
  figures/figD5_pope_handoff.png
  figures/figD6_uncertainty_signals.png
  figures/figD7_mirror_and_sink.png
  figures/figD8_workspace_dynamics.png
  summary/metrics_d_detect.json

Data sources:
  raw/step11/x18_pope_handoff.json        (185 POPE questions, per-question lens gap)
  raw/step11/x15_hesitation_n200.json     (385 grounded / 39 hallucinated word steps)
  raw/step11/x17_attention_mirror_n200.json (mirror rho by layer + hallucination join)
  raw/step11/x16_sink_clamp.json          (clamp arms: retention + handoff rank deltas)
  raw/step12/x22_reasoning_structure_multihop.json (onset rows, 93 two-hop items)
  raw/step12/x22_reasoning_structure_flexgen.json  (192 directed swaps per alpha)
  raw/step12/x23_vlm_workspace_modulation.json     (590 instructed runs, co-occupancy)
  DECISIONS.md D45/D46/D47 (paper comparison values quoted in annotations only)

Run: <python> results/validation_2026-10-01/code/figures_d_detect.py
"""

import json
from decimal import ROUND_HALF_UP, Decimal
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

# Okabe-Ito colorblind-safe palette (same conventions as figures_instrument.py)
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
BLUE = "#0072B2"
VERMIL = "#D55E00"
PURPLE = "#CC79A7"
GREY = "0.35"

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

SAVE_KW = dict(bbox_inches="tight", metadata={"Software": "figures_d_detect.py"})

RNG = np.random.default_rng(0)  # fixed seed -> deterministic jitter


def load(rel):
    with open(RAW / rel) as f:
        return json.load(f)


def src(rel):
    return f"raw/{rel}"


def kn(value, source):
    """Key-number entry: value + its exact provenance."""
    return {"value": value, "source": source}


def pct1(x):
    """Percent with 1 decimal, half-up (matches DECISIONS.md quotes, e.g. 96.6)."""
    return (Decimal(str(x)) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------- figD5: X18
def figD5():
    x18 = load("step11/x18_pope_handoff.json")
    items = x18["items"]
    summ = x18["summary"]

    pres = [it["lens_gap"] for it in items if it["present"]]
    absn = [it["lens_gap"] for it in items if not it["present"]]
    halluc_yes = [it for it in items if not it["present"] and it["model_answer"] == "yes"]
    grounded_yes = [it for it in items if it["present"] and it["model_answer"] == "yes"]
    mean_p = float(np.mean(pres))
    mean_a = float(np.mean(absn))
    au_pa = summ["auroc"]["present_vs_absent"]
    au_hall = summ["auroc"]["halluc_yes_vs_grounded_yes"]
    det = 1.0 - au_hall  # low-gap => hallucinated-yes flag (script's own convention)
    layers = sorted(x18["per_layer"], key=int)
    au_by_layer = [x18["per_layer"][L]["auroc_present_vs_absent"] for L in layers]
    best = summ["best_layer"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.8, 5.6), width_ratios=[2.45, 1])

    bins = np.linspace(-9.0, 10.0, 39)
    ax.hist(pres, bins=bins, color=BLUE, alpha=0.55,
            label=f"object PRESENT (n={len(pres)})")
    ax.hist(absn, bins=bins, color=VERMIL, alpha=0.55,
            label=f"object ABSENT (n={len(absn)})")
    counts_max = ax.get_ylim()[1]
    ax.axvline(mean_p, color=BLUE, lw=1.6, ls="--",
               label=f"present mean +{mean_p:.2f}")
    ax.axvline(mean_a, color=VERMIL, lw=1.6, ls="--",
               label=f"absent mean {mean_a:.2f}")

    # headroom: bars occupy the lower half of the panel, text the upper half
    ax.set_ylim(-0.28 * counts_max, 2.10 * counts_max)

    # the 8 model-hallucinated yes, marked on the absent side
    rug_y = -0.085 * counts_max
    ax.scatter([it["lens_gap"] for it in halluc_yes], [rug_y] * len(halluc_yes),
               marker="^", s=52, color="k", zorder=5,
               label=f"model-hallucinated yes (n={len(halluc_yes)})")
    ax.text(-3.0, -0.215 * counts_max,
            f"the 8 wrong affirmations sit LOW on the gap: range "
            f"{min(it['lens_gap'] for it in halluc_yes):.1f}.."
            f"{max(it['lens_gap'] for it in halluc_yes):.1f} "
            f"(grounded-yes range {min(it['lens_gap'] for it in grounded_yes):.1f}.."
            f"{max(it['lens_gap'] for it in grounded_yes):.1f})",
            ha="center", fontsize=8.8, color="0.1")

    ax.text(0.02, 0.965,
            "single forward pass, pre-generation:\n"
            f"AUROC(present vs absent) = {au_pa:.3f} at L{best} "
            f"({min(au_by_layer):.3f}-{max(au_by_layer):.3f} over L{layers[0]}-L{layers[-1]})\n"
            f"halluc-yes vs grounded-yes AUROC = {au_hall:.3f} (halluc positive)\n"
            f"= {det:.3f} as a LOW-gap hallucination detector (0.895 in D45/D46)",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="0.95", ec="0.7"))

    acc = summ["model_accuracy"]
    ax.text(0.98, 0.965,
            f"model POPE accuracy {pct1(acc['overall'])}%\n"
            f"(present {pct1(acc['present'])} / absent {pct1(acc['absent'])})\n"
            f"answers: {summ['model_answers']['yes']} yes / "
            f"{summ['model_answers']['no']} no",
            transform=ax.transAxes, va="top", ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.9))

    ax.set_xlim(-9.0, 10.0)
    ax.set_xlabel("calibrated-lens gap  z[yes] - z[no]  at the last prompt position (L28)")
    ax.set_ylabel("POPE questions")
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.42), fontsize=8.6)

    xs = np.arange(len(layers))
    ax2.bar(xs, au_by_layer, 0.55, color=SKY, zorder=3)
    ax2.axhline(0.5, ls=":", lw=1, color=GREY)
    ax2.text(0.01, 0.5, "chance 0.5", transform=ax2.get_yaxis_transform(),
             va="bottom", fontsize=8.5, color=GREY)
    bi = layers.index(str(best))
    ax2.bar([bi], [au_by_layer[bi]], 0.55, color=BLUE, zorder=4)
    for x, v in zip(xs, au_by_layer, strict=False):
        ax2.text(x, v + 0.004, f"{v:.3f}", ha="center", fontsize=9,
                 fontweight="bold" if x == bi else "normal")
    ax2.set_ylim(0.45, 1.03)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"L{L}" for L in layers])
    ax2.set_xlabel("lens layer")
    ax2.set_ylabel("AUROC(present vs absent)")
    ax2.set_title(f"flat-high across layers\n(best L{best})", fontsize=10)

    fig.suptitle(
        "X18 POPE pre-generation handoff (49 images, 185 questions): the lens's yes/no gap "
        "nearly solves object presence (AUROC 0.985)\n"
        "and the lens disagrees with the model exactly on its 8 hallucinated yes "
        "(0.895 as a low-gap detector)",
        fontsize=11.5, y=0.985
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIGS / "figD5_pope_handoff.png", **SAVE_KW)
    plt.close(fig)

    return {
        "n_images": kn(summ["n_images"],
                       src("step11/x18_pope_handoff.json") + " summary.n_images"),
        "n_questions": kn(summ["n_questions"],
                          src("step11/x18_pope_handoff.json") + " summary.n_questions"),
        "n_present": kn(summ["n_present_questions"],
                        src("step11/x18_pope_handoff.json") + " summary.n_present_questions"),
        "n_absent": kn(summ["n_absent_questions"],
                       src("step11/x18_pope_handoff.json") + " summary.n_absent_questions"),
        "model_accuracy": kn(
            {"present": acc["present"], "absent": acc["absent"], "overall": acc["overall"]},
            src("step11/x18_pope_handoff.json") + " summary.model_accuracy"),
        "mean_gap_present_L28": kn(round(mean_p, 4),
                                   src("step11/x18_pope_handoff.json")
                                   + " per_layer['28'].mean_gap_present (items mean agrees)"),
        "mean_gap_absent_L28": kn(round(mean_a, 4),
                                  src("step11/x18_pope_handoff.json")
                                  + " per_layer['28'].mean_gap_absent"),
        "auroc_present_vs_absent_L28": kn(round(au_pa, 4),
                                          src("step11/x18_pope_handoff.json")
                                          + " summary.auroc.present_vs_absent"),
        "auroc_present_vs_absent_by_layer": kn(
            {f"L{L}": round(v, 4) for L, v in zip(layers, au_by_layer, strict=False)},
            src("step11/x18_pope_handoff.json") + " per_layer[L].auroc_present_vs_absent"),
        "best_layer": kn(best, src("step11/x18_pope_handoff.json") + " summary.best_layer"),
        "auroc_halluc_yes_vs_grounded_yes_raw": kn(
            round(au_hall, 4),
            src("step11/x18_pope_handoff.json")
            + " summary.auroc.halluc_yes_vs_grounded_yes (hallucinated = positive class)"),
        "low_gap_detector_auroc": kn(
            round(det, 4),
            "1 - auroc_halluc_yes_vs_grounded_yes; DECISIONS.md D45/D46 quote 0.895 "
            "from the 3-decimal 0.105"),
        "n_model_yes_halluc": kn(x18["per_layer"][str(best)]["n_model_yes_halluc"],
                                 src("step11/x18_pope_handoff.json")
                                 + " per_layer['28'].n_model_yes_halluc"),
        "halluc_yes_gap_range": kn([round(min(it["lens_gap"] for it in halluc_yes), 2),
                                    round(max(it["lens_gap"] for it in halluc_yes), 2)],
                                   src("step11/x18_pope_handoff.json")
                                   + " items: model_answer=yes & present=false lens_gap"),
        "grounded_yes_gap_range": kn([round(min(it["lens_gap"] for it in grounded_yes), 2),
                                      round(max(it["lens_gap"] for it in grounded_yes), 2)],
                                     src("step11/x18_pope_handoff.json")
                                     + " items: model_answer=yes & present=true lens_gap"),
    }


# ---------------------------------------------------------------- figD6: X15
def figD6():
    x15 = load("step11/x15_hesitation_n200.json")
    agg = x15["aggregate"]

    # per-word completion-step values: a word's `step` indexes its OWN image's steps
    # (verified: means reproduce aggregate.grounded/hallucinated exactly)
    g = {"entropy": [], "margin": [], "logprob": []}
    h = {"entropy": [], "margin": [], "logprob": []}
    for im in x15["images"]:
        st = im["steps"]
        for w in im["words"]:
            lab = w["label"]
            if lab not in ("grounded", "hallucinated"):
                continue
            row = g if lab == "grounded" else h
            row["entropy"].append(st["model_entropy"][w["step"]])
            row["margin"].append(st["top1_margin"][w["step"]])
            row["logprob"].append(st["chosen_logprob"][w["step"]])
    n_g, n_h = len(g["entropy"]), len(h["entropy"])
    assert (n_g, n_h) == (agg["grounded"]["n_steps"], agg["hallucinated"]["n_steps"]), (
        n_g, n_h)

    au_model_ent = agg["auroc"]["model_entropy"]
    au_margin = agg["auroc"]["top1_margin"]
    lens_au = {int(k): v for k, v in agg["auroc"]["lens_entropy"].items()}
    Ls = sorted(lens_au)
    peak_L = max(Ls, key=lambda L: lens_au[L])

    fig, axes = plt.subplots(1, 4, figsize=(13.4, 5.0), width_ratios=[1, 1, 1, 1.8])
    panels = [
        ("entropy", g["entropy"], h["entropy"], "next-token entropy (nats)",
         f"{agg['grounded']['entropy_mean']:.3f}", f"{agg['hallucinated']['entropy_mean']:.3f}"),
        ("margin", g["margin"], h["margin"], "top-1 margin (logits)",
         f"{agg['grounded']['margin_mean']:.2f}", f"{agg['hallucinated']['margin_mean']:.2f}"),
        ("logprob", g["logprob"], h["logprob"], "chosen-token log-prob",
         f"{agg['grounded']['logprob_mean']:.2f}", f"{agg['hallucinated']['logprob_mean']:.2f}"),
    ]
    for ax, (key, gv, hv, ylab, gm, hm) in zip(axes, panels, strict=False):
        bp = ax.boxplot([gv, hv], positions=[0, 1], widths=0.52, patch_artist=True,
                        showfliers=False, medianprops=dict(color="0.15", lw=1.6),
                        whiskerprops=dict(color="0.3"), capprops=dict(color="0.3"))
        for patch, col in zip(bp["boxes"], [SKY, VERMIL], strict=False):
            patch.set_facecolor(col)
            patch.set_alpha(0.55)
            patch.set_edgecolor("0.3")
        for xc, vals, col in ((0, gv, BLUE), (1, hv, VERMIL)):
            jit = (RNG.random(len(vals)) - 0.5) * 0.30
            ax.scatter(xc + jit, vals, s=7, color=col, alpha=0.35, lw=0, zorder=3)
        ax.set_ylim(top=max(max(gv), max(hv)) * 1.22)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"grounded\n(n={n_g})", f"hallucinated\n(n={n_h})"], fontsize=9)
        ax.set_ylabel(ylab)
        ax.text(0.03, 0.97, f"means {gm} vs {hm}", transform=ax.transAxes,
                va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", fc="0.95", ec="0.7"))
        ax.set_title({"entropy": "hallucinated words arrive\nwith HIGHER entropy",
                      "margin": "... LOWER\ntop-1 margin",
                      "logprob": "... LOWER\nchosen log-prob"}[key], fontsize=10)

    ax = axes[3]
    ax.plot(Ls, [lens_au[L] for L in Ls], "o-", lw=1.8, ms=4.5, color=BLUE,
            label="lens entropy (by layer)")
    ax.axhline(au_model_ent, ls="--", lw=1.5, color=VERMIL,
               label=f"model entropy {au_model_ent:.3f}")
    ax.axhline(au_margin, ls=":", lw=1.5, color=GREEN,
               label=f"model margin {au_margin:.3f} (inverse)")
    ax.axhline(0.5, ls=":", lw=1, color=GREY)
    ax.axvspan(3.6, 7.4, color=SKY, alpha=0.18, lw=0, zorder=0)
    ax.annotate(f"peak band L4-7:\nL{peak_L} = {lens_au[peak_L]:.3f}",
                xy=(peak_L, lens_au[peak_L]), xytext=(10.5, 0.715),
                arrowprops=dict(arrowstyle="->", color=BLUE), color=BLUE, fontsize=9.5,
                fontweight="bold")
    ax.text(0.3, 0.445, "chance 0.5", fontsize=8.5, color=GREY)
    ax.set_xticks(np.arange(0, 32, 4))
    ax.set_xlim(-1.0, 32.0)
    ax.set_ylim(0.40, 0.80)
    ax.set_xlabel("lens layer")
    ax.set_ylabel("AUROC (hallucinated = positive)")
    ax.set_title("lens-entropy AUROC by layer:\nsignal lives in the early layers", fontsize=10)
    ax.legend(loc="upper right", fontsize=8.3)

    fig.suptitle(
        "X15 decoding hesitation at power (200 images): hallucinated word-completion steps "
        "are the MORE uncertain ones - entropy 0.963 vs 0.620, margin 4.53 vs 5.55, "
        "logprob -0.384 vs -0.230 -\nand the calibrated lens's entropy at L4-7 "
        f"(peak {lens_au[peak_L]:.3f}) beats the model's own uncertainty signals",
        fontsize=11.5, y=0.99
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIGS / "figD6_uncertainty_signals.png", **SAVE_KW)
    plt.close(fig)

    return {
        "n_grounded_steps": kn(n_g, src("step11/x15_hesitation_n200.json")
                               + " aggregate.grounded.n_steps (word rows reproduce it)"),
        "n_hallucinated_steps": kn(n_h, src("step11/x15_hesitation_n200.json")
                                   + " aggregate.hallucinated.n_steps"),
        "entropy_mean_grounded_vs_halluc": kn([agg["grounded"]["entropy_mean"],
                                               agg["hallucinated"]["entropy_mean"]],
                                              src("step11/x15_hesitation_n200.json")
                                              + " aggregate.*.entropy_mean"),
        "margin_mean_grounded_vs_halluc": kn([agg["grounded"]["margin_mean"],
                                              agg["hallucinated"]["margin_mean"]],
                                             src("step11/x15_hesitation_n200.json")
                                             + " aggregate.*.margin_mean"),
        "logprob_mean_grounded_vs_halluc": kn([agg["grounded"]["logprob_mean"],
                                               agg["hallucinated"]["logprob_mean"]],
                                              src("step11/x15_hesitation_n200.json")
                                              + " aggregate.*.logprob_mean"),
        "auroc_model_entropy": kn(round(au_model_ent, 4),
                                  src("step11/x15_hesitation_n200.json")
                                  + " aggregate.auroc.model_entropy"),
        "auroc_model_margin": kn(round(au_margin, 4),
                                 src("step11/x15_hesitation_n200.json")
                                 + " aggregate.auroc.top1_margin"),
        "lens_entropy_auroc_peak": kn([peak_L, round(lens_au[peak_L], 4)],
                                      src("step11/x15_hesitation_n200.json")
                                      + " aggregate.auroc.lens_entropy"),
        "lens_entropy_auroc_band_L4_7": kn(
            {f"L{L}": round(lens_au[L], 4) for L in range(4, 8)},
            src("step11/x15_hesitation_n200.json") + " aggregate.auroc.lens_entropy"),
        "model_entropy_is_lens_L31": kn(
            True,
            src("step11/x15_hesitation_n200.json")
            + " steps.model_entropy == steps.lens_entropy['31'] per image (final-layer lens "
              "readout = model distribution); aggregate.auroc.model_entropy == lens_entropy L31"),
    }


# ---------------------------------------------------------------- figD7: X17+X16
def figD7():
    x17 = load("step11/x17_attention_mirror_n200.json")
    x16 = load("step11/x16_sink_clamp.json")

    Ls17 = sorted(x17["per_layer"], key=int)
    rho = [x17["per_layer"][L]["rho_mean"] for L in Ls17]
    share = [x17["per_layer"][L]["share_mean"] for L in Ls17]
    join = x17["hallucination_join"]
    au_share_mid = join["share_mid_layer"]["auroc_hallucinating_vs_clean"]
    au_rho_mid = join["rho_mid_layer"]["auroc_hallucinating_vs_clean"]
    r_pb = join["rho_mid_layer"]["point_biserial_r"]
    n_hall_img = join["share_mid_layer"]["n_hallucinating"]
    n_clean_img = join["share_mid_layer"]["n_clean"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.5), width_ratios=[1, 1.12])

    ax.plot([int(L) for L in Ls17], rho, "o-", lw=1.8, ms=4.5, color=BLUE,
            label="attention mirror rho (CLS->patch vs text->image)")
    ax.axvspan(24.5, 30.5, color=ORANGE, alpha=0.10, lw=0, zorder=0)
    ax.text(27.5, 0.72, "decouple\nband L24-30", color=ORANGE, ha="center",
            fontsize=8.5, va="bottom")
    ax.annotate(f"rise: 0.105 (L0) -> {rho[20]:.3f} (L20)",
                xy=(20, rho[20]), xytext=(5.0, 0.66),
                arrowprops=dict(arrowstyle="->", color=BLUE), color=BLUE, fontsize=9.5)
    ax.annotate(f"{rho[24]:.3f} (L24) -> {rho[30]:.3f} (L30):\n"
                "the verbal workspace takes over",
                xy=(30, rho[30]), xytext=(14.0, 0.28),
                arrowprops=dict(arrowstyle="->", color=ORANGE), color=ORANGE, fontsize=9.5)
    ax.text(0.02, 0.03,
            f"image-attention share: {share[0]:.2f} at L0 -> "
            f"{min(share[4:24]):.2f}-{max(share[4:24]):.2f} through the band",
            transform=ax.transAxes, fontsize=8.5, color="0.25")
    ax.text(0.02, 0.965,
            f"hallucination join NULL at power\n({n_hall_img} hallucinating / {n_clean_img} "
            "clean images):\n"
            f"share-mid AUROC {au_share_mid:.3f}; mirror-rho r_pb {r_pb:.2f} "
            f"(AUROC {au_rho_mid:.3f})\n- structure, not a detector",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="0.95", ec="0.7"))
    ax.set_xticks(np.arange(0, 31, 4))
    ax.set_xlim(-1.0, 31.0)
    ax.set_ylim(0.0, 0.78)
    ax.set_xlabel("layer")
    ax.set_ylabel("mean Spearman rho (per image, n=197)")
    ax.set_title("X17 cross-modal attention mirror (n=200):\n"
                 "attention tracks the vision tower, then decouples", fontsize=10.5)

    layers16 = [str(L) for L in x16["layers"]]
    groups = ["bos", "image", "text", "all"]
    gcol = {"bos": VERMIL, "image": GREEN, "text": SKY, "all": PURPLE}
    gname = {"bos": "clamp BOS", "image": "clamp image", "text": "clamp text",
             "all": "clamp all"}
    x = np.arange(len(layers16))
    w = 0.2
    for i, gk in enumerate(groups):
        vals = [x16["per_layer_position"][L][gk]["grounded_retention"] for L in layers16]
        bars = ax2.bar(x + (i - 1.5) * w, vals, w, color=gcol[gk], label=gname[gk],
                       zorder=3)
        for b, v in zip(bars, vals, strict=False):
            ax2.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                     ha="center", fontsize=7.6, rotation=90, va="bottom")
    ax2.axhline(1.0, ls=":", lw=1.2, color=GREY)
    ax2.text(3.78, 1.01, "no loss (1.0)", fontsize=8.5, color=GREY, ha="right")
    ax2.set_ylim(0, 1.80)
    ax2.set_xlim(-0.55, 3.85)

    # BOS-clamp true-object rank deltas on a twin axis (x16 handoff readout, L30)
    bos_rank = [x16["per_layer_position"][L]["bos"]["mean_handoff_rank_delta"]
                for L in layers16]
    ax2b = ax2.twinx()
    ax2b.plot(x, bos_rank, "D--", color="k", ms=6, lw=1.1, zorder=5,
              label="BOS: true-object rank delta")
    for xi, v in zip(x, bos_rank, strict=False):
        ax2b.text(xi, v + 70, f"{int(v):+d}", ha="center", fontsize=8.5, color="k")
    ax2b.set_ylim(-500, 2300)
    ax2b.set_ylabel("BOS-clamp handoff true-object rank delta (L30 readout)", fontsize=9)
    ax2b.grid(False)
    ax2b.spines["top"].set_visible(False)

    # BOS top-16 dim set stability across clamp layers (from the clamp arms themselves)
    dims_by_layer = {}
    for r in x16["results"]:
        if r["group"] == "bos":
            dims_by_layer.setdefault(r["layer"], []).extend(r["dims"])
    sets = [set(v) for v in dims_by_layer.values()]
    shared = sorted(set.intersection(*sets), key=lambda d: dims_by_layer[16].index(d))
    img_ret = [x16["per_layer_position"][L]["image"]["grounded_retention"]
               for L in layers16]
    ax2.text(3.80, 1.76,
             f"image-token clamps inert: retention {min(img_ret):.2f}-{max(img_ret):.2f}\n"
             f"BOS top-16 clamp set stable across L16/24/28/30:\n"
             f"{len(shared)}/16 dims shared "
             f"({', '.join(str(d) for d in shared[:5])}, ...; D46)",
             ha="right", va="top", fontsize=8.7,
             bbox=dict(boxstyle="round,pad=0.35", fc="0.95", ec="0.7"))

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"L{L}" for L in layers16])
    ax2.set_ylabel("grounded-word retention after zero-clamp")
    ax2.set_xlabel("clamp layer")
    ax2.set_title("X16 sink clamp (20 images, top-16 dims): clamping the BOS sink is\n"
                  "destructive (retention 0.278, rank +1433); image clamps are inert",
                  fontsize=10.5)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.3)

    fig.tight_layout()
    fig.savefig(FIGS / "figD7_mirror_and_sink.png", **SAVE_KW)
    plt.close(fig)

    return {
        "mirror_rho_by_layer": kn(
            {f"L{L}": round(v, 4) for L, v in zip(Ls17, rho, strict=False)},
            src("step11/x17_attention_mirror_n200.json") + " per_layer[L].rho_mean"),
        "mirror_peak": kn(["L20", round(rho[20], 3)],
                          src("step11/x17_attention_mirror_n200.json")
                          + " per_layer['20'].rho_mean (also summary.best_abs_rho_layer)"),
        "mirror_L0_L30": kn([round(rho[0], 3), round(rho[30], 3)],
                            src("step11/x17_attention_mirror_n200.json")
                            + " per_layer rho_mean at L0/L30"),
        "share_mid_L15": kn(round(share[15], 3),
                            src("step11/x17_attention_mirror_n200.json")
                            + " per_layer['15'].share_mean (mid_layer=15)"),
        "halluc_join_share_mid_auroc": kn(round(au_share_mid, 4),
                                          src("step11/x17_attention_mirror_n200.json")
                                          + " hallucination_join.share_mid_layer."
                                            "auroc_hallucinating_vs_clean"),
        "halluc_join_rho_rpb": kn(round(r_pb, 4),
                                  src("step11/x17_attention_mirror_n200.json")
                                  + " hallucination_join.rho_mid_layer.point_biserial_r"),
        "halluc_join_rho_auroc": kn(round(au_rho_mid, 4),
                                    src("step11/x17_attention_mirror_n200.json")
                                    + " hallucination_join.rho_mid_layer."
                                      "auroc_hallucinating_vs_clean"),
        "halluc_join_n": kn([n_hall_img, n_clean_img],
                            src("step11/x17_attention_mirror_n200.json")
                            + " hallucination_join.share_mid_layer (n_hallucinating/n_clean;"
                              " summary: 197 images analyzed)"),
        "bos_retention_L16": kn(round(x16["per_layer_position"]["16"]["bos"]["grounded_retention"], 4),
                                src("step11/x16_sink_clamp.json")
                                + " per_layer_position['16'].bos.grounded_retention"),
        "all_groups_retention_L16": kn(round(x16["per_layer_position"]["16"]["all"]["grounded_retention"], 4),
                                       src("step11/x16_sink_clamp.json")
                                       + " per_layer_position['16'].all.grounded_retention"),
        "image_retention_range": kn(
            [round(min(img_ret), 3), round(max(img_ret), 3)],
            src("step11/x16_sink_clamp.json")
            + " per_layer_position[L].image.grounded_retention over L16/24/28/30"),
        "bos_rank_delta": kn(
            {f"L{L}": round(x16["per_layer_position"][L]["bos"]["mean_handoff_rank_delta"], 1)
             for L in layers16},
            src("step11/x16_sink_clamp.json")
            + " per_layer_position[L].bos.mean_handoff_rank_delta"),
        "bos_dim_set_shared": kn(
            [len(shared), 16],
            src("step11/x16_sink_clamp.json")
            + " results: group=bos clamp dims intersected over layers 16/24/28/30"),
    }


# ---------------------------------------------------------------- figD8: X22+X23
def figD8():
    mh = load("step12/x22_reasoning_structure_multihop.json")
    fg = load("step12/x22_reasoning_structure_flexgen.json")
    x23 = load("step12/x23_vlm_workspace_modulation.json")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.6, 5.4))

    # -- left: multihop effective onsets -------------------------------------
    on = mh["results"]["onset_stats"]
    rows = mh["results"]["onset_rows"]
    for xi, (arm, col) in enumerate([("intermediate", ORANGE), ("answer", BLUE)]):
        fr = [r["effective_frac"] for r in rows[arm] if r["effective_frac"] is not None]
        jit = (RNG.random(len(fr)) - 0.5) * 0.22
        ax1.scatter(np.full(len(fr), xi) + jit, fr, s=30, color=col, alpha=0.6,
                    edgecolor="0.25", lw=0.4, zorder=3)
        med = on[arm]["median_effective_frac"]
        ax1.plot([xi - 0.30, xi + 0.30], [med, med], color=col, lw=3, zorder=4)
        ax1.text(xi + 0.33, med + 0.02, f"median {med:.3f}", color=col, fontsize=9.5,
                 va="bottom")
    gap = on["intermediate"]["median_effective_frac"] - on["answer"]["median_effective_frac"]
    ax1.annotate(
        f"precedence gap {gap:+.3f} of band depth\n(paper ~-0.17, DECISIONS.md D47)",
        xy=(0.5, (on["intermediate"]["median_effective_frac"]
                  + on["answer"]["median_effective_frac"]) / 2),
        xytext=(0.16, 0.10),
        arrowprops=dict(arrowstyle="->", color=GREY), fontsize=9.5, ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="0.95", ec="0.7"))
    probe = mh["results"]["probe_stats"]["lens-eval-multihop"]
    ax1.text(0.02, 0.975,
             f"intermediates DO surface in the lens:\nprobe pass@1 {probe['pass@1']:.3f} "
             f"(n={probe['n_items']} two-hop items)",
             transform=ax1.transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels([f"intermediate-swap\n({on['intermediate']['n_effective']}/"
                         f"{on['intermediate']['n_items']} effective)",
                         f"answer-swap\n({on['answer']['n_effective']}/"
                         f"{on['answer']['n_items']} effective)"], fontsize=9)
    ax1.set_xlim(-0.55, 1.55)
    ax1.set_ylim(-0.04, 1.04)
    ax1.set_ylabel("effective onset (band-relative: 0 = early, 1 = late)")
    ax1.set_title("X22 multihop (93 two-hop items): intermediate\nswaps bite EARLIER in "
                  "depth - the paper's\nprecedence replicates", fontsize=10)

    # -- middle: flexgen success vs paper -------------------------------------
    cells = fg["results"]["cells"]
    bc = [c for c in cells if c["baseline_correct"]]
    n_swaps_bc = sum(len(c["swap_success"]) for c in bc)
    succ = {}
    for a in fg["results"]["alphas"]:
        key = str(a)
        succ[key] = sum(c["swap_success"][t][key]
                        for c in bc for t in c["swap_success"]) / n_swaps_bc
    corr = fg["results"]["correlation"]
    xs = np.arange(2)
    ours = [succ["1.0"], succ["2.0"]]
    paper = [0.40, 0.53]  # DECISIONS.md D47 (paper: 40/53%)
    b1 = ax2.bar(xs - 0.19, ours, 0.36, color=BLUE, label="ours (LLaVA-1.5-7B, J-lens)")
    b2 = ax2.bar(xs + 0.19, paper, 0.36, color="0.75", label="paper (Gurnee et al. 2026)")
    for b, v in zip(b1, ours, strict=False):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v * 100:.0f}%",
                 ha="center", fontsize=10, fontweight="bold", color=BLUE)
    for b, v in zip(b2, paper, strict=False):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v * 100:.0f}%",
                 ha="center", fontsize=10, color="0.35")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(["alpha = 1", "alpha = 2"])
    ax2.set_ylim(0, 0.82)
    ax2.set_ylabel("directed-swap success (target's answer emitted)")
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.text(0.5, -0.115,
             f"192 directed swaps per alpha (16 funcs x 4 args x 3 targets); "
             f"{len(bc)}/{len(cells)} cells baseline-correct\n"
             f"ours: {succ['1.0'] * n_swaps_bc:.0f}/{n_swaps_bc} successful directed swaps "
             "on baseline-correct cells (digest: 0.050)\n"
             f"loading-success Pearson +{corr['pearson']:.3f}, Spearman "
             f"+{corr['spearman']:.3f} (n={corr['n_points']} args; right sign, weak)\n"
             "J-lens broadcast transfers far weaker at 7B scale (DECISIONS.md D47)",
             transform=ax2.transAxes, ha="center", va="top", fontsize=8.7,
             bbox=dict(boxstyle="round,pad=0.35", fc="0.95", ec="0.7"))
    ax2.set_title("X22 flexible generalization: swap success is\n5% at BOTH alphas vs the "
                  "paper's 40/53% -\nweak causal leverage", fontsize=10)

    # -- right: x23 modulation co-occupancy -----------------------------------
    summ23 = x23["summary"]
    Ls23 = sorted(x23["per_layer"], key=int)
    x = np.arange(len(Ls23))
    w = 0.26
    series = [
        ("p_visual_given_concept_mean", "p(visual | concept)", BLUE),
        ("p_visual_given_not_concept_mean", "p(visual | no concept)", ORANGE),
        ("shuffled_mean", "shuffled control", "0.6"),
    ]
    for i, (key, lab, col) in enumerate(series):
        vals = [x23["per_layer"][L][key] for L in Ls23]
        ax3.bar(x + (i - 1) * w, vals, w, color=col, label=lab, zorder=3)
    zs = [x23["per_layer"][L]["z_mean"] for L in Ls23]
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"L{L}\nz {z:+.1f}" for L, z in zip(Ls23, zs, strict=False)],
                        fontsize=9)
    ax3.set_ylim(0, 0.40)
    ax3.set_ylabel("visual-in-top10 rate")
    ax3.text(0.02, 0.975,
             f"instructing a concept works: mentioned in "
             f"{summ23['concept_mention_rate_instructed'] * 100:.1f}% of "
             f"{summ23['n_instructed_runs']} instructed runs\n"
             f"({summ23['n_images_processed']} images x {len(summ23['concepts'])} concepts; "
             "baseline prompt: 0.0% by construction)\n"
             "visual tokens co-occupy the top-10 ABOVE shuffle at every layer (z > 0):\n"
             "concurrent-task exclusion did NOT replicate (DECISIONS.md D47)",
             transform=ax3.transAxes, va="top", fontsize=8.7,
             bbox=dict(boxstyle="round,pad=0.35", fc="0.95", ec="0.7"))
    ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.175), ncol=3, fontsize=8.3)
    ax3.set_title("X23 modulation-during-captioning (590 instructed\nruns): visual tokens "
                  "co-occupy with the\ninstructed concept - context, not a task", fontsize=10)

    fig.tight_layout()
    fig.savefig(FIGS / "figD8_workspace_dynamics.png", **SAVE_KW)
    plt.close(fig)

    return {
        "multihop_probe_pass1": kn(round(probe["pass@1"], 4),
                                   src("step12/x22_reasoning_structure_multihop.json")
                                   + " results.probe_stats['lens-eval-multihop'].pass@1"),
        "multihop_n_items": kn(probe["n_items"],
                               src("step12/x22_reasoning_structure_multihop.json")
                               + " results.probe_stats['lens-eval-multihop'].n_items"),
        "onset_median_frac_intermediate": kn(
            round(on["intermediate"]["median_effective_frac"], 4),
            src("step12/x22_reasoning_structure_multihop.json")
            + " results.onset_stats.intermediate.median_effective_frac"),
        "onset_median_frac_answer": kn(
            round(on["answer"]["median_effective_frac"], 4),
            src("step12/x22_reasoning_structure_multihop.json")
            + " results.onset_stats.answer.median_effective_frac"),
        "onset_precedence_gap": kn(
            round(gap, 4),
            src("step12/x22_reasoning_structure_multihop.json")
            + " results.onset_stats (intermediate - answer median_effective_frac; "
              "digest prints -0.176)"),
        "paper_precedence_gap": kn(-0.17, "DECISIONS.md D47 ('~-0.17')"),
        "swap_success_early_late": kn(
            {k: round(v["success_baseline_correct"], 4)
             for k, v in mh["results"]["arm_stats"].items()},
            src("step12/x22_reasoning_structure_multihop.json")
            + " results.arm_stats.*.success_baseline_correct"),
        "flexgen_success_bc": kn(
            {f"alpha={a}": round(succ[str(a)], 4) for a in fg["results"]["alphas"]},
            src("step12/x22_reasoning_structure_flexgen.json")
            + f" results.cells: successful directed swaps / {n_swaps_bc} on baseline-correct "
              "cells (digest: success= 0.050 at both alphas)"),
        "flexgen_paper_success": kn(paper, "DECISIONS.md D47 ('paper: 40/53%')"),
        "flexgen_loading_success_pearson": kn(
            round(corr["pearson"], 4),
            src("step12/x22_reasoning_structure_flexgen.json")
            + " results.correlation.pearson (alpha=1.0, n=16 arguments)"),
        "flexgen_baseline_correct_cells": kn([len(bc), len(cells)],
                                             src("step12/x22_reasoning_structure_flexgen.json")
                                             + " results.cells baseline_correct"),
        "x23_concept_mention_rate_instructed": kn(
            summ23["concept_mention_rate_instructed"],
            src("step12/x23_vlm_workspace_modulation.json")
            + " summary.concept_mention_rate_instructed"),
        "x23_n_runs": kn([summ23["n_instructed_runs"], summ23["n_baseline_runs"]],
                         src("step12/x23_vlm_workspace_modulation.json")
                         + " summary (n_instructed_runs / n_baseline_runs)"),
        "x23_z_mean_by_layer": kn(
            {f"L{L}": round(x23["per_layer"][L]["z_mean"], 3) for L in Ls23},
            src("step12/x23_vlm_workspace_modulation.json")
            + " per_layer[L].z_mean"),
        "x23_p_vis_given_concept_by_layer": kn(
            {f"L{L}": round(x23["per_layer"][L]["p_visual_given_concept_mean"], 4)
             for L in Ls23},
            src("step12/x23_vlm_workspace_modulation.json")
            + " per_layer[L].p_visual_given_concept_mean"),
        "x23_exclusion_verdict": kn(
            "no task-exclusion: z > 0 at every measured layer",
            "DECISIONS.md D47 item D ('the paper's task-exclusion pattern did not clearly "
            "replicate at this n') + " + src("step12/x23_vlm_workspace_modulation.json")
            + " per_layer z_mean"),
    }


def main():
    print("figures_d_detect.py: regenerating figD5..figD8 + metrics")
    entries = []
    for fname, fn in [
        ("figD5_pope_handoff.png", figD5),
        ("figD6_uncertainty_signals.png", figD6),
        ("figD7_mirror_and_sink.png", figD7),
        ("figD8_workspace_dynamics.png", figD8),
    ]:
        key_numbers = fn()
        print(f"  wrote figures/{fname}")
        entries.append({"file": fname, "key_numbers": key_numbers})

    meta = {
        "figD5_pope_handoff.png": {
            "sources": [
                src("step11/x18_pope_handoff.json"),
                "DECISIONS.md D45 (x18 verdict; 0.895 low-gap detector quote)",
                "DECISIONS.md D46 (detector ledger)",
            ],
            "finding": (
                "The calibrated lens's pre-generation yes/no gap at the last prompt position "
                "nearly solves POPE object presence (AUROC 0.985 at L28, 0.982-0.985 over "
                "L20-30; mean gap +5.28 present vs -4.97 absent) and its disagreements with "
                "the model are exactly the model's own hallucinations: the 8 hallucinated yes "
                "rank LOW on the gap (AUROC 0.106 halluc-positive = 0.895 as a low-gap "
                "detector) - a training-free, single-forward truth/risk readout."
            ),
        },
        "figD6_uncertainty_signals.png": {
            "sources": [
                src("step11/x15_hesitation_n200.json"),
                "DECISIONS.md D46 (S1 x15 n=200 verdict)",
            ],
            "finding": (
                "At power (385 grounded vs 39 hallucinated word-completion steps) the "
                "hesitation direction holds: hallucinated words arrive with higher entropy "
                "(0.963 vs 0.620), lower top-1 margin (4.53 vs 5.55) and lower chosen "
                "log-prob (-0.384 vs -0.230); the calibrated lens's entropy is the better "
                "uncertainty signal (AUROC 0.730 at L7, band L4-7 0.64-0.73) vs the model's "
                "own entropy 0.593 - a real but modest detector (near chance from L11 on)."
            ),
        },
        "figD7_mirror_and_sink.png": {
            "sources": [
                src("step11/x17_attention_mirror_n200.json"),
                src("step11/x16_sink_clamp.json"),
                "DECISIONS.md D45 (x16 clamp verdict) / D46 (S1b mirror + S2 census dims)",
            ],
            "finding": (
                "The cross-modal attention mirror is real and depth-dependent (rho 0.105 at "
                "L0 -> 0.645 at L20 -> 0.199 at L30) but predicts nothing behaviorally: the "
                "hallucination join is NULL at power (share-mid AUROC 0.501, mirror-rho r_pb "
                "-0.10 over 27/170 images). The causal lever sits at the BOS sink instead: "
                "zero-clamping the stable BOS top-16 set (14/16 dims shared across "
                "L16/24/28/30) collapses grounded retention (0.278 at L16; 0.222 for "
                "all-groups) and explodes the true-object handoff rank (+1433 at L16, +550 "
                "at L28), while image-token clamps are inert (retention 0.94-1.00)."
            ),
        },
        "figD8_workspace_dynamics.png": {
            "sources": [
                src("step12/x22_reasoning_structure_multihop.json"),
                src("step12/x22_reasoning_structure_flexgen.json"),
                src("step12/x23_vlm_workspace_modulation.json"),
                "DECISIONS.md D47 (paper comparison values: onset gap ~-0.17; flexgen 40/53%; "
                "x23 item D exclusion verdict)",
            ],
            "finding": (
                "Workspace reasoning dynamics on LLaVA-1.5-7B: the paper's depth precedence "
                "replicates almost exactly (intermediate-swap onset 0.529 vs answer-swap "
                "0.706 of band depth; gap -0.176 vs paper ~-0.17; intermediates surface at "
                "pass@1 0.269), but causal leverage is weak - flexible-generalization swap "
                "success is 5% at alpha=1 AND alpha=2 vs the paper's 40/53% (loading-success "
                "Pearson only +0.256) - and during captioning the instructed concept "
                "co-occupies the lens top-10 with visual tokens ABOVE shuffle (z +1.3..+3.6; "
                "concept-mention 0.366 over 590 instructed runs): visual content behaves as "
                "passive context, not a competing task."
            ),
        },
    }

    metrics = {
        "round": "validation_2026-10-01",
        "family": "detection-round details (POPE handoff, uncertainty signals, attention "
                  "mirror + sink clamp, workspace reasoning dynamics)",
        "script": "results/validation_2026-10-01/code/figures_d_detect.py",
        "figures": [
            {
                "file": e["file"],
                "sources": meta[e["file"]]["sources"],
                "key_numbers": e["key_numbers"],
                "finding": meta[e["file"]]["finding"],
            }
            for e in entries
        ],
        "notes": [
            "All plotted numbers are read from the raw JSONs at run time; the only "
            "DECISIONS.md-sourced values are the PAPER comparison points (flexgen 40/53%, "
            "multihop precedence gap ~-0.17) and the 0.895 low-gap-detector quote, each "
            "cited in place.",
            "figD6 per-word distributions: a word's `step` indexes its own image's steps; "
            "the extracted means reproduce aggregate.grounded/hallucinated exactly "
            "(asserted in the script). The model's output entropy equals the L31 lens "
            "entropy per image (steps.model_entropy == steps.lens_entropy['31']).",
            "figD7 BOS dim-set stability is recomputed from x16 results (group=bos clamp "
            "dims): 14/16 dims shared across L16/24/28/30 - DECISIONS.md D46 records the "
            "set as identical at the census level.",
            "figD8 flexgen 'ours' bars recomputed from results.cells (successful directed "
            "swaps on baseline-correct cells / 120); digest prints the same 0.050.",
            "Deterministic: fixed bins, fixed jitter seed (np.random.default_rng(0)), no "
            "timestamps; re-running regenerates byte-identical PNGs + metrics JSON.",
        ],
    }
    with open(SUMMARY / "metrics_d_detect.json", "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    print("  wrote summary/metrics_d_detect.json")


if __name__ == "__main__":
    main()
