#!/usr/bin/env python
"""
figures_e_pope.py - extension-round figures (figE1..figE3) for the 2026-10-01
validation round: the x24 POPE microscope (3 x 2596 questions) read through
the calibrated J-lens - commitment layering raw vs calibrated, the
prior-vs-evidence gap, and the training-free per-question error-prediction
ledger on the adversarial split.

Deterministic: fixed palette, fixed layout, fixed bins/jitter seed, no
timestamps inside figures; re-running regenerates byte-identical PNGs and the
metrics JSON.

Outputs (relative to this script's parent dir, results/validation_2026-10-01/):
  figures/figE1_commitment_layering.png
  figures/figE2_prior_evidence_gap.png
  figures/figE3_error_prediction_ledger.png
  summary/metrics_e_pope.json

Data sources:
  raw/step40/x24_pope_raw.json       (raw lens: random/popular/adversarial)
  raw/step40/x24_pope_calib.json     (calibrated lens: random split)
  raw/step40/x24_pope_calib_adv.json (calibrated lens: adversarial split)
  DECISIONS.md D49 + EXPERIMENTS.md Stage 6 (prose comparison values are quoted
  in annotations only; every plotted number is recomputed from the JSONs at
  run time using the files' own error_auc / commitment conventions)

Run: <python> results/validation_2026-10-01/code/figures_e_pope.py
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

# Okabe-Ito colorblind-safe palette (same conventions as figures_d_detect.py)
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
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.framealpha": 0.9,
        "legend.fontsize": 9,
    }
)

SAVE_KW = dict(bbox_inches="tight", metadata={"Software": "figures_e_pope.py"})

RNG = np.random.default_rng(0)  # fixed seed -> deterministic jitter (none used)

RAW_FILE = "step40/x24_pope_raw.json"
CAL_FILE = "step40/x24_pope_calib.json"
CAL_ADV_FILE = "step40/x24_pope_calib_adv.json"


def load(rel):
    with open(RAW / rel) as f:
        return json.load(f)


def src(rel):
    return f"raw/{rel}"


def kn(value, source):
    """Key-number entry: value + its exact provenance."""
    return {"value": value, "source": source}


def pct1(x):
    """Percent with 1 decimal, half-up (matches DECISIONS.md quotes, e.g. 30.9)."""
    return (Decimal(str(x)) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


def au3(x):
    """AUROC to 3 decimals, half-up (matches DECISIONS.md quotes, e.g. 0.823)."""
    return Decimal(str(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)


def rank_auc(feat, err):
    """Rank AUROC (Mann-Whitney U), positive group = err, average ranks on ties.

    Reproduces the x24 files' own `error_auc` convention (summary.error_auc.rule)
    exactly: the recomputed per-layer auc_error_gap / auc_error_pyes match the
    stored values to machine precision.
    """
    f = np.asarray(feat, dtype=float)
    e = np.asarray(err, dtype=bool)
    order = np.argsort(f, kind="mergesort")
    fs = f[order]
    ranks = np.empty(len(f))
    i = 0
    while i < len(f):
        j = i
        while j + 1 < len(f) and fs[j + 1] == fs[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1 = int(e.sum())
    n0 = len(e) - n1
    return float((ranks[e].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def aligned_conf(q, layer):
    """Answer-aligned lens confidence (rules convention: P_yes for a yes answer,
    1 - P_yes for a no answer)."""
    p = q["p_yes"][str(layer)]
    return p if q["model_answer"] == "yes" else 1.0 - p


def gap_feature(qs, layer):
    """error_auc convention (b): -(answer-signed gap) ranks 'the committed answer
    is not supported' high; positive group = model answer != POPE label."""
    return [
        -(q["gap"][str(layer)] if q["model_answer"] == "yes" else -q["gap"][str(layer)])
        for q in qs
    ]


def err_flags(qs):
    return [not q["correct"] for q in qs]


# --------------------------------------------------------------- figE1: X24
def figE1():
    raw = load(RAW_FILE)
    cal = load(CAL_FILE)
    cadv = load(CAL_ADV_FILE)

    n_layers = len(raw["results"]["random"]["questions"][0]["p_yes"])
    layers = list(range(n_layers))

    # calibrated runs exist only for random (x24_pope_calib) and adversarial
    # (x24_pope_calib_adv); popular has no calibrated counterpart
    runs = [
        ("random", raw["results"]["random"], cal["results"]["random"]),
        ("popular", raw["results"]["popular"], None),
        ("adversarial", raw["results"]["adversarial"], cadv["results"]["adversarial"]),
    ]

    def comm(res):
        h = res["summary"]["commitment_histogram"]
        n = res["summary"]["n_questions"]
        per = [h.get(str(L), 0) / n for L in layers]
        return per, h.get("none", 0) / n, n

    fig, axes = plt.subplots(1, 4, figsize=(16.6, 4.9), width_ratios=[1, 1, 1, 1.6])

    l5_raw, l03_cal, l5_cal = {}, {}, {}
    for ax, (split, rres, cres) in zip(axes[:3], runs, strict=False):
        raw_fr, raw_none, n = comm(rres)
        xs = np.arange(n_layers + 1)  # last slot = never committed ("none")
        ax.bar(xs - 0.2, raw_fr + [raw_none], 0.38, color=ORANGE,
               label="raw lens", zorder=3)
        if cres is not None:
            cal_fr, cal_none, _ = comm(cres)
            ax.bar(xs + 0.2, cal_fr + [cal_none], 0.38, color=BLUE,
                   label="calibrated lens", zorder=3)
            l5_cal[split] = cal_fr[5]
            l03_cal[split] = sum(cal_fr[:4])
        l5_raw[split] = raw_fr[5]
        ax.axvspan(4.55, 5.45, color=ORANGE, alpha=0.12, zorder=1)
        ax.set_xticks([0, 3, 5, 10, 14, 18, 22, 26, 28, 31, 32])
        ax.set_xticklabels(["0", "3", "5", "10", "14", "18", "22", "26", "28", "31", "n"])
        ax.set_xlim(-0.8, n_layers + 0.8)
        ax.set_ylim(0, 0.36)
        ax.set_xlabel("commitment layer (n = never)")
        if ax is axes[0]:
            ax.set_ylabel("fraction of questions")
        ax.set_title(f"{split}  (n={n})", fontsize=10.5)
        ax.legend(loc="upper right", fontsize=8.2)
        ax.text(0.03, 0.80, f"raw L5 {pct1(raw_fr[5])}%",
                transform=ax.transAxes, fontsize=8.8, color=ORANGE,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=ORANGE, alpha=0.9))
        if cres is not None:
            ax.text(0.03, 0.66, f"cal L0-3 {pct1(l03_cal[split])}%",
                    transform=ax.transAxes, fontsize=8.8, color=BLUE,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=BLUE, alpha=0.9))
        else:
            ax.text(0.03, 0.66, "no calibrated run",
                    transform=ax.transAxes, fontsize=8.4, color=GREY, style="italic")

    # right panel: per-layer mean P_yes, present (label yes) vs absent (label no),
    # raw vs calibrated, random split
    ax = axes[3]
    pl_r = raw["results"]["random"]["per_layer"]
    pl_c = cal["results"]["random"]["per_layer"]
    pres_r = [pl_r[str(L)]["mean_p_yes_label_yes"] for L in layers]
    absn_r = [pl_r[str(L)]["mean_p_yes_label_no"] for L in layers]
    pres_c = [pl_c[str(L)]["mean_p_yes_label_yes"] for L in layers]
    absn_c = [pl_c[str(L)]["mean_p_yes_label_no"] for L in layers]
    ax.plot(layers, pres_r, color=ORANGE, ls="--", lw=1.4,
            label="raw, object present (label yes)")
    ax.plot(layers, absn_r, color=VERMIL, ls="--", lw=1.4,
            label="raw, object absent (label no)")
    ax.plot(layers, pres_c, color=BLUE, lw=1.8, label="calibrated, object present")
    ax.plot(layers, absn_c, color=SKY, lw=1.8, label="calibrated, object absent")
    ax.axhline(0.5, color=GREY, ls=":", lw=1)
    ax.text(0.99, 0.5, "chance 0.5", transform=ax.get_yaxis_transform(),
            va="bottom", ha="right", fontsize=8.5, color=GREY)
    ax.set_xlim(0, n_layers - 1)
    ax.set_ylim(0, 1.34)
    ax.set_xticks([0, 5, 10, 15, 18, 22, 26, 28, 31])
    ax.set_xlabel("lens layer")
    ax.set_ylabel("mean P_yes at the answer position")
    ax.set_title("random split: the raw trajectory is the artifact", fontsize=10.5)
    ax.legend(loc="upper left", fontsize=8.2)

    sep_r5 = abs(pl_r["5"]["mean_p_yes_label_yes"] - pl_r["5"]["mean_p_yes_label_no"])
    sep_c5 = abs(pl_c["5"]["mean_p_yes_label_yes"] - pl_c["5"]["mean_p_yes_label_no"])
    cross_raw = {
        s: float(np.mean([q["p_yes"]["5"] >= 0.5
                          for q in raw["results"][s]["questions"]]))
        for s in ("random", "popular", "adversarial")
    }
    ax.text(0.02, 0.985,
            "the misreading calibration removes: raw P_yes whipsaws (dip L4,\n"
            f"spike L5) and first 'commits' at L5 for {pct1(l5_raw['random'])}/"
            f"{pct1(l5_raw['popular'])}/\n"
            f"{pct1(l5_raw['adversarial'])}% (random/popular/adversarial) while "
            "present/absent means\n"
            f"differ by {sep_r5:.3f} there (P_yes(5) >= 0.5 on "
            f"{pct1(cross_raw['random'])}/{pct1(cross_raw['popular'])}/\n"
            f"{pct1(cross_raw['adversarial'])}%) - the over-scaled lens crossing 0.5 "
            "without object\n"
            "evidence. Calibrated: smooth rise, L5 commitment 0.0%, mass at L0-3\n"
            f"({pct1(l03_cal['random'])}% random / {pct1(l03_cal['adversarial'])}%) "
            "+ genuine L18/L28 re-formation.",
            transform=ax.transAxes, va="top", fontsize=7.8, color="0.1",
            bbox=dict(boxstyle="round,pad=0.4", fc="0.95", ec="0.7"))

    fig.suptitle(
        "X24 POPE microscope: calibration dissolves the raw lens's fake early commitment - "
        "the L5 spike is a\nmiscalibration artifact (it carries no object evidence), "
        "not the model's disposition timeline",
        fontsize=11.5, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIGS / "figE1_commitment_layering.png", **SAVE_KW)
    plt.close(fig)

    cal_hist_rand = cal["results"]["random"]["summary"]["commitment_histogram"]
    return {
        "n_questions_per_split": kn(
            {s: r["summary"]["n_questions"] for s, r, _ in runs},
            src(RAW_FILE) + " results.<split>.summary.n_questions"),
        "raw_L5_commitment_frac": kn(
            {s: float(pct1(v)) for s, v in l5_raw.items()},
            src(RAW_FILE) + " results.<split>.summary.commitment_histogram['5'] / n"),
        "raw_pyes5_ge05_frac": kn(
            {s: float(pct1(v)) for s, v in cross_raw.items()},
            src(RAW_FILE) + " per-question p_yes['5'] >= 0.5"),
        "calibrated_L0-3_mass_frac": kn(
            {s: float(pct1(v)) for s, v in l03_cal.items()},
            src(CAL_FILE) + " + " + src(CAL_ADV_FILE)
            + " commitment_histogram['0'..'3'] / n (no calibrated popular run)"),
        "calibrated_L5_commitment_frac": kn(
            {s: float(pct1(v)) for s, v in l5_cal.items()},
            src(CAL_FILE) + " + " + src(CAL_ADV_FILE) + " commitment_histogram['5'] / n"),
        "calibrated_random_hist_counts": kn(
            cal_hist_rand,
            src(CAL_FILE) + " results.random.summary.commitment_histogram (680 @L3, "
            "419 @L0, 462 @L18, 480 @L28 of 2596)"),
        "raw_present_absent_meanPyes_sep_L5": kn(
            {"raw": round(sep_r5, 4), "calibrated": round(sep_c5, 4)},
            src(RAW_FILE) + " / " + src(CAL_FILE)
            + " per_layer['5'].mean_p_yes_label_yes vs _label_no (random)"),
        "commitment_none_frac": kn(
            {s: float(pct1(comm(r)[1])) for s, r, _ in runs},
            src(RAW_FILE) + " commitment_histogram['none'] / n"),
    }


# --------------------------------------------------------------- figE2: X24
def figE2():
    raw = load(RAW_FILE)
    cal = load(CAL_FILE)
    cadv = load(CAL_ADV_FILE)

    n_layers = len(raw["results"]["random"]["questions"][0]["p_yes"])
    layers = list(range(n_layers))

    # calibrated: random from x24_pope_calib, adversarial from x24_pope_calib_adv
    runs = {
        "random": (cal["results"]["random"], raw["results"]["random"], BLUE),
        "adversarial": (cadv["results"]["adversarial"],
                        raw["results"]["adversarial"], VERMIL),
    }

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.4), width_ratios=[1, 1])

    # ---- left: per-layer mean P_yes(image) vs P_yes(image-free), calibrated
    plateau = {}
    dips = {}
    for split, (res, _, color) in runs.items():
        gc = res["gap_curve"]
        img = [gc[L]["mean_p_yes"] for L in layers]
        free = [gc[L]["mean_p_yes_free"] for L in layers]
        ax.plot(layers, free, color=color, ls="--", lw=1.6,
                label=f"{split}: image-free (prior path)")
        ax.plot(layers, img, color=color, lw=2.0,
                label=f"{split}: image (evidence path)")
        # contiguous plateau run (>= 0.995) containing L18
        lo = hi = 18
        while lo - 1 >= 0 and free[lo - 1] >= 0.995:
            lo -= 1
        while hi + 1 < n_layers and free[hi + 1] >= 0.995:
            hi += 1
        plateau[split] = (lo, hi, min(free[lo : hi + 1]))
        di = int(np.argmin(img))
        dips[split] = (img[18], di, img[di])
    ax.axhline(0.5, color=GREY, ls=":", lw=1)
    ax.text(0.99, 0.5, "chance 0.5", transform=ax.get_yaxis_transform(),
            va="bottom", ha="right", fontsize=8.5, color=GREY)
    ax.set_xlim(0, n_layers - 1)
    ax.set_ylim(0, 1.34)
    ax.set_xticks([0, 5, 10, 15, 18, 22, 26, 28, 31])
    ax.set_xlabel("lens layer")
    ax.set_ylabel("mean P_yes at the answer position")
    ax.set_title("prior path saturates, evidence path dips (calibrated)", fontsize=10.5)
    ax.legend(loc="lower left", fontsize=8.2, ncol=2)
    rp = plateau["random"]
    ra = plateau["adversarial"]
    d_r, d_a = dips["random"], dips["adversarial"]
    ax.text(0.02, 0.985,
            f"image-free saturates at prior-yes: mean P_yes >= 0.995 over L{rp[0]}-L{rp[1]} "
            f"(min {rp[2]:.3f});\n"
            f"adversarial L{ra[0]}-L{ra[1]} (min {ra[2]:.3f}). The image path dips "
            "mid-band exactly there:\n"
            f"random {d_r[0]:.3f} @L18 (min {d_r[2]:.3f} @L{d_r[1]}), adversarial "
            f"{d_a[0]:.3f} @L18 (min {d_a[2]:.3f} @L{d_a[1]})\n"
            "- the image EVIDENCE pulls against the language prior where the "
            "sensory block reads.",
            transform=ax.transAxes, va="top", fontsize=8.2,
            bbox=dict(boxstyle="round,pad=0.4", fc="0.95", ec="0.7", alpha=0.93))

    # ---- right: per-layer answer-signed gap AUROC for error prediction
    curves, baselines, invar = {}, {}, {}
    for split, (res, rres, color) in runs.items():
        qs = res["questions"]
        err = err_flags(qs)
        auc = [rank_auc(gap_feature(qs, L), err) for L in layers]
        curves[split] = auc
        mb = res["summary"]["error_auc"]["final_logit_baseline"]["auc_margin_abs"]
        baselines[split] = mb
        ax2.plot(layers, auc, color=color, lw=2.0,
                 label=f"{split}: answer-signed gap")
        ax2.axhline(mb, color=color, ls=":", lw=1.4,
                    label=f"{split}: model |margin| {mb:.3f}")
        # calibration-invariance: same feature on the raw file
        rqs = rres["questions"]
        rerr = err_flags(rqs)
        rauc = [rank_auc(gap_feature(rqs, L), rerr) for L in layers]
        d = [abs(a - b) for a, b in zip(auc, rauc, strict=True)]
        invar[split] = {
            "L31_exact": auc[31] == rauc[31],
            "L31_value": auc[31],
            "max_delta": max(d),
            "argmax_delta": int(np.argmax(d)),
        }
    ax2.axhline(0.5, color=GREY, ls=":", lw=1)
    ax2.text(0.01, 0.5, "chance 0.5", transform=ax2.get_yaxis_transform(),
             va="bottom", fontsize=8.5, color=GREY)
    ax2.axvspan(20.5, 25.5, color=GREY, alpha=0.08, zorder=1)
    ax2.set_xlim(0, n_layers - 1)
    ax2.set_ylim(0.3, 1.0)
    ax2.set_xticks([0, 5, 10, 15, 18, 21, 25, 29, 31])
    ax2.set_xlabel("lens layer")
    ax2.set_ylabel("AUROC(model error), signed-gap feature")
    ax2.set_title("the gap feature vs the model's own margin", fontsize=10.5)
    ax2.legend(loc="upper left", fontsize=8.2)

    a_r, a_a = curves["random"], curves["adversarial"]
    b_r, b_a = baselines["random"], baselines["adversarial"]
    b_r_i, b_a_i = invar["random"], invar["adversarial"]
    band_lo, band_hi = min(a_a[21:26]), max(a_a[21:26])
    best_a = int(np.argmax(a_a))
    best_r = int(np.argmax(a_r))
    ax2.annotate(
        f"adversarial best L{best_a} = {a_a[best_a]:.3f}\n"
        f"(> |margin| {b_a:.3f}); random best\n"
        f"L{best_r} = {a_r[best_r]:.3f} (< |margin| {b_r:.3f})",
        xy=(best_a, a_a[best_a]), xytext=(0.42, 0.30), textcoords="axes fraction",
        va="bottom", fontsize=8.4, arrowprops=dict(arrowstyle="->", color="0.2", lw=0.9),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.9),
    )
    ax2.text(0.02, 0.965,
             f"L21-25 band (shaded): {band_lo:.3f}-{band_hi:.3f} (adversarial)\n"
             f"calibration-invariance: L31 AUROC identical raw vs calibrated\n"
             f"(adversarial {b_a_i['L31_value']:.3f} exact="
             f"{b_a_i['L31_exact']}; random {b_r_i['L31_value']:.3f} exact="
             f"{b_r_i['L31_exact']});\n"
             f"through the band the signed-gap AUC shifts by <= "
             f"{max(b_r_i['max_delta'], b_a_i['max_delta']):.3f}\n"
             f"(max @L{b_a_i['argmax_delta']} adversarial, "
             f"{b_r_i['max_delta']:.3f} @L{b_r_i['argmax_delta']} random)",
             transform=ax2.transAxes, va="top", fontsize=8.2,
             bbox=dict(boxstyle="round,pad=0.4", fc="0.95", ec="0.7", alpha=0.93))

    fig.suptitle(
        "X24: the prior-vs-evidence gap is only readable calibrated - and its answer-signed "
        "feature beats the model's own confidence\nexactly where priors bite (adversarial), "
        "while on random the margin stays on top",
        fontsize=11.5, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(FIGS / "figE2_prior_evidence_gap.png", **SAVE_KW)
    plt.close(fig)

    return {
        "n_errors": kn(
            {"random": int(sum(err_flags(cal["results"]["random"]["questions"]))),
             "adversarial": int(sum(err_flags(cadv["results"]["adversarial"]["questions"])))},
            src(CAL_FILE) + " / " + src(CAL_ADV_FILE)
            + " per-question correct == false"),
        "image_free_plateau": kn(
            {s: {"layers": [v[0], v[1]], "min_mean_p_yes_free": round(v[2], 4)}
             for s, v in plateau.items()},
            src(CAL_FILE) + " / " + src(CAL_ADV_FILE)
            + " gap_curve[l].mean_p_yes_free, contiguous run >= 0.995 through L18"),
        "image_path_dip": kn(
            {s: {"mean_L18": round(v[0], 4), "min": round(v[2], 4), "argmin_layer": v[1]}
             for s, v in dips.items()},
            src(CAL_FILE) + " / " + src(CAL_ADV_FILE) + " gap_curve[l].mean_p_yes"),
        "signed_gap_auroc_best": kn(
            {"random": {"layer": best_r, "auc": round(a_r[best_r], 4)},
             "adversarial": {"layer": best_a, "auc": round(a_a[best_a], 4)}},
            "recomputed rank AUROC, rules convention (b); matches "
            + src(CAL_FILE) + " / " + src(CAL_ADV_FILE)
            + " summary.error_auc.best (feature gap)"),
        "signed_gap_auroc_band_L21_25": kn(
            {"random": [round(v, 4) for v in a_r[21:26]],
             "adversarial": [round(v, 4) for v in a_a[21:26]]},
            "recomputed per layer L21..L25 (calibrated)"),
        "margin_baseline_auroc": kn(
            {"random": round(b_r, 4), "adversarial": round(b_a, 4)},
            src(CAL_FILE) + " / " + src(CAL_ADV_FILE)
            + " summary.error_auc.final_logit_baseline.auc_margin_abs"),
        "calibration_invariance": kn(
            {s: {"L31_auroc_calibrated": round(v["L31_value"], 4),
                 "L31_identical_raw_vs_calibrated": v["L31_exact"],
                 "max_band_delta": round(v["max_delta"], 4),
                 "argmax_layer": v["argmax_delta"]}
             for s, v in invar.items()},
            "same feature recomputed on " + src(RAW_FILE)
            + "; L31 readout is calibration-invariant (per-question gap['31'] "
            "identical in the raw and calibrated files)"),
    }


# --------------------------------------------------------------- figE3: X24
def figE3():
    cadv = load(CAL_ADV_FILE)
    res = cadv["results"]["adversarial"]
    qs = res["questions"]
    err = err_flags(qs)
    n = len(qs)
    n_err = int(sum(err))
    n_layers = len(qs[0]["p_yes"])

    # 1. final-layer image-free contrast |p_yes_free - p_yes| @L31.
    #    Bare |gap| ranks CORRECT answers high on this split (AUC 0.155 < 0.5);
    #    the detector direction - flag error when the image adds nothing - is
    #    1 - that (the same keep-the-detector-direction convention the file's
    #    own final_logit_baseline uses for |margin|).
    contrast31 = [abs(q["p_yes_free"]["31"] - q["p_yes"]["31"]) for q in qs]
    a_contrast_bare = rank_auc(contrast31, err)
    a_contrast = 1.0 - a_contrast_bare

    # 2. the model's own |margin| (same questions, same n; model logits, never
    #    calibrated). Recomputed == stored final_logit_baseline.auc_margin_abs.
    a_margin_bare = rank_auc([abs(q["margin"]) for q in qs], err)
    a_margin = 1.0 - a_margin_bare
    a_margin_stored = res["summary"]["error_auc"]["final_logit_baseline"]["auc_margin_abs"]

    # 3. answer-signed gap at its best layer (rules convention (b), kept sign)
    gap_auc = [rank_auc(gap_feature(qs, L), err) for L in range(n_layers)]
    best_gap_layer = int(np.argmax(gap_auc))
    a_gap = gap_auc[best_gap_layer]

    # 4. mid-band lens confidence at its most anti-predictive layer (rules
    #    convention (a): feature = 1 - aligned P_yes; kept sign, so AUROC < 0.5
    #    reads 'mid-band certainty marks the model's errors')
    mid_band = res["summary"]["error_auc"]["mid_band"]["layers"]
    conf_auc = {
        L: rank_auc([1.0 - aligned_conf(q, L) for q in qs], err) for L in mid_band
    }
    best_conf_layer = min(mid_band, key=lambda L: conf_auc[L])
    a_conf = conf_auc[best_conf_layer]

    rows = [  # (label, auroc, color, hatch)
        (f"answer-signed gap @L{best_gap_layer} (best layer)", a_gap, BLUE, ""),
        ("image-free contrast |p_yes_free - p_yes| @L31\n(low-contrast error flag)",
         a_contrast, GREEN, ""),
        ("model's own |margin|", a_margin, ORANGE, ""),
        (f"mid-band lens confidence @L{best_conf_layer}\n"
         f"(best of mid band L{mid_band[0]}-L{mid_band[-1]}; sign kept)",
         a_conf, PURPLE, "///"),
    ]
    rows.sort(key=lambda r: -r[1])

    fig, ax = plt.subplots(figsize=(11.6, 4.4))
    ys = np.arange(len(rows))[::-1]
    for y, (_lab, v, color, hatch) in zip(ys, rows, strict=True):
        ax.barh(y, v, 0.62, color=color, hatch=hatch,
                edgecolor="white" if not hatch else color, zorder=3)
        note = ""
        if v < 0.5:
            note = "  (below chance)"
        ax.text(v + 0.012, y, f"{v:.3f}{note}", va="center", fontsize=9.5,
                fontweight="bold" if v >= 0.5 else "normal",
                color="0.1" if v >= 0.5 else PURPLE)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.2)
    ax.axvline(0.5, color=GREY, ls=":", lw=1.2, zorder=2)
    ax.text(0.5, len(rows) - 0.28, " chance 0.5", fontsize=8.6, color=GREY)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.55, len(rows) - 0.15)
    ax.set_xlabel(
        f"training-free per-question error prediction, AUROC "
        f"(POPE adversarial, n={n}, {n_err} errors, calibrated lens)")
    ax.set_title("the lens's counterfactual beats the model's own confidence; "
                 "mid-band certainty is about the wrong thing", fontsize=11)
    ax.text(
        0.985, 0.04,
        f"the lens's counterfactual (image removed) beats the model's own confidence exactly\n"
        f"where priors bite: low final-layer contrast flags {n_err}/{n} errors at "
        f"{a_contrast:.3f} vs |margin| {a_margin:.3f};\n"
        f"the signed gap peaks at L{best_gap_layer} ({a_gap:.3f}); mid-band certainty "
        f"anti-predicts ({a_conf:.3f} < 0.5 @L{best_conf_layer}).",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.6,
        bbox=dict(boxstyle="round,pad=0.4", fc="0.95", ec="0.7"))

    fig.savefig(FIGS / "figE3_error_prediction_ledger.png", **SAVE_KW)
    plt.close(fig)

    return {
        "n_questions": kn(n, src(CAL_ADV_FILE) + " results.adversarial.summary.n_questions"),
        "n_errors": kn(n_err, src(CAL_ADV_FILE)
                       + " per-question correct == false (adversarial)"),
        "image_free_contrast_L31": kn(
            {"auroc_detector": round(a_contrast, 4),
             "auroc_bare_abs_gap": round(a_contrast_bare, 4)},
            src(CAL_ADV_FILE) + " per-question |p_yes_free['31'] - p_yes['31']|; "
            "bare |gap| ranks correct answers high on this split, so the ledger "
            "keeps the detector direction (low contrast -> error), the same "
            "convention as the file's final_logit_baseline for |margin|"),
        "model_margin": kn(
            {"auroc_detector": round(a_margin, 4),
             "auroc_bare_abs_margin": round(a_margin_bare, 4),
             "stored_auc_margin_abs": round(a_margin_stored, 4),
             "recompute_matches_stored": abs(a_margin - a_margin_stored) < 1e-12},
            src(CAL_ADV_FILE)
            + " summary.error_auc.final_logit_baseline.auc_margin_abs; per-question "
              "|margin| recomputation agrees"),
        "signed_gap_best_layer": kn(
            {"layer": best_gap_layer, "auc": round(a_gap, 4)},
            "recomputed rank AUROC, rules convention (b); matches "
            + src(CAL_ADV_FILE) + " summary.error_auc.best (auc 0.8659 @L31)"),
        "mid_band_confidence_best_layer": kn(
            {"layer": best_conf_layer, "auc": round(a_conf, 4),
             "mid_band_layers": mid_band},
            "rules convention (a) feature (1 - aligned P_yes), sign kept; most "
            "below-chance layer of summary.error_auc.mid_band.layers; recomputed "
            "per-layer values bracket the register's 0.31-0.61 quote"),
        "chance": kn(0.5, "definition"),
    }


def main():
    print("figures_e_pope.py: regenerating figE1..figE3 + metrics")
    entries = []
    for fname, fn in [
        ("figE1_commitment_layering.png", figE1),
        ("figE2_prior_evidence_gap.png", figE2),
        ("figE3_error_prediction_ledger.png", figE3),
    ]:
        key_numbers = fn()
        entries.append({"file": fname, "key_numbers": key_numbers})

    e1 = entries[0]["key_numbers"]
    e2 = entries[1]["key_numbers"]
    e3 = entries[2]["key_numbers"]
    print(
        f"figE1 commitment layering: raw L5 fake-commit "
        f"{e1['raw_L5_commitment_frac']['value']['random']}%"
        f"/{e1['raw_L5_commitment_frac']['value']['popular']}%"
        f"/{e1['raw_L5_commitment_frac']['value']['adversarial']}% "
        f"(random/popular/adversarial; P_yes(5)>=0.5 on "
        f"{e1['raw_pyes5_ge05_frac']['value']['random']}%"
        f"/{e1['raw_pyes5_ge05_frac']['value']['popular']}%"
        f"/{e1['raw_pyes5_ge05_frac']['value']['adversarial']}%) -> calibrated L0-3 "
        f"{e1['calibrated_L0-3_mass_frac']['value']['random']}% (random) / "
        f"{e1['calibrated_L0-3_mass_frac']['value']['adversarial']}% (adversarial), "
        f"calibrated L5 {e1['calibrated_L5_commitment_frac']['value']['random']}%; "
        "wrote figures/figE1_commitment_layering.png"
    )
    pl = e2["image_free_plateau"]["value"]
    dp = e2["image_path_dip"]["value"]
    sg = e2["signed_gap_auroc_best"]["value"]
    bd = e2["signed_gap_auroc_band_L21_25"]["value"]["adversarial"]
    mb = e2["margin_baseline_auroc"]["value"]
    iv = e2["calibration_invariance"]["value"]
    print(
        f"figE2 prior-vs-evidence gap: image-free saturates "
        f"{pl['random']['min_mean_p_yes_free']:.3f}"
        f"+ over L{pl['random']['layers'][0]}-L{pl['random']['layers'][1]} (adversarial "
        f"{pl['adversarial']['min_mean_p_yes_free']:.3f}+) vs image dip "
        f"{dp['random']['mean_L18']:.3f} @L18 (adversarial {dp['adversarial']['mean_L18']:.3f}); "
        f"signed-gap AUROC best L{sg['adversarial']['layer']} {au3(sg['adversarial']['auc'])} "
        f"(adversarial) / L{sg['random']['layer']} {au3(sg['random']['auc'])} (random), "
        f"band L21-25 adversarial {au3(min(bd))}-{au3(max(bd))} vs |margin| "
        f"{au3(mb['adversarial'])} (random {au3(mb['random'])}); "
        f"calibration-invariance: L31 identical raw=calibrated "
        f"(adv {au3(iv['adversarial']['L31_auroc_calibrated'])}, random "
        f"{au3(iv['random']['L31_auroc_calibrated'])}), band shift <= "
        f"{au3(max(iv['adversarial']['max_band_delta'], iv['random']['max_band_delta']))}; "
        "wrote figures/figE2_prior_evidence_gap.png"
    )
    c3 = e3["image_free_contrast_L31"]["value"]
    m3 = e3["model_margin"]["value"]
    g3 = e3["signed_gap_best_layer"]["value"]
    f3 = e3["mid_band_confidence_best_layer"]["value"]
    print(
        f"figE3 error ledger (adversarial, n={e3['n_questions']['value']}, "
        f"{e3['n_errors']['value']} errors): signed gap L{g3['layer']} {au3(g3['auc'])} > "
        f"image-free contrast @L31 {au3(c3['auroc_detector'])} (bare |gap| "
        f"{au3(c3['auroc_bare_abs_gap'])}) > model |margin| {au3(m3['auroc_detector'])} >> "
        f"mid-band confidence L{f3['layer']} {au3(f3['auc'])} (below chance); "
        "wrote figures/figE3_error_prediction_ledger.png"
    )

    meta = {
        "figE1_commitment_layering.png": {
            "sources": [
                src(RAW_FILE),
                src(CAL_FILE),
                src(CAL_ADV_FILE),
                "DECISIONS.md D49 item 2 (fake early commitment; 31-39% L5 quote)",
                "EXPERIMENTS.md Stage 6 (recomputed 30.9/32.8/38.1% L5 fractions; "
                "L0-3 680 @L3 + 419 @L0 on random)",
            ],
            "finding": (
                "Calibration dissolves the raw lens's fake early commitment. Raw, the "
                "first sign-stable commitment falls at L5 for 30.9/32.8/38.1% of "
                "questions (random/popular/adversarial) while the present/absent mean "
                "P_yes curves differ by 0.005 there and raw P_yes(5) >= 0.5 on "
                "70.6/73.5/73.3% - the over-scaled lens crossing 0.5 without object "
                "evidence. Calibrated, the L5 spike is gone (0.0%) and commitment mass "
                "moves to L0-3 (42.3% random, 51.2% adversarial) plus genuine mid/late "
                "re-formation (L18 17.8%, L28 18.5% on random): the yes/no disposition "
                "forms extremely early for a large subset, then re-forms at the motor "
                "layers (DECISIONS.md D49)."
            ),
        },
        "figE2_prior_evidence_gap.png": {
            "sources": [
                src(CAL_FILE),
                src(CAL_ADV_FILE),
                src(RAW_FILE) + " (calibration-invariance check)",
                "DECISIONS.md D49 item 2 (0.999 prior-yes saturation; gap feature)",
                "EXPERIMENTS.md Stage 6 (0.744-0.823 L21-25 adversarial, 0.866 @L31)",
            ],
            "finding": (
                "The prior-vs-evidence gap is only readable calibrated: the image-free "
                "path saturates at prior-yes (mean P_yes >= 0.995 over L11-27, both "
                "splits) while the image path dips mid-band (0.484 @L18 random, 0.612 "
                "adversarial) - the image evidence pulls against the language prior "
                "exactly where the sensory block reads. The answer-signed gap feature "
                "(error_auc convention b) is a moderate per-question error predictor "
                "rising through the band: 0.744-0.823 across L21-25 on adversarial, "
                "best 0.866 @L31 > the model's own |margin| 0.795; on random the gap "
                "never beats the margin (best 0.802 < 0.874). Calibration-invariance: "
                "the L31 AUROC is identical raw vs calibrated on both splits (the "
                "per-question gap['31'] is byte-identical in the two files) and the "
                "band values shift by <= 0.087 (max @L24 adversarial), preserving the "
                "split ordering and the rising-band shape."
            ),
        },
        "figE3_error_prediction_ledger.png": {
            "sources": [
                src(CAL_ADV_FILE),
                src(RAW_FILE) + " (calibration-invariance check)",
                "DECISIONS.md D49 item 2 (honest error-prediction verdict; 0.845 vs 0.795)",
                "EXPERIMENTS.md Stage 6 (extended detector ledger table)",
            ],
            "finding": (
                "Honest error-prediction ledger on the adversarial split (n=2596, 384 "
                "errors): the answer-signed gap at its best layer (0.866 @L31) and the "
                "final-layer image-free contrast (0.845 as the low-contrast error flag; "
                "the bare |p_yes_free - p_yes| scores 0.155, i.e. high contrast marks "
                "correct answers) both beat the model's own |margin| (0.795); mid-band "
                "lens confidence anti-predicts (0.346 @L17, the most below-chance layer "
                "of the mid band L12-29; sign kept, so mid-band certainty marks the "
                "model's errors). Computing 'what would the model say without the "
                "image' from the lens framework predicts the model's hallucination "
                "errors better than the model's own confidence, where priors bite "
                "(DECISIONS.md D49)."
            ),
        },
    }

    metrics = {
        "round": "validation_2026-10-01",
        "family": "extension round: x24 POPE microscope (commitment layering, "
                  "prior-vs-evidence gap, training-free error-prediction ledger)",
        "script": "results/validation_2026-10-01/code/figures_e_pope.py",
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
            "All plotted numbers are computed from the raw JSONs at run time; the "
            "register prose values (30.9/32.8/38.1%, 0.744-0.823, 0.866, 0.845, 0.795) "
            "are quoted in annotations only and reproduce from these files.",
            "rank_auc implements the files' own error_auc rule (rank AUROC, "
            "Mann-Whitney U, positive group = model answer != POPE label, average "
            "ranks on ties); the recomputed per-layer auc_error_gap / auc_error_pyes "
            "match the stored per_layer values to machine precision, and the "
            "recomputed |margin| AUROC equals the stored "
            "final_logit_baseline.auc_margin_abs (0.7945 adversarial).",
            "Detector-direction convention (as in the register's extended ledger): "
            "the image-free contrast and |margin| rows are scored in their "
            "error-flagging direction (low contrast / low margin -> error), while the "
            "mid-band confidence row keeps the feature's own sign (AUROC < 0.5 = "
            "mid-band certainty marks the model's errors).",
            "No calibrated popular run exists (x24_pope_calib covers random, "
            "x24_pope_calib_adv covers adversarial); the popular panel shows the raw "
            "histogram only and the L0-3 calibrated mass is quoted for random and "
            "adversarial.",
            "Deterministic: fixed palette, fixed layout, no timestamps; re-running "
            "regenerates byte-identical PNGs + metrics JSON.",
        ],
    }
    with open(SUMMARY / "metrics_e_pope.json", "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    print("  wrote summary/metrics_e_pope.json")


if __name__ == "__main__":
    main()
