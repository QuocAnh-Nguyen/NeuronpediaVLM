#!/usr/bin/env python
"""Figures B1-B5: hallucination mechanism + intervention family.

Run:  ~/miniforge3/envs/vlm-lens/bin/python results/validation_2026-10-01/code/figures_hallucination.py
Outputs (deterministic, no timestamps):
  results/validation_2026-10-01/figures/figB1_dose_response.png
  results/validation_2026-10-01/figures/figB2_cure_test.png
  results/validation_2026-10-01/figures/figB3_commitment_boundary.png
  results/validation_2026-10-01/figures/figB4_prior_grounding_race.png
  results/validation_2026-10-01/figures/figB5_detector_ledger.png
  results/validation_2026-10-01/summary/metrics_hallucination.json

Every plotted number comes from raw/step10/*.json, raw/step11/*.json, or is quoted
verbatim from DECISIONS.md (D39/D40) where the underlying step4 JSON was not
archived under results/raw (verified: no step4/ directory, no x9b/x7_x9 JSON).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "raw"
FIGDIR = BASE / "figures"
SUMDIR = BASE / "summary"

# Okabe-Ito colorblind-safe palette
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMIL = "#D55E00"
SKY = "#56B4E9"
PINK = "#CC79A7"
GREY = "#7F7F7F"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.30,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "legend.framealpha": 0.9,
        "legend.fontsize": 8.5,
    }
)


def load(rel):
    with open(RAW / rel) as f:
        return json.load(f)


x12 = load("step10/x12_cross_image_patch.json")
x13 = load("step10/x13_cure_test.json")
x14 = load("step10/x14_workspace_probe.json")
x15n = load("step11/x15_hesitation_n200.json")
x17n = load("step11/x17_attention_mirror_n200.json")
x18 = load("step11/x18_pope_handoff.json")
x19 = load("step11/x19_halluc_risk.json")


def save(fig, name):
    out = FIGDIR / name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# figB1 -- dose-response (x9b raw JSON not archived in results/raw;
#          numbers quoted from DECISIONS.md D39 + D40 P3, as instructed)
# ----------------------------------------------------------------------------
ALPHAS = [0.5, 1, 2, 4]
ABLATE_L16 = [7, 16, 24, 34]  # % generations changed, DECISIONS.md D40 P3
SWAP_L16 = [7, 12, 24, 39]  # DECISIONS.md D40 P3
SWAP_L24_A4 = 35  # only documented L24 point, DECISIONS.md D40 P3


def figB1():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True, constrained_layout=True)
    xmax = 1400

    for ax, _layer in zip(axes, [16, 24], strict=False):
        ax.set_xscale("log")
        ax.set_xlim(0.42, xmax)
        ax.set_ylim(0, 100)
        ax.set_xticks([0.5, 1, 2, 4, 10, 27, 100, 300, 1035])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        # sane window / degeneration shading (D40 P3)
        ax.axvspan(0.42, 4, color=GREEN, alpha=0.10, lw=0)
        ax.axvspan(27, xmax, color=VERMIL, alpha=0.10, lw=0)

    ax16, ax24 = axes

    ax16.plot(ALPHAS, ABLATE_L16, "o-", color=BLUE, lw=2, ms=6, label="ablate @ L16")
    ax16.plot(ALPHAS, SWAP_L16, "s-", color=ORANGE, lw=2, ms=6, label="swap @ L16")
    for a, v in zip(ALPHAS, ABLATE_L16, strict=False):
        ax16.annotate(f"{v}%", (a, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=BLUE)
    for a, v in zip(ALPHAS, SWAP_L16, strict=False):
        ax16.annotate(f"{v}%", (a, v), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8, color=ORANGE)
    ax16.axhline(3.3, color=VERMIL, lw=1.4, ls=":")
    ax16.annotate(
        "change is cheap, direction is rare:\n7/215 changed generations (3.3%)\nare concept-directed (D39); directed\nedits need alpha = 5-14x residual norm",
        (5.6, 8.5),
        fontsize=8,
        color=VERMIL,
        va="bottom",
    )
    ax16.annotate(
        "sane window\nalpha <= 4",
        (0.55, 88),
        fontsize=8.5,
        color=GREEN,
        fontweight="bold",
    )
    ax16.annotate(
        "degeneration alpha >= 27:\nadd grid ran at alpha 27-1035\n(k label mislabeled); 'directed'\nrows = repetition collapse\n(dog->cat gives 'cat cat cat')",
        (30, 52),
        fontsize=8,
        color=VERMIL,
        va="top",
    )
    ax16.set_title("L16 edits: monotone dose-response, swap steepest")
    ax16.set_xlabel("edit strength alpha (x residual norm, log scale)")
    ax16.set_ylabel("generations with changed caption (%)")
    ax16.legend(loc="upper left", bbox_to_anchor=(0.0, 0.80))

    ax24.plot([4], [SWAP_L24_A4], "D", color=ORANGE, ms=9)
    ax24.annotate("swap @ L24, alpha 4:\n35% (the only L24 point\nrecorded in D40 P3)", (4, 35), textcoords="offset points", xytext=(10, -4), fontsize=8.5, color=ORANGE)
    ax24.annotate(
        "add grid @ L24 runs at alpha ~ 58-1035\n(same k units; alpha = 27@L16 maps to 58@L24)\n- inside the degeneration zone",
        (60, 78),
        fontsize=8,
        color=VERMIL,
        va="top",
    )
    ax24.annotate("sane window\nalpha <= 4", (0.55, 88), fontsize=8.5, color=GREEN, fontweight="bold")
    ax24.annotate("degeneration alpha >= 27", (30, 92), fontsize=8.5, color=VERMIL, fontweight="bold")
    ax24.set_title("L24 edits: same zones; only the alpha=4 swap point is recorded")
    ax24.set_xlabel("edit strength alpha (x residual norm, log scale)")

    fig.suptitle(
        "Dose-response: caption change is cheap, concept-directed change is rare (~3%)",
        fontweight="bold",
    )
    fig.supxlabel(
        "Sources: DECISIONS.md D40 P3 (x9b alpha sweep, 2800 greedy generations) and D39; x9b JSON not archived under results/raw.",
        fontsize=7.5,
        color=GREY,
    )
    return save(fig, "figB1_dose_response.png")


# ----------------------------------------------------------------------------
# figB2 -- cure test (x13)
# ----------------------------------------------------------------------------
def figB2():
    layers = ["16", "24", "28", "30"]
    pl = x13["per_layer"]
    removed = [pl[l]["removed_rate"] for l in layers]
    ret_hall = [pl[l]["grounded_retention"] for l in layers]
    ret_ctrl = [pl[l]["control"]["grounded_retention"] for l in layers]
    ctrl_n = pl["16"]["control"]["n_arms"]
    n_arms = pl["16"]["n_halluc_arms"]

    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    xs = range(len(layers))
    w = 0.26
    b1 = ax.bar([x - w for x in xs], removed, w, color=VERMIL, label="hallucination removed (halluc. direction ablated)")
    b2 = ax.bar(list(xs), ret_hall, w, color=GREEN, label="grounded retention, halluc.-direction arm")
    b3 = ax.bar([x + w for x in xs], ret_ctrl, w, color=BLUE, label="grounded retention, GROUNDED-direction control")
    for bars in (b1, b2, b3):
        for r in bars:
            ax.annotate(f"{r.get_height():.2f}", (r.get_x() + r.get_width() / 2, r.get_height()), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_ylim(0, 1.14)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("ablation layer")
    ax.set_ylabel("rate over hallucination arms (fraction)")
    fig.legend([b1, b2, b3], [b.get_label() for b in (b1, b2, b3)], loc="outside lower center", ncol=3, fontsize=8.5)
    ax.set_title(
        "Mean-replacement ablation of the hallucinated category's J-lens direction: 90% surgical cure at L30\n"
        f"(alpha=1; {x13['summary']['n_images']} images, {n_arms} hallucination arms, {ctrl_n} grounded-control arms; halluc-dir retention stays 1.00 at every layer)",
        fontsize=10.5,
    )
    return save(fig, "figB2_cure_test.png")


# ----------------------------------------------------------------------------
# figB3 -- commitment boundary (x12)
# ----------------------------------------------------------------------------
def figB3():
    layers = sorted(int(k) for k in x12["digest"])
    dig = x12["digest"]

    def series(var, key):
        return [dig[str(l)][var][key] for l in layers]

    changed = series("image", "changed_rate")
    a_van = series("image", "a_disappearance_rate")
    b_app = series("image", "b_appearance_rate")
    last1 = series("image-last", "changed_rate")
    text = series("text", "changed_rate")

    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    ax.axvspan(20, 24, color=ORANGE, alpha=0.15, lw=0)
    ax.annotate(
        "commitment boundary L20-24:\nvisual content stops reaching the caption",
        (22, 0.985),
        ha="center",
        va="top",
        fontsize=8.5,
        color="#8a5a00",
    )
    h1 = ax.plot(layers, changed, "o-", color=BLUE, lw=2, ms=6, label="caption changed (all 576 image tokens patched)")
    h2 = ax.plot(layers, a_van, "s-", color=VERMIL, lw=2, ms=6, label="image A's objects vanish")
    h3 = ax.plot(layers, b_app, "^-", color=GREEN, lw=2, ms=6, label="image B's objects appear")
    h4 = ax.plot(layers, last1, "--", color=GREY, lw=1.6, marker="D", ms=4, label="single-position control (last image token only)")
    h5 = ax.plot(layers, text, ":", color=PINK, lw=1.6, marker="x", ms=4, label="text-position control (0.12-0.37 floor)")
    ax.set_xticks(layers)
    ax.set_xlabel("patched layer")
    ax.set_ylabel("rate over 8 disjoint-object COCO pairs (fraction)")
    ax.set_ylim(-0.04, 1.12)
    hs = [h1[0], h2[0], h3[0], h4[0], h5[0]]
    fig.legend(hs, [h.get_label() for h in hs], loc="outside lower center", ncol=2, fontsize=8.5)
    ax.set_title(
        "The commitment boundary: rewriting image tokens after L20-24 no longer changes the caption\n"
        "(B-objects appear only through L16; content is distributed, not stored in one handoff token)",
        fontsize=10.5,
    )
    return save(fig, "figB3_commitment_boundary.png")


# ----------------------------------------------------------------------------
# figB4 -- prior-vs-grounding race + text-mediated handoff (x14)
# ----------------------------------------------------------------------------
def figB4():
    pl = x14["per_layer"]
    layers = sorted(int(k) for k in pl)
    grounded = [pl[str(l)]["grounded_rank"] for l in layers]
    halluc = [pl[str(l)]["halluc_rank"] for l in layers]
    hand_mean = [pl[str(l)]["handoff_true_object_rank_mean"] for l in layers]
    hand_best = [pl[str(l)]["handoff_true_object_rank_best"] for l in layers]
    last_patch = [pl[str(l)]["last_patch_object_rank"] for l in layers]

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.5, 4.9), constrained_layout=True)

    axl.plot(layers, grounded, "o-", color=BLUE, lw=2, ms=4.5, label="grounded words (true rank)")
    axl.plot(layers, halluc, "s-", color=ORANGE, lw=2, ms=4.5, label="hallucinated words (prior rank)")
    axl.set_yscale("log")
    axl.axvspan(24, 26, color=GREEN, alpha=0.15, lw=0)
    axl.annotate("crossing L24-26:\ngrounded 4.9 vs\nhalluc 6.5 at L24;\n1.2 vs 4.5 at L26", (25, 30), ha="center", fontsize=8.5, color=GREEN, fontweight="bold")
    axl.annotate("L0: hallucinated ~787 vs\ngrounded ~5358 - the language\nprior leads ~7x before any\nvisual computation", (0.4, 1500), fontsize=8, color=ORANGE, va="top")
    axl.invert_yaxis()  # rank 1 (best) at top
    axl.set_xlabel("layer")
    axl.set_ylabel("mean lens rank of the word (1 = top; lower is better)")
    axl.set_title("The prior-vs-grounding race: grounding is computed,\nthe prior is the default", fontsize=10.5)
    axl.legend(loc="lower right", fontsize=8.5)
    axl.set_yticks([1, 10, 100, 1000])
    axl.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axl.yaxis.set_minor_formatter(NullFormatter())
    axl.set_ylim(0.8, 7000)

    axr.plot(layers, hand_mean, "o-", color=BLUE, lw=2, ms=4.5, label="best-per-image true-object rank (last prompt pos.)")
    axr.plot(layers, hand_best, "s-", color=GREEN, lw=2, ms=4.5, label="mean true-object rank (last prompt pos.)")
    axr.plot(layers, last_patch, ":", color=GREY, lw=1.6, label="last IMAGE position (never decodable, ~1.2-2.7k)")
    axr.set_yscale("log")
    axr.invert_yaxis()
    axr.axvspan(20, 25, color=SKY, alpha=0.15, lw=0)
    axr.annotate(
        "handoff completes ~L20-25:\nmean 1423 (L16) -> best 34 (L20)\n-> 6-15 (L22-25)",
        (23, 60),
        ha="center",
        va="top",
        fontsize=8.5,
        color="#1a6b8a",
    )
    axr.annotate(
        "motor regime: L31 readout flips to the\nimminent token - mean rank jumps to 3464",
        xy=(31, 3463.6),
        xytext=(22.8, 2300),
        fontsize=8.5,
        color=VERMIL,
        ha="left",
        va="top",
        arrowprops=dict(arrowstyle="->", color=VERMIL, lw=1.4),
    )
    axr.set_xlabel("layer")
    axr.set_ylabel("lens rank of the image's true categories (1 = top)")
    axr.set_title("Text-mediated visual->verbal handoff: true objects surface\nat the last prompt position, then the motor regime erases them", fontsize=10.5)
    axr.legend(loc="upper left", fontsize=8)
    axr.set_yticks([10, 100, 1000])
    axr.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axr.yaxis.set_minor_formatter(NullFormatter())
    axr.set_ylim(4, 4200)
    fig.suptitle(
        "Hallucination is the prior winning early; grounding must be computed to overtake it at L24-26 (24 images, calibrated lens)",
        fontweight="bold",
        fontsize=11,
    )
    return save(fig, "figB4_prior_grounding_race.png")


# ----------------------------------------------------------------------------
# figB5 -- detector ledger (x18, x15 n=200, x19, x17 n=200)
# ----------------------------------------------------------------------------
def figB5():
    auroc_x18 = x18["summary"]["auroc"]["present_vs_absent"]  # 0.985 @ L28
    lens7 = x15n["aggregate"]["auroc"]["lens_entropy"]["7"]  # 0.730
    model_ent = x15n["aggregate"]["auroc"]["model_entropy"]  # 0.593
    fused = x19["per_layer"]["30"]["auroc_fused"]  # 0.666
    gap_img = x19["per_layer"]["30"]["auroc_category_gap"]  # 0.632
    mean_ent = x19["per_layer"]["28"]["auroc_mean_entropy"]  # 0.620
    attn = x17n["hallucination_join"]["share_mid_layer"]["auroc_hallucinating_vs_clean"]  # 0.501

    rows = [
        ("Pre-generation yes/no gap, per question (L28)", auroc_x18, BLUE, "training-free; ONE forward at the last prompt position"),
        ("Per-step calibrated-lens entropy L7, per word", lens7, BLUE, "training-free; readout during generation"),
        ("Fused z(mean entropy) + z(category gap), per image (L30)", fused, GREEN, "training-free; single greedy pass, 2 signals"),
        ("Pre-generation true-category gap, per image (L30)", gap_img, BLUE, "training-free; single forward pass"),
        ("Image mean lens entropy, per image (L28)", mean_ent, BLUE, "training-free; single greedy pass"),
        ("Model's own entropy, per word (baseline)", model_ent, PINK, "no lens; model logits only"),
        ("Attention-to-image share at mid layer, per image", attn, GREY, "correlational only; NULL result"),
    ]
    rows.sort(key=lambda r: r[1])
    labels = [f"{r[0]}\n{r[3]}" for r in rows]

    fig, ax = plt.subplots(figsize=(11.5, 5.0), constrained_layout=True)
    ys = range(len(rows))
    vals = [r[1] for r in rows]
    cols = [r[2] for r in rows]
    ax.barh(ys, vals, color=cols, height=0.62)
    for y, (_label, v, _c, _tag) in zip(ys, rows, strict=False):
        ax.annotate(f"{v:.3f}", (v, y), textcoords="offset points", xytext=(5, 0), va="center", fontsize=9, fontweight="bold")
    ax.axvline(0.5, color="k", ls="--", lw=1.2)
    ax.annotate("chance (0.5)", (0.5, len(rows) - 0.25), fontsize=8.5, ha="center", xytext=(0, 6), textcoords="offset points")
    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("AUROC (hallucination-risk discrimination; 0.5 = chance)")
    ax.set_xlim(0, 1.0)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_title(
        "Detector ledger: workspace content predicts hallucination; attention does not\n"
        "pre-gen object presence nearly solved (0.985); caption-level risk 0.6-0.73\n"
        f"({x15n['meta']['n_images']} images; {x15n['aggregate']['n_words_hallucinated']}/{x15n['aggregate']['n_words_grounded']} halluc/grounded words; {x18['summary']['n_questions']} POPE questions)",
        fontsize=10.5,
    )
    fig.supxlabel(
        "All lens-based detectors are training-free (calibrated logit lens = identity transport + moment payload, zero fitting beyond moments) and single-pass unless noted.",
        fontsize=8,
        color=GREY,
    )
    return save(fig, "figB5_detector_ledger.png")


def main():
    FIGDIR.mkdir(exist_ok=True)
    SUMDIR.mkdir(exist_ok=True)
    files = {
        "figB1": figB1(),
        "figB2": figB2(),
        "figB3": figB3(),
        "figB4": figB4(),
        "figB5": figB5(),
    }

    pl13 = x13["per_layer"]
    dig12 = x12["digest"]
    pl14 = x14["per_layer"]
    pl19 = x19["per_layer"]
    metrics = {
        "figures": {
            "figB1_dose_response": {
                "file": "figures/figB1_dose_response.png",
                "sources": [
                    "DECISIONS.md D40 P3 (x9b alpha sweep: 2800 greedy generations, add/ablate/swap at L16/L24; raw JSON not archived under results/raw)",
                    "DECISIONS.md D39 (directionality mining of X9's 215 changed generations)",
                ],
                "key_numbers": {
                    "ablate_change_rate_pct_L16_alpha_0.5_1_2_4": ABLATE_L16,
                    "swap_change_rate_pct_L16_alpha_0.5_1_2_4": SWAP_L16,
                    "swap_change_rate_pct_L24_alpha_4": SWAP_L24_A4,
                    "add_grid_alpha_range_D40": [27, 1035],
                    "add_grid_alpha_range_D39": [1.34, 14.68],
                    "concept_directed_of_changed": {"n_directed": 7, "n_changed": 215, "fraction_pct": 3.3},
                    "directed_alpha_needed_x_residual_norm": [5, 14],
                },
                "finding": "Caption change is cheap (7-39% of generations at alpha 0.5-4, monotone in alpha) but concept-DIRECTED change is rare (7/215 = 3.3% of changed generations, needing alpha 5-14x the residual norm); alpha >= 27 breaks down into repetition degeneration, not injection.",
            },
            "figB2_cure_test": {
                "file": "figures/figB2_cure_test.png",
                "sources": ["raw/step10/x13_cure_test.json (per_layer, summary)", "DECISIONS.md D44 (X13)"],
                "key_numbers": {
                    "halluc_removed_rate_by_layer_L16_24_28_30": [pl13[l]["removed_rate"] for l in ("16", "24", "28", "30")],
                    "clean_cure_rate_by_layer": [pl13[l]["clean_cure_rate"] for l in ("16", "24", "28", "30")],
                    "grounded_retention_halluc_dir_by_layer": [pl13[l]["grounded_retention"] for l in ("16", "24", "28", "30")],
                    "grounded_retention_grounded_dir_control_by_layer": [round(pl13[l]["control"]["grounded_retention"], 4) for l in ("16", "24", "28", "30")],
                    "n_images": x13["summary"]["n_images"],
                    "n_hallucination_arms": pl13["16"]["n_halluc_arms"],
                    "n_control_arms": pl13["16"]["control"]["n_arms"],
                },
                "finding": "Mean-replacement ablation of the hallucinated category's J-lens direction at alpha=1 removes 0.50/0.60/0.70/0.90 of hallucinations at L16/24/28/30 while grounded-category retention stays 1.00 on those arms; ablating a GROUNDED category's direction is destructive (retention falls 0.97 -> 0.35 by L30) - a training-free 90% surgical cure at L30.",
            },
            "figB3_commitment_boundary": {
                "file": "figures/figB3_commitment_boundary.png",
                "sources": ["raw/step10/x12_cross_image_patch.json (digest)", "DECISIONS.md D44 (X12)"],
                "key_numbers": {
                    "changed_rate_by_layer_L0_8_12_16_20_24_28_30": [dig12[str(l)]["image"]["changed_rate"] for l in (0, 8, 12, 16, 20, 24, 28, 30)],
                    "a_objects_vanish_by_layer": [dig12[str(l)]["image"]["a_disappearance_rate"] for l in (0, 8, 12, 16, 20, 24, 28, 30)],
                    "b_objects_appear_by_layer": [round(dig12[str(l)]["image"]["b_appearance_rate"], 4) for l in (0, 8, 12, 16, 20, 24, 28, 30)],
                    "single_position_control_changed_max": max(dig12[str(l)]["image-last"]["changed_rate"] for l in (0, 8, 12, 16, 20, 24, 28, 30)),
                    "text_control_changed_range": [
                        min(dig12[str(l)]["text"]["changed_rate"] for l in (0, 8, 12, 16, 20, 24, 28, 30)),
                        max(dig12[str(l)]["text"]["changed_rate"] for l in (0, 8, 12, 16, 20, 24, 28, 30)),
                    ],
                    "n_pairs": x12["n_pairs"],
                },
                "finding": "Visual content is causally live at image positions only until ~L16-20: patching all 576 image tokens changes the caption at rate 1.00 -> 0.125 across L0-24 (A-objects vanish 1.00 -> 0 by L24; B-objects appear 0.41-0.49 only through L16), and single-token patches do nothing (<= 0.25) - the commitment boundary sits at L20-24 and content is distributed.",
            },
            "figB4_prior_grounding_race": {
                "file": "figures/figB4_prior_grounding_race.png",
                "sources": ["raw/step10/x14_workspace_probe.json (per_layer)", "DECISIONS.md D44 (X14)"],
                "key_numbers": {
                    "grounded_rank_L0": pl14["0"]["grounded_rank"],
                    "halluc_rank_L0": pl14["0"]["halluc_rank"],
                    "crossing_L24": {"grounded_rank": pl14["24"]["grounded_rank"], "halluc_rank": pl14["24"]["halluc_rank"]},
                    "crossing_L26": {"grounded_rank": pl14["26"]["grounded_rank"], "halluc_rank": pl14["26"]["halluc_rank"]},
                    "handoff_true_object_rank_mean_L16": pl14["16"]["handoff_true_object_rank_mean"],
                    "handoff_true_object_rank_best_L20": pl14["20"]["handoff_true_object_rank_best"],
                    "handoff_true_object_rank_best_L22_25": [pl14[str(l)]["handoff_true_object_rank_best"] for l in (22, 23, 24, 25)],
                    "motor_flip_mean_rank_L31": pl14["31"]["handoff_true_object_rank_mean"],
                    "n_images": x14["meta"]["n_images"],
                },
                "finding": "The language prior leads from L0 (hallucinated words rank ~787 vs grounded ~5358) while grounding is computed (5358 -> 1.18); the two trajectories cross at L24-26, the image's true objects surface at the last prompt position (mean 1423 at L16 -> best 34 at L20 -> 6-15 at L22-25), and the L31 motor regime flips the readout to the imminent token (mean rank 3464) - hallucination is the prior's default, grounding the computation that must overtake it.",
            },
            "figB5_detector_ledger": {
                "file": "figures/figB5_detector_ledger.png",
                "sources": [
                    "raw/step11/x18_pope_handoff.json (summary.auroc)",
                    "raw/step11/x15_hesitation_n200.json (aggregate.auroc)",
                    "raw/step11/x19_halluc_risk.json (per_layer)",
                    "raw/step11/x17_attention_mirror_n200.json (hallucination_join.share_mid_layer)",
                    "DECISIONS.md D46 (detector ledger table)",
                ],
                "key_numbers": {
                    "auroc_pre_generation_yes_no_gap_per_question_L28": x18["summary"]["auroc"]["present_vs_absent"],
                    "auroc_lens_entropy_L7_per_word": x15n["aggregate"]["auroc"]["lens_entropy"]["7"],
                    "auroc_fused_per_image_L30": pl19["30"]["auroc_fused"],
                    "auroc_true_category_gap_per_image_L30": pl19["30"]["auroc_category_gap"],
                    "auroc_image_mean_lens_entropy_per_image_L28": pl19["28"]["auroc_mean_entropy"],
                    "auroc_model_entropy_per_word": x15n["aggregate"]["auroc"]["model_entropy"],
                    "auroc_attention_share_mid_per_image_NULL": x17n["hallucination_join"]["share_mid_layer"]["auroc_hallucinating_vs_clean"],
                    "n_questions_pope": x18["summary"]["n_questions"],
                    "n_images_x19": pl19["30"]["n_images"],
                    "n_words_grounded_vs_hallucinated_x15": [x15n["aggregate"]["n_words_grounded"], x15n["aggregate"]["n_words_hallucinated"]],
                },
                "finding": "Hallucination is best predicted by calibrated workspace content, not attention: the pre-generation yes/no gap nearly solves object presence (0.985 over 185 questions; 0.895 as a low-gap detector of the model's own hallucinated yes), while caption-level risk is a 0.6-0.73 problem (fused 0.666) and the attention mirror is NULL (0.501); all lens detectors are training-free and single-pass.",
            },
        }
    }
    with open(SUMDIR / "metrics_hallucination.json", "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")

    for k, v in files.items():
        print(f"{k}: {v} ({v.stat().st_size} bytes)")
    print(f"metrics: {SUMDIR / 'metrics_hallucination.json'}")


if __name__ == "__main__":
    main()
