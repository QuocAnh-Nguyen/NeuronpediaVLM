#!/usr/bin/env python
"""Figures B1-B5: hallucination mechanism + intervention family.

Run:  ~/miniforge3/envs/vlm-lens/bin/python results/validation_2026-10-01/code/figures_hallucination.py
Outputs (deterministic, no timestamps):
  results/validation_2026-10-01/figures/figB1_dose_response.png
  results/validation_2026-10-01/figures/figB2_cure_test.png
  results/validation_2026-10-01/figures/figB3_commitment_boundary.png
  results/validation_2026-10-01/figures/figB4_prior_grounding_race.png
  results/validation_2026-10-01/figures/figB5_detector_ledger.png
  results/validation_2026-10-01/figures/figB6_x14_powerup.png

Every plotted number comes from raw/step10/*.json, raw/step11/*.json, raw/step4/*.json
or is quoted verbatim from DECISIONS.md (D39/D40) for the VALIDATED directedness
mining, whose per-generation output was not archived (only its citation).
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
x9b = load("step4/x9b_alpha_sweep.json")  # the x9b sweep itself: 2800 rows, n=100/cell
x7x9 = load("step4/x7_x9.json")          # the earlier X9 run: 1400 rows, 215 changed
x14n = load("step10/x14_workspace_probe_n200.json")  # X14 power-up: n=200 images
x19 = load("step11/x19_halluc_risk.json")


def save(fig, name):
    out = FIGDIR / name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# figB1 -- dose-response, JSON-grounded from raw/step4/x9b_alpha_sweep.json.
#
# That file IS the x9b sweep: 2800 greedy generations on 10 held-out COCO concept
# pairs x 10 samples x L16/L24, add/ablate/swap, one ResidualEdit per generation
# (positions="last" default). Per plotted point n=100 (10 pairs x 10 samples). It
# was synced from the box's /data/anhnq/vlm-lens-out/validation/step4/ (box mtime
# 2026-10-08T21:28:52Z) in commit 765138b; an earlier figure comment claiming
# "not archived" predated that sync and was wrong.
#
# Alpha scale: ablate/swap alphas are literal multiples of the residual norm; the
# add grid's k=1..32 is normalized per pair (alpha = k * norm / v_norm), so its
# cells span ~27-902 (L16) and ~58-2348 (L24) - ENTIRELY inside the degeneration
# zone. Its rows' `directed` flag is a raw word-presence check ("target word
# appears in output but not baseline") that counts repetition collapse as
# directed; the validated figure is D39's 7/215 = 3.3% from the earlier X9 run
# (raw/step4/x7_x9.json), mined with independent word-boundary recomputation at
# alpha 1.34-14.68.
# ----------------------------------------------------------------------------
import statistics  # noqa: E402


def _cells(mode: str, layer: int) -> list[dict]:
    """Aggregate one (mode, layer) sweep arm into per-strength cells (n=100 each)."""
    rows = [row for row in x9b["results"] if row["mode"] == mode and row["layer"] == layer]
    cells = []
    for strength in sorted({row["strength"] for row in rows}):
        selected = [row for row in rows if row["strength"] == strength]
        alphas = [row["edit"]["alpha"] for row in selected]
        n = len(selected)
        cells.append(
            {
                "strength": strength,
                "n": n,
                "changed": sum(row["changed"] for row in selected) / n,
                "directed_raw": sum(row["directed"] for row in selected) / n,
                "alpha": statistics.median(alphas),
                "alpha_min": min(alphas),
                "alpha_max": max(alphas),
            }
        )
    return cells


def _ci95(cells: list[dict]) -> tuple[list[float], list[float]]:
    """Binomial 95% CI half-widths (fraction units), clipped to [0, 1]."""
    low, high = [], []
    for cell in cells:
        probability, n = cell["changed"], cell["n"]
        half = 1.96 * (probability * (1 - probability) / n) ** 0.5
        low.append(min(half, probability))
        high.append(min(half, 1 - probability))
    return low, high


SERIES = [  # (mode, color, marker, label)
    ("ablate", BLUE, "o", "ablate"),
    ("swap", ORANGE, "s", "swap"),
    ("add", PINK, "^", "add (k-normalized: alpha = k*norm/v_norm)"),
]


def figB1():
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), sharey=True, constrained_layout=True)
    xmax = 3600.0

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim(0.42, xmax)
        ax.set_ylim(0, 116)
        ax.set_xticks([0.5, 1, 2, 4, 10, 27, 100, 300, 1000, 3000])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.axvspan(0.42, 4, color=GREEN, alpha=0.10, lw=0)
        ax.axvspan(27, xmax, color=VERMIL, alpha=0.10, lw=0)

    for ax, layer in zip(axes, [16, 24], strict=False):
        for mode, color, marker, label in SERIES:
            cells = _cells(mode, layer)
            low, high = _ci95(cells)
            ax.errorbar(
                [cell["alpha"] for cell in cells],
                [100 * cell["changed"] for cell in cells],
                yerr=[[100 * v for v in low], [100 * v for v in high]],
                xerr=[
                    [cell["alpha"] - cell["alpha_min"] for cell in cells],
                    [cell["alpha_max"] - cell["alpha"] for cell in cells],
                ],
                fmt=marker + "-",
                color=color,
                lw=2,
                ms=7,
                capsize=3,
                elinewidth=1.1,
                label=f"{label} @ L{layer}",
            )
            if mode == "add":
                # per-point RAW word-presence directed fraction (unvalidated)
                for cell in cells:
                    ax.annotate(
                        f"{100 * cell['directed_raw']:.0f}%",
                        (cell["alpha"], 100 * cell["changed"]),
                        textcoords="offset points",
                        xytext=(0, -13),
                        ha="center",
                        fontsize=7.5,
                        color=color,
                    )

        # D39's VALIDATED directedness on the earlier X9 run (7/215 = 3.3%)
        ax.axhline(3.3, color=GREY, lw=1.4, ls=":")
        ax.annotate(
            "D39 validated: 7/215 changed generations are\n"
            "concept-directed (3.3%), needing alpha 5-14x;\n"
            "add-point labels = RAW word-presence flag\n"
            "(unvalidated: counts 'cat cat cat' collapse)",
            (0.55, 12.5),
            fontsize=8,
            color=GREY,
            va="bottom",
        )
        ax.annotate(
            "sane window\nalpha <= 4",
            (0.55, 113),
            fontsize=8.5,
            color=GREEN,
            fontweight="bold",
            va="top",
        )
        ax.annotate(
            "degeneration alpha >= 27:\nrepetition collapse,\nnot injection",
            (32, 113),
            fontsize=8,
            color=VERMIL,
            va="top",
        )
        ax.set_title(f"L{layer}: change is cheap; the add grid never leaves the red zone")
        ax.set_xlabel("edit strength alpha (x residual norm, log; xerr = per-pair alpha range)")
        ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.78), fontsize=8)
    axes[0].set_ylabel("generations with changed caption (% of n=100)")

    fig.suptitle(
        "Dose-response from raw/step4/x9b_alpha_sweep.json (2800 generations, n=100/point, 95% CI)",
        fontweight="bold",
    )
    fig.supxlabel(
        "Source: results/validation_2026-10-01/raw/step4/x9b_alpha_sweep.json (box mtime 2026-10-08T21:28:52Z, "
        "md5 a3d7a85c...); the earlier X9 run raw/step4/x7_x9.json (1400 generations, 215 changed = 15.4%) "
        "is the run D39 mined for the validated directedness line.",
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


# ----------------------------------------------------------------------------
# figB6 -- X14 power-up: the same probe at n=200 (the n=24 hallucinated arm was
#          n=2). Run on the box 2026-10-18 (x14_workspace_probe.py --n-images 200
#          --handoff-rank, seeded selection seed=0, greedy, identity-transport +
#          moment payload). Thin dashed = the original n=24 run, overlaid.
# ----------------------------------------------------------------------------
def figB6():
    pl = x14n["per_layer"]
    pl0 = x14["per_layer"]
    layers = sorted(int(k) for k in pl)

    def get(table, key, default=None):
        return [table[str(l)].get(key, default) for l in layers]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), sharey=True, constrained_layout=True)
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylim(0.8, 9000)
        ax.set_xlabel("layer")
        ax.axvspan(24, 26, color=GREEN, alpha=0.10, lw=0)

    ax = axes[0]
    ax.plot(layers, get(pl0, "grounded_rank"), "--", color=BLUE, lw=1.2, alpha=0.7,
            label=f"grounded words, n=24 (n={pl0['0']['n_grounded']} words)")
    ax.plot(layers, get(pl0, "halluc_rank"), "--", color=VERMIL, lw=1.2, alpha=0.7,
            label=f"hallucinated words, n=24 (n={pl0['0']['n_hallucinated']} words)")
    ax.plot(layers, get(pl, "grounded_rank"), "o-", color=BLUE, lw=2, ms=4,
            label=f"grounded words, n=200 (n={pl['0']['n_grounded']})")
    ax.plot(layers, get(pl, "halluc_rank"), "s-", color=VERMIL, lw=2, ms=4,
            label=f"hallucinated words, n=200 (n={pl['0']['n_hallucinated']})")
    ax.annotate(
        "n=24 crossing L24-26 was an n=2 sample:\n"
        "at n=39 words the first grounded<halluc\n"
        "crossing moves to L30, where both are ~2",
        (1.2, 3.5),
        fontsize=8.5,
        color=VERMIL,
        va="bottom",
    )
    ax.set_ylabel("mean rank of the word's token id in the lens distribution")
    ax.set_title("Prior-vs-grounding race: prior lead survives (L0: halluc < grounded),\nthe crossover did not")
    ax.legend(loc="lower left", fontsize=8)

    ax = axes[1]
    ax.plot(layers, get(pl0, "last_patch_object_rank"), "--", color=GREY, lw=1.2, alpha=0.7,
            label="true objects at last IMAGE token, n=24")
    ax.plot(layers, get(pl, "last_patch_object_rank"), "d-", color=GREY, lw=2, ms=4,
            label="true objects at last IMAGE token, n=200")
    ax.plot(layers, get(pl, "true_object_rank_at_halluc_positions"), "^-", color=PINK, lw=2, ms=4,
            label="true objects AT hallucination positions, n=200")
    ax.plot(layers, get(pl, "halluc_rank"), "s-", color=VERMIL, lw=2, ms=4,
            label="hallucinated words (same positions), n=200")
    if pl["0"].get("handoff_true_object_rank_mean") is not None:
        ax.plot(layers, get(pl0, "handoff_true_object_rank_mean"), "--", color=GREEN, lw=1.2, alpha=0.7,
                label="true objects at last PROMPT (handoff), n=24")
        ax.plot(layers, get(pl, "handoff_true_object_rank_mean"), "o-", color=GREEN, lw=2, ms=4,
                label="true objects at last PROMPT (handoff), n=200")
    else:
        ax.annotate("handoff not computed in this file (needs --handoff-rank)", (2, 1200), fontsize=8, color=GREY)
    ax.annotate(
        "handoff MEAN flat at n=200 (~1500-1900):\n"
        "the n=24 mean fall (704 at L24) was a\n"
        "24-image fluctuation; per-image BEST still\n"
        "dips to 5-14 at L20-28 = a SUBSET of images\n"
        "surfaces truth at the last prompt position",
        (1.2, 3300),
        fontsize=8,
        color=GREEN,
        va="bottom",
    )
    ax.set_title("Where truth lives: never at image tokens; the hallucinated word\nbeats the truth at its own decoding positions")
    ax.legend(loc="lower left", fontsize=7.5)

    fig.suptitle(
        "X14 power-up: n=200 images (39 hallucinated / 385 grounded word instances vs n=24's 2 / 51)",
        fontweight="bold",
    )
    fig.supxlabel(
        "Source: raw/step10/x14_workspace_probe_n200.json (box run 2026-10-18, md5-verified below) overlaid on "
        "raw/step10/x14_workspace_probe.json (n=24); same script, same seed=0 selection, same greedy decoding, "
        "same identity-transport + moment-payload readout; ranks are 1-based, ties count as better.",
        fontsize=7.5,
        color=GREY,
    )
    return save(fig, "figB6_x14_powerup.png")


def main():
    FIGDIR.mkdir(exist_ok=True)
    SUMDIR.mkdir(exist_ok=True)
    files = {
        "figB1": figB1(),
        "figB2": figB2(),
        "figB3": figB3(),
        "figB4": figB4(),
        "figB5": figB5(),
        "figB6": figB6(),
    }

    pl13 = x13["per_layer"]
    dig12 = x12["digest"]
    pl14 = x14["per_layer"]
    pl14n = x14n["per_layer"]
    first_cross = next(
        (
            layer
            for layer in range(32)
            if pl14n[str(layer)]["grounded_rank"] < pl14n[str(layer)]["halluc_rank"]
        ),
        None,
    )
    pl19 = x19["per_layer"]
    metrics = {
        "figures": {
            "figB1_dose_response": {
                "file": "figures/figB1_dose_response.png",
                "sources": [
                    "raw/step4/x9b_alpha_sweep.json (the x9b sweep: 2800 generations, n=100 per point; synced from the box 2026-10-08T21:28:52Z, md5 a3d7a85c...)",
                    "raw/step4/x7_x9.json (the earlier X9 run: 1400 generations, 215 changed = 15.4%)",
                    "DECISIONS.md D39 (validated directionality mining of the 215 changed generations)",
                    "DECISIONS.md D40 P3 (alpha-normalization rule; earlier transcription, superseded by the archived JSON)",
                ],
                "key_numbers": {
                    "ablate_change_rate_pct_L16": [round(100 * c["changed"]) for c in _cells("ablate", 16)],
                    "ablate_change_rate_pct_L24": [round(100 * c["changed"]) for c in _cells("ablate", 24)],
                    "swap_change_rate_pct_L16": [round(100 * c["changed"]) for c in _cells("swap", 16)],
                    "swap_change_rate_pct_L24": [round(100 * c["changed"]) for c in _cells("swap", 24)],
                    "add_change_rate_pct_L16": [round(100 * c["changed"]) for c in _cells("add", 16)],
                    "add_change_rate_pct_L24": [round(100 * c["changed"]) for c in _cells("add", 24)],
                    "add_directed_raw_pct_L16": [round(100 * c["directed_raw"]) for c in _cells("add", 16)],
                    "add_alpha_median_range": [
                        round(_cells("add", 16)[0]["alpha"], 1),
                        round(_cells("add", 24)[-1]["alpha"], 0),
                    ],
                    "add_alpha_full_range_all_rows": [
                        round(min(r["edit"]["alpha"] for r in x9b["results"] if r["mode"] == "add"), 2),
                        round(max(r["edit"]["alpha"] for r in x9b["results"] if r["mode"] == "add"), 1),
                    ],
                    "concept_directed_of_changed": {"n_directed": 7, "n_changed": 215, "fraction_pct": 3.3},
                    "directed_alpha_needed_x_residual_norm": [5, 14],
                },
                "finding": "Caption change is cheap and monotone in alpha (ablate 7-34%, swap 7-39% at alpha 0.5-4; n=100/point) while concept-DIRECTED change is rare (D39 validated: 7/215 = 3.3% of the earlier run's changed generations, needing alpha 5-14x); the x9b add grid runs at k-normalized alpha ~27-2348, entirely in the degeneration zone, and its raw word-presence directed flag (20-60% of changed) counts repetition collapse as directed - the validated 3.3% line is the honest figure.",
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
            "figB6_x14_powerup": {
                "file": "figures/figB6_x14_powerup.png",
                "sources": [
                    "raw/step10/x14_workspace_probe_n200.json (box rerun 2026-10-18: x14_workspace_probe.py --n-images 200 --handoff-rank, seeded selection seed=0, greedy)",
                    "raw/step10/x14_workspace_probe.json (the original n=24 run, overlaid)",
                ],
                "key_numbers": {
                    "n_images": x14n["meta"]["n_images"],
                    "n_images_skipped_no_annotations": x14n["meta"]["n_images_skipped_no_annotations"],
                    "n_tokens_generated": x14n["meta"]["n_tokens_generated"],
                    "n_word_instances_grounded": x14n["meta"]["n_words_grounded"],
                    "n_word_instances_hallucinated": x14n["meta"]["n_words_hallucinated"],
                    "grounded_rank_L0": pl14n["0"]["grounded_rank"],
                    "halluc_rank_L0": pl14n["0"]["halluc_rank"],
                    "crossing_first_grounded_below_halluc_layer": first_cross,
                    "true_at_halluc_positions_L24": pl14n["24"]["true_object_rank_at_halluc_positions"],
                    "halluc_rank_L24": pl14n["24"]["halluc_rank"],
                    "handoff_mean_L0_L24_L30": [pl14n[str(l)]["handoff_true_object_rank_mean"] for l in (0, 24, 30)],
                    "handoff_best_L16_L20_L26": [pl14n[str(l)]["handoff_true_object_rank_best"] for l in (16, 20, 26)],
                    "last_patch_rank_range_L0_L30": [
                        min(pl14n[str(l)]["last_patch_object_rank"] for l in range(32)),
                        max(pl14n[str(l)]["last_patch_object_rank"] for l in range(32)),
                    ],
                },
                "finding": "At n=200 (39 hallucinated / 385 grounded word instances vs the n=24 run's 2 / 51; the two passes are bit-identical on all non-handoff fields, confirming greedy determinism): the prior-leads-from-L0 result survives qualitatively (hallucinated words rank below grounded at L0: 4024 vs 5622) but the lead shrinks from 6.8x to 1.4x; the n=24 crossover at L24-26 does NOT replicate - the first grounded<hallucinated crossing moves to L30, where both classes sit at rank ~2, so the late-band separation between the word classes is much weaker than n=24 implied; the does-the-model-know verdict SURVIVES (at L24 the hallucinated word ranks 5.6 vs the true object's 372 at the same positions); image-position non-decodability SURVIVES (last-patch ~1400-1800 at every layer); the L31 motor flip SURVIVES (both classes at rank 1.0). The mean-handoff headline does NOT survive: the mean true-object rank at the last prompt position is flat at n=200 (1590 at L0, 1511 at L24, 2214 at L30) while the n=24 mean fell to 704 at L24 - a 24-image fluctuation; the per-image BEST still dips to 5-14 across L20-28 (vs 39 at L16), so truth at the last prompt position surfaces for a SUBSET of images, not the typical image. Revision: the n=24 race-and-handoff narrative rested on n=2 hallucinated instances and 24 images.",
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
