#!/usr/bin/env python
"""
figures_e_causal.py - extension-round figures (figE4, figE5) for the 2026-10-01
validation round: the mitigation-mitigation frontier (x25: does the J-lens
causal arm beat the contrastive baselines WITHOUT paying for it) and visual
register blindness (x26: probe-readable states the unembed cannot verbalize).

Deterministic: fixed palette, fixed layout, fixed bins/jitter seed, no
timestamps inside figures; re-running regenerates byte-identical PNGs and the
metrics JSON.

Outputs (relative to this script's parent dir, results/validation_2026-10-01/):
  figures/figE4_mitigation_frontier.png
  figures/figE5_register_blindness.png
  summary/metrics_e_causal.json

Data sources:
  raw/step40/x25_pope.json      (9 arms x 60 POPE-random questions: accuracy,
                                unparsed, yes-rate, F1 per arm)
  raw/step40/x25_captions.json  (same 9 arms: CHAIR-s/CHAIR-r over the caption
                                images + teacher-forced lens-trajectory
                                object-top10 shift vs baseline per layer)
  raw/step40/x26_registers.json (20 images: ridge-probe object AUC per row
                                class, lens/unembed object-in-top10 per row
                                class, register stickiness z, attention-
                                stickiness correlation, D48 band)
  DECISIONS.md D49/D50 (E3 verdict quoted in annotation only)

Run: <python> results/validation_2026-10-01/code/figures_e_causal.py
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
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.framealpha": 0.9,
        "legend.fontsize": 9,
    }
)

ARM_DISPLAY = {
    "baseline": "baseline",
    "m3id_tau_0.5": "M3ID \u03c4=0.5",
    "m3id_tau_1": "M3ID \u03c4=1",
    "m3id_tau_2": "M3ID \u03c4=2",
    "vcd_tau_0.5": "VCD \u03c4=0.5",
    "vcd_tau_1": "VCD \u03c4=1",
    "vcd_tau_2": "VCD \u03c4=2",
    "prior_ablate_alpha_0.5": "prior-ablate \u03b1=0.5",
    "prior_ablate_alpha_1": "prior-ablate \u03b1=1",
}

# annotation offset per arm, in points: (dx, dy, ha) - fixed layout, not data
ARM_LABEL_OFFSET = {
    "baseline": (-8, 10, "right"),
    "m3id_tau_1": (9, 7, "left"),
    "m3id_tau_0.5": (-9, 3, "right"),
    "m3id_tau_2": (10, 8, "left"),
    "vcd_tau_0.5": (-9, 4, "right"),
    "vcd_tau_1": (9, 4, "left"),
    "vcd_tau_2": (10, -11, "left"),
    "prior_ablate_alpha_0.5": (8, 10, "left"),
    "prior_ablate_alpha_1": (8, -14, "left"),
}
SAVE_KW = dict(bbox_inches="tight", metadata={"Software": "figures_e_causal.py"})

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


def arm_table(x25_pope, x25_cap):
    """Per-arm flat records: POPE accuracy/unparsed, CHAIR-s/r, mid-band
    teacher-forced object-top10 delta vs baseline (mean over the lens layers)."""
    layers = [str(L) for L in x25_cap["layers"]]  # teacher-forced layers 16..28
    arms = x25_pope["summary"]["arm_ids"]
    rows = {}
    for a in arms:
        pr = x25_pope["arms"][a]["pope"]["random"]
        cap = x25_cap["arms"][a]["captions"]
        m = cap["metrics"]
        shifts = x25_cap["lens_trajectory"]["shift_vs_baseline"].get(a, {})
        per_layer = {L: shifts[L]["object_topk_rate"] for L in layers if L in shifts}
        mid = float(np.mean(list(per_layer.values()))) if per_layer else 0.0
        rows[a] = {
            "family": x25_pope["arms"][a]["params"]["family"],
            "acc": pr["accuracy"],
            "unparsed": pr["n_unparsed"],
            "n_questions": pr["n"],
            "chair_s": m["chair_s"],
            "chair_r": m["chair_r"],
            "n_captions": m["n_captions"],
            "midband_delta": mid,  # fraction; *100 = percentage points
            "per_layer_delta": per_layer,
        }
    return rows


def family_color(family):
    return {"baseline": GREY, "m3id": VERMIL, "vcd": ORANGE}.get(family, GREEN)


# ---------------------------------------------------------------- figE4: X25
def figE4():
    x25_pope = load("step40/x25_pope.json")
    x25_cap = load("step40/x25_captions.json")
    rows = arm_table(x25_pope, x25_cap)
    arm_ids = x25_pope["summary"]["arm_ids"]

    b = rows["baseline"]
    b_acc, b_chair = b["acc"], b["chair_s"]
    causal = [a for a in arm_ids if rows[a]["family"] == "prior-ablate"]
    contrast = [a for a in arm_ids if rows[a]["family"] in ("m3id", "vcd")]
    best_causal = max(causal, key=lambda a: rows[a]["acc"])
    c_deltas = [rows[a]["midband_delta"] * 100 for a in causal]
    k_deltas = [rows[a]["midband_delta"] * 100 for a in contrast]
    broken = [a for a in contrast if rows[a]["unparsed"] > 0]
    clean_hi = [a for a in contrast if rows[a]["chair_s"] > b_chair]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.8), width_ratios=[1.5, 1])

    # -- left: mitigation frontier --------------------------------------------
    # region where CHAIR-s rises above the baseline arm
    y_top = max(rows[a]["chair_s"] for a in arm_ids) * 1.22
    ax.axhspan(b_chair, y_top, color=VERMIL, alpha=0.07, zorder=1)
    ax.text(
        0.985, 0.965,
        "more hallucination, cleaner format\n"
        f"(CHAIR-s > {pct1(b_chair)}%; all arms here: unparsed = 0)",
        transform=ax.transAxes, va="top", ha="right", fontsize=8.6,
        color=VERMIL, zorder=5,
    )

    # baseline cross-reference
    ax.axvline(b_acc, color=GREY, ls=":", lw=1.1, zorder=2)
    ax.axhline(b_chair, color=GREY, ls=":", lw=1.1, zorder=2)

    seen = {}
    for a in arm_ids:
        r = rows[a]
        x, y = r["acc"], r["chair_s"]
        # deterministic jitter only for arms sharing a rounded coordinate
        key = (round(x, 3), round(y, 3))
        dup = seen.get(key, 0)
        seen[key] = dup + 1
        if dup:
            x = x + (RNG.random() - 0.5) * 0.012 * dup
            y = y + (RNG.random() - 0.5) * 0.012 * dup
        marker = "o" if r["unparsed"] == 0 else "s"
        ax.scatter(
            [x], [y], s=110, marker=marker, color=family_color(r["family"]),
            edgecolor="k", linewidth=0.7, zorder=6,
            label=None,
        )
        dx, dy, ha = ARM_LABEL_OFFSET[a]
        ax.annotate(
            f"{ARM_DISPLAY[a]}\nunparsed {r['unparsed']}",
            (x, y), textcoords="offset points", xytext=(dx, dy),
            ha=ha, va="center", fontsize=8.0,
            color=family_color(r["family"]), fontweight="bold",
            zorder=7,
        )

    # family legend (colour = family, shape = format kept vs broken)
    ax.scatter([], [], s=90, marker="o", color=VERMIL, edgecolor="k", lw=0.7,
               label="M3ID (contrast)")
    ax.scatter([], [], s=90, marker="o", color=ORANGE, edgecolor="k", lw=0.7,
               label="VCD (contrast)")
    ax.scatter([], [], s=90, marker="o", color=GREEN, edgecolor="k", lw=0.7,
               label="prior-ablate (J-lens causal)")
    ax.scatter([], [], s=90, marker="o", color=GREY, edgecolor="k", lw=0.7,
               label="baseline")
    ax.scatter([], [], s=90, marker="s", color="0.7", edgecolor="k", lw=0.7,
               label="square = unparsed outputs > 0")
    ax.legend(loc="lower right", fontsize=8.2, ncol=1)

    ax.text(
        0.015, 0.985,
        "THE J-LENS CAUSAL ARM IS THE ONLY\n"
        "CLEAN WIN: only prior-ablate improves\n"
        f"POPE acc ({pct1(b_acc)}% \u2192 {pct1(rows[best_causal]['acc'])}%) "
        f"with CHAIR-s\n"
        f"untouched ({pct1(b_chair)}% \u2192 {pct1(rows[best_causal]['chair_s'])}%) "
        f"and mid-band\n"
        f"content nearly so ({min(c_deltas):.1f}..{max(c_deltas):.1f} pts).\n"
        "Contrast arms pay: \u03c4\u22651 breaks the format\n"
        f"({min(rows[a]['unparsed'] for a in broken)}-"
        f"{max(rows[a]['unparsed'] for a in broken)}/60 unparsed, acc "
        f"{pct1(min(rows[a]['acc'] for a in broken))}-"
        f"{pct1(max(rows[a]['acc'] for a in broken))}%);\n"
        f"even format-clean \u03c4=0.5 doses pay\n"
        f"{min(k_deltas):.1f}..{max(k_deltas):.1f} pts of mid-band content.",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.4", fc="0.95", ec="0.7", alpha=0.95),
    )

    ax.set_xlim(-0.06, 1.12)
    ax.set_ylim(-0.03, y_top)
    ax.set_xlabel("POPE accuracy (random split, n=60 questions)")
    ax.set_ylabel(f"CHAIR-s ({rows['baseline']['n_captions']}/60 caption images scored)")
    ax.set_title("mitigation frontier: hallucination metric vs answer accuracy",
                 fontsize=10.5)

    # -- right: mid-band workspace-content bill --------------------------------
    order = sorted(arm_ids, key=lambda a: rows[a]["midband_delta"], reverse=True)
    ys = np.arange(len(order))
    vals = [rows[a]["midband_delta"] * 100 for a in order]
    ax2.barh(
        ys, vals, 0.62,
        color=[family_color(rows[a]["family"]) for a in order],
        edgecolor="k", linewidth=0.6, zorder=3,
    )
    ax2.axvline(0.0, color="k", lw=1.2, zorder=4)
    for y, v in zip(ys, vals, strict=False):
        ax2.text(v - 0.12 if v < 0 else v + 0.12, y, f"{v:+.2f}",
                 va="center", ha="right" if v < 0 else "left",
                 fontsize=8.6, fontweight="bold", zorder=5)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([ARM_DISPLAY[a] for a in order], fontsize=8.8)
    ax2.invert_yaxis()
    ax2.set_xlim(min(vals) * 1.28, 0.9)
    ax2.set_xlabel(
        "mid-band object-top10 delta vs baseline (points)\n"
        "(mean L16-28, teacher-forced captions)"
    )
    ax2.set_title("the workspace-content bill per arm", fontsize=10.5)
    ax2.text(
        0.03, 0.97,
        f"destructive contrast doses: {min(k_deltas):.1f}..{max(k_deltas):.1f} pts\n"
        f"J-lens causal arms: {min(c_deltas):.2f}..{max(c_deltas):.2f} pts\n"
        "(zero line = baseline workspace content)",
        transform=ax2.transAxes, va="top", fontsize=8.6,
        bbox=dict(boxstyle="round,pad=0.35", fc="0.95", ec="0.7"),
    )

    fig.suptitle(
        "X25 mitigation microscope (LLaVA-1.5-7B, 9 arms): M3ID/VCD trade accuracy or "
        "workspace content for their CHAIR gains,\nwhile the J-lens prior-ablation is the "
        "only arm that improves POPE accuracy with caption metric and mid-band content "
        "untouched",
        fontsize=11.5, y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIGS / "figE4_mitigation_frontier.png", **SAVE_KW)
    plt.close(fig)

    return {
        "n_arms": kn(len(arm_ids), src("step40/x25_pope.json") + " summary.n_arms"),
        "pope_split": kn("random", src("step40/x25_pope.json") + " arms.*.pope keys"),
        "n_questions": kn(b["n_questions"],
                          src("step40/x25_pope.json") + " arms.baseline.pope.random.n"),
        "n_caption_images": kn(x25_cap["n_images"],
                               src("step40/x25_captions.json") + " n_images"),
        "baseline_accuracy": kn(round(b_acc, 4),
                                src("step40/x25_pope.json")
                                + " arms.baseline.pope.random.accuracy"),
        "best_causal_arm": kn(best_causal,
                              src("step40/x25_pope.json")
                              + " arms: prior-ablate arm with max accuracy"),
        "best_f1_arm": kn(
            x25_pope["summary"]["pope"]["random"]["best_f1_arm"],
            src("step40/x25_pope.json") + " summary.pope.random.best_f1_arm"),
        "per_arm": kn(
            {a: {
                "accuracy": round(rows[a]["acc"], 4),
                "unparsed": rows[a]["unparsed"],
                "chair_s": round(rows[a]["chair_s"], 4),
                "chair_r": round(rows[a]["chair_r"], 4),
                "midband_delta_pts": round(rows[a]["midband_delta"] * 100, 3),
                "per_layer_delta_pts": {
                    L: round(v * 100, 3) for L, v in rows[a]["per_layer_delta"].items()},
            } for a in arm_ids},
            src("step40/x25_pope.json") + " + " + src("step40/x25_captions.json")
            + " (accuracy/unparsed; chair; mid-band = mean over "
            + ",".join(str(L) for L in x25_cap["layers"])
            + " of lens_trajectory.shift_vs_baseline.object_topk_rate)"),
        "causal_midband_delta_pts": kn([round(d, 3) for d in sorted(c_deltas)],
                                       src("step40/x25_captions.json")
                                       + " shift_vs_baseline prior-ablate arms, L16-28 mean"),
        "contrast_midband_delta_pts": kn([round(d, 3) for d in sorted(k_deltas)],
                                         src("step40/x25_captions.json")
                                         + " shift_vs_baseline m3id/vcd arms, L16-28 mean"),
        "contrast_unparsed_range": kn(
            [min(rows[a]["unparsed"] for a in broken),
             max(rows[a]["unparsed"] for a in broken)],
            src("step40/x25_pope.json") + " arms.*.pope.random.n_unparsed (m3id/vcd tau>=1)"),
        "clean_format_higher_chair_arms": kn(
            {a: round(rows[a]["chair_s"], 4) for a in clean_hi},
            src("step40/x25_captions.json")
            + " arms.*.captions.metrics.chair_s where chair_s > baseline (unparsed = 0)"),
    }


# ---------------------------------------------------------------- figE5: X26
def figE5():
    x26 = load("step40/x26_registers.json")
    layers = sorted(x26["probes"]["object_targets"]["per_layer"], key=int)
    xs = np.array([int(L) for L in layers])
    band = x26["workspace_band"]

    auc = {
        rt: [x26["probes"]["object_targets"]["per_layer"][L][rt]["mean_auc"]
             for L in layers]
        for rt in ("register", "image_nonregister", "text")
    }
    rate = {
        rt: [x26["probes"]["unembed_crosscheck_object_in_top10"][L][rt]
             for L in layers]
        for rt in ("register", "image_nonregister", "text")
    }
    z_reg = [x26["registers"]["per_layer"][L]["image"]["stickiness_z"] for L in layers]
    attn = x26["attention"]["aggregate"]
    corr = [attn[L]["mean_corr_stickiness_mass"] for L in layers]
    corr_mean = float(np.mean(corr))
    n_targets = x26["probes"]["object_targets"]["per_layer"][layers[0]]["register"]["n_targets"]

    series = [
        ("register", "register rows", BLUE),
        ("image_nonregister", "non-register image patches", ORANGE),
        ("text", "text rows", PURPLE),
    ]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.6))

    # -- left: ridge probes READ the object everywhere -------------------------
    for axx in (ax, ax2):
        axx.axvspan(band["band_lo"], band["band_hi"], color=SKY, alpha=0.12, zorder=1)
    ax.text(band["band_hi"], 0.45,
            f"workspace band L{band['band_lo']}-L{band['band_hi']} (D48)",
            transform=ax.get_xaxis_transform(),
            ha="right", va="bottom", fontsize=8.4, color=BLUE)

    for rt, lab, col in series:
        ax.plot(xs, auc[rt], "-o", color=col, lw=2.0, ms=5.5, zorder=4, label=lab)
        ax.text(xs[-1] + 0.25, auc[rt][-1], f"{auc[rt][-1]:.3f}",
                va="center", fontsize=8.6, color=col, fontweight="bold")
    ax.axhline(0.5, ls=":", lw=1.2, color=GREY, zorder=3)
    ax.text(xs[-1], 0.505, "chance 0.5", fontsize=8.5, color=GREY,
            va="bottom", ha="right")

    all_auc = [v for rt in auc for v in auc[rt]]
    ax.text(
        0.02, 0.03,
        "PROBE-READABLE, UNEMBED-BLIND\n"
        "(E3 verdict, D50): a ridge probe reads GT-object\n"
        f"presence at AUC {min(all_auc):.3f}-{max(all_auc):.3f} ({n_targets} targets,\n"
        "chance 0.5) from ALL row classes at every layer -\n"
        f"including register rows ({auc['register'][0]:.3f}-"
        f"{auc['register'][-1]:.3f}).\n"
        "The lens/unembed readout at the same positions stays\n"
        f"object-free on registers (\u2264{max(rate['register']) * 100:.1f}% in top-10\n"
        f"at every layer) while patches/text reach "
        f"{max(rate['image_nonregister']) * 100:.1f}%/\n"
        f"{max(rate['text']) * 100:.1f}% - a trained head can read what\n"
        "the lens cannot.",
        transform=ax.transAxes, va="bottom", fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.4", fc="0.95", ec="0.7", alpha=0.95),
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([f"L{L}" for L in xs])
    ax.set_ylim(0.44, 0.86)
    ax.set_xlabel("lens layer")
    ax.set_ylabel("ridge-probe AUROC, GT-object presence")
    ax.set_title(
        f"trained probe: object presence readable everywhere\n"
        f"({x26['probes']['n_usable_images']} images, leave-one-image-out folds)",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=8.6)

    # -- right: the unembed readout is register-blind ---------------------------
    for rt, lab, col in series:
        ax2.plot(xs, [v * 100 for v in rate[rt]], "-o", color=col, lw=2.0, ms=5.5,
                 zorder=4, label=lab)
    for rt, _, col in series:
        va = {"register": "center", "image_nonregister": "top", "text": "bottom"}[rt]
        ax2.text(xs[-1] + 0.25, rate[rt][-1] * 100, f"{rate[rt][-1] * 100:.1f}%",
                 va=va, fontsize=8.6, color=col, fontweight="bold")
    ax2.text(xs[2], rate["text"][2] * 100 + 0.55, f"{rate['text'][2] * 100:.1f}%",
             ha="center", fontsize=8.4, color=PURPLE)
    ax2.plot(xs, z_reg, "--s", color=GREY, lw=1.4, ms=4.5, zorder=3,
             label="register stickiness z (right axis)")
    ax2.set_ylim(-0.4, 10.2)
    ax2.set_xlim(xs[0] - 1.0, xs[-1] + 2.6)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"L{L}" for L in xs])
    ax2.set_xlabel("lens layer")
    ax2.set_ylabel("unembed object-in-top10 rate (%)")
    ax2.set_title(
        "unembed readout: register rows flat at ~0\nwhile patches/text rise through the band",
        fontsize=10,
    )

    ax3 = ax2.twinx()
    ax3.set_ylim(min(z_reg) * 0.8, max(z_reg) * 1.15)
    ax3.set_ylabel("stickiness z", color=GREY)
    ax3.tick_params(axis="y", colors=GREY)
    ax3.spines["right"].set_visible(True)
    ax3.grid(False)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.4)
    ax2.text(
        0.02, 0.62,
        f"register rows: 0.0-{max(rate['register']) * 100:.1f}% at EVERY layer\n"
        f"(patches 0-{max(rate['image_nonregister']) * 100:.1f}%, "
        f"text 0-{max(rate['text']) * 100:.1f}%)\n"
        f"grey dashed: the register states are sticky\n"
        f"(z {min(z_reg):.1f}-{max(z_reg):.1f}) and attention-cold (r = {corr_mean:.2f}, "
        f"range {min(corr):.2f}..{max(corr):.2f})",
        transform=ax2.transAxes, va="top", fontsize=8.4,
        bbox=dict(boxstyle="round,pad=0.35", fc="0.95", ec="0.7", alpha=0.92),
    )

    fig.suptitle(
        "X26 visual registers (LLaVA-1.5-7B, 20 images): register states are "
        "probe-readable but unembed-blind -\nthe blindness is the output projection's row "
        "space, not the representation, so a trained head can read what the lens cannot",
        fontsize=11.5, y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIGS / "figE5_register_blindness.png", **SAVE_KW)
    plt.close(fig)

    return {
        "n_images": kn(x26["n_samples"], src("step40/x26_registers.json") + " n_samples"),
        "n_probe_targets": kn(n_targets,
                              src("step40/x26_registers.json")
                              + " probes.object_targets.n_targets"),
        "workspace_band": kn([band["band_lo"], band["band_hi"]],
                             src("step40/x26_registers.json")
                             + " workspace_band (band_lo, band_hi)"),
        "probe_auc_by_layer": kn(
            {f"L{L}": {rt: round(v, 4) for rt, v in
                       zip(("register", "image_nonregister", "text"),
                           (auc["register"][i], auc["image_nonregister"][i], auc["text"][i]),
                           strict=False)}
             for i, L in enumerate(layers)},
            src("step40/x26_registers.json")
            + " probes.object_targets.per_layer[L].*.mean_auc"),
        "unembed_object_in_top10_by_layer": kn(
            {f"L{L}": {rt: round(rate[rt][i], 4) for rt in rate}
             for i, L in enumerate(layers)},
            src("step40/x26_registers.json")
            + " probes.unembed_crosscheck_object_in_top10[L]"),
        "register_stickiness_z_by_layer": kn(
            {f"L{L}": round(z, 2) for L, z in zip(layers, z_reg, strict=False)},
            src("step40/x26_registers.json")
            + " registers.per_layer[L].image.stickiness_z"),
        "attention_stickiness_corr_by_layer": kn(
            {f"L{L}": round(v, 4) for L, v in zip(layers, corr, strict=False)},
            src("step40/x26_registers.json")
            + " attention.aggregate[L].mean_corr_stickiness_mass"),
        "attention_stickiness_corr_mean": kn(
            round(corr_mean, 4),
            "mean over L" + "-L".join([layers[0], layers[-1]]) + " of "
            + src("step40/x26_registers.json")
            + " attention.aggregate[L].mean_corr_stickiness_mass"),
    }


def main():
    print("figures_e_causal.py: regenerating figE4..figE5 + metrics")
    x25_pope = load("step40/x25_pope.json")
    x25_cap = load("step40/x25_captions.json")
    rows = arm_table(x25_pope, x25_cap)

    k4 = figE4()
    print("  wrote figures/figE4_mitigation_frontier.png")
    causal = [a for a in rows if rows[a]["family"] == "prior-ablate"]
    print(
        f"figE4 summary: baseline acc {pct1(rows['baseline']['acc'])}% / CHAIR-s "
        f"{pct1(rows['baseline']['chair_s'])}%; "
        + "; ".join(
            f"{a} acc {pct1(rows[a]['acc'])}% unparsed {rows[a]['unparsed']}/60 "
            f"CHAIR-s {pct1(rows[a]['chair_s'])}% mid-band "
            f"{rows[a]['midband_delta'] * 100:+.2f} pts"
            for a in causal + ["m3id_tau_0.5", "m3id_tau_1", "m3id_tau_2",
                               "vcd_tau_0.5", "vcd_tau_1", "vcd_tau_2"]
        )
    )

    k5 = figE5()
    print("  wrote figures/figE5_register_blindness.png")
    x26 = load("step40/x26_registers.json")
    layers = sorted(x26["probes"]["object_targets"]["per_layer"], key=int)
    print(
        "figE5 summary: probe AUC register/patch/text "
        + "; ".join(
            f"L{L} {x26['probes']['object_targets']['per_layer'][L]['register']['mean_auc']:.3f}/"
            f"{x26['probes']['object_targets']['per_layer'][L]['image_nonregister']['mean_auc']:.3f}/"
            f"{x26['probes']['object_targets']['per_layer'][L]['text']['mean_auc']:.3f}"
            for L in layers)
        + " | unembed obj-in-top10 "
        + "; ".join(
            f"L{L} {x26['probes']['unembed_crosscheck_object_in_top10'][L]['register']:.4f}/"
            f"{x26['probes']['unembed_crosscheck_object_in_top10'][L]['image_nonregister']:.4f}/"
            f"{x26['probes']['unembed_crosscheck_object_in_top10'][L]['text']:.4f}"
            for L in layers)
        + f" | register z {min(x26['registers']['per_layer'][L]['image']['stickiness_z'] for L in layers):.1f}-"
        f"{max(x26['registers']['per_layer'][L]['image']['stickiness_z'] for L in layers):.1f}"
    )

    metrics = {
        "round": "validation_2026-10-01",
        "family": "extension-round figures (mitigation frontier, register blindness)",
        "script": "results/validation_2026-10-01/code/figures_e_causal.py",
        "figures": [
            {
                "file": "figE4_mitigation_frontier.png",
                "sources": [
                    src("step40/x25_pope.json"),
                    src("step40/x25_captions.json"),
                    "DECISIONS.md D49 (x25 mitigation microscope verdict)",
                ],
                "key_numbers": k4,
                "finding": (
                    "Mitigation frontier on LLaVA-1.5-7B (9 arms, POPE-random n=60 + "
                    "CHAIR over the caption images): the J-lens causal arm (prior "
                    "ablation of the lens-measured prior direction at L20) is the only "
                    "arm that improves POPE accuracy while leaving the caption metric "
                    "and the mid-band workspace content untouched; M3ID/VCD pay with "
                    "broken answer format (unparsed) or with mid-band object content."
                ),
            },
            {
                "file": "figE5_register_blindness.png",
                "sources": [
                    src("step40/x26_registers.json"),
                    "DECISIONS.md D50 (E3 verdict: blindness is the unembed's row space)",
                ],
                "key_numbers": k5,
                "finding": (
                    "Visual registers are probe-readable but unembed-blind: a ridge "
                    "probe decodes GT-object presence from register rows (and patches/"
                    "text) at every layer while the lens/unembed top-10 stays "
                    "object-free on register positions; the transport preserves the "
                    "content for a probe (E3), so the blindness is the output "
                    "projection's row space, not the representation."
                ),
            },
        ],
        "notes": [
            "All plotted numbers are read from the raw JSONs at run time; the only "
            "DECISIONS.md-sourced value is the E3 verdict phrasing (D50), cited in "
            "place.",
            "figE4 mid-band delta = mean over the teacher-forced lens layers (16, 20, "
            "24, 28) of x25_captions lens_trajectory.shift_vs_baseline.object_topk_rate "
            "(paired same-images shift; baseline is 0 by definition and has no entry).",
            "figE4 marker shape encodes POPE unparsed outputs (n_unparsed > 0 -> "
            "square); coincident (accuracy, CHAIR-s) points get a deterministic jitter "
            "from np.random.default_rng(0).",
            "figE5 register band = x26 workspace_band (band_lo..band_hi); stickiness z "
            "is the register (image) class from registers.per_layer; the unembed "
            "crosscheck object-in-top10 rates equal registers.per_layer rates (asserted "
            "by construction in x26).",
            "Deterministic: fixed layout, fixed jitter seed, no timestamps; re-running "
            "regenerates byte-identical PNGs + metrics JSON.",
        ],
    }
    with open(SUMMARY / "metrics_e_causal.json", "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    print("  wrote summary/metrics_e_causal.json")


if __name__ == "__main__":
    main()
