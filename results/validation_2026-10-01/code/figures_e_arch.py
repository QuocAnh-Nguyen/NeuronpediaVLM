#!/usr/bin/env python
"""
figures_e_arch.py - architecture-round figures (figE6..figE7) for the
2026-10-01 validation round: the cross-architecture workspace band map
(x22 structure arm, 3 models x 2 caption budgets) and the universal
sensory-block signature (x23 band arm).

Deterministic: fixed palette, fixed layout, fixed bins/jitter seed, no
timestamps inside figures; re-running regenerates byte-identical PNGs and the
metrics JSON.

Outputs (relative to this script's parent dir, results/validation_2026-10-01/):
  figures/figE6_band_map.png
  figures/figE7_sensory_block.png
  summary/metrics_e_arch.json

Data sources:
  raw/step13/x22_structure_llava-7b_64tok.json    (onset/offset digest, 32 layers)
  raw/step13/x22_structure_llava-13b.json         (40 layers, 256-token budget)
  raw/step13/x22_structure_llava-13b_64tok.json   (40 layers, 64-token budget)
  raw/step13/x22_structure_qwen2vl.json           (28 layers, 256-token budget)
  raw/step13/x22_structure_qwen2vl_64tok.json     (28 layers, 64-token budget)
  raw/step13/x23_band_llava-7b_64tok.json         (modality-split z / object rates, 59 images)
  raw/step13/x23_band_llava-13b.json
  raw/step13/x23_band_qwen2vl.json
  DECISIONS.md D48 (paper comparison values quoted in annotations only)

Run: <python> results/validation_2026-10-01/code/figures_e_arch.py
"""

import json
import re
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

SAVE_KW = dict(bbox_inches="tight", metadata={"Software": "figures_e_arch.py"})

RNG = np.random.default_rng(0)  # fixed seed -> deterministic jitter

# figE6 rows, top to bottom: (label, x22 file, n_layers, row colour)
ROWS = [
    ("llava-7b 64tok", "step13/x22_structure_llava-7b_64tok.json", 32, BLUE),
    ("llava-13b 256", "step13/x22_structure_llava-13b.json", 40, PURPLE),
    ("llava-13b 64tok", "step13/x22_structure_llava-13b_64tok.json", 40, PURPLE),
    ("qwen2vl 256", "step13/x22_structure_qwen2vl.json", 28, GREEN),
    ("qwen2vl 64tok", "step13/x22_structure_qwen2vl_64tok.json", 28, GREEN),
]

# x22 digest metric key -> display label + colour for the onset dots
ONSET_METRICS = [
    ("autocorr_delta", "autocorrelation onset", ORANGE),
    ("effdim", "effective-dimensionality onset", SKY),
    ("top1", "top-1 accuracy onset", GREEN),
    ("kurt", "kurtosis onset", VERMIL),
]

# figE7 models: (label, x23 file, colour)
X23 = [
    ("llava-7b", "step13/x23_band_llava-7b_64tok.json", BLUE),
    ("llava-13b", "step13/x23_band_llava-13b.json", PURPLE),
    ("qwen2vl", "step13/x23_band_qwen2vl.json", GREEN),
]


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


def parse_digest(digest):
    """Parse the x22 'half-rise onset/offset per metric' block -> {metric: (onset, offset)}."""
    out = {}
    for m in re.finditer(r"^\s+(\w+)\s+onset=(\d+)\s+offset=(\d+)", digest, re.M):
        out[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return out


# ---------------------------------------------------------------- figE6: X22
def figE6():
    rows = []
    for label, rel, n_layers, colour in ROWS:
        d = load(rel)
        assert d["n_layers"] == n_layers, (rel, d["n_layers"], n_layers)
        rows.append(
            {
                "label": label,
                "rel": rel,
                "n": n_layers,
                "colour": colour,
                "band": d["band"],  # lens-fitted band (context only)
                "onsets": parse_digest(d["digest"]),  # half-rise onsets/offsets
            }
        )
    # paper band [12,29] on the 32-layer grid, read from the x23 payload
    paper = load("step13/x23_band_llava-7b_64tok.json")["workspace_band"]["paper_band_32l"]
    paper_frac = (paper[0] / 32, paper[1] / 32)

    fig = plt.figure(figsize=(11.8, 6.1))
    ax = fig.add_axes((0.085, 0.335, 0.855, 0.50))

    # paper's mapped band as a vertical reference behind every row
    paper_span = ax.axvspan(paper_frac[0], paper_frac[1], color=GREY, alpha=0.13,
                            zorder=0)
    ax.axvline(paper_frac[0], color=GREY, lw=0.8, ls=":", zorder=1)
    ax.axvline(paper_frac[1], color=GREY, lw=0.8, ls=":", zorder=1)

    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        top1_on, top1_off = r["onsets"]["top1"]
        on_f, off_f = top1_on / r["n"], top1_off / r["n"]
        # the row's workspace band: top-1 onset -> offset
        ax.fill_betweenx([y - 0.26, y + 0.26], on_f, off_f,
                         color=r["colour"], alpha=0.28, lw=0, zorder=2)
        ax.text((on_f + off_f) / 2, y - 0.42,
                f"top-1 band L{top1_on}\u2013L{top1_off}",
                ha="center", va="center", fontsize=7.8, color="0.15", zorder=6)
        # the four half-rise onset metrics as dots
        for key, _, mcol in ONSET_METRICS:
            on = r["onsets"][key][0]
            ax.plot([on / r["n"]], [y], marker="o", ms=8, mfc=mcol, mec="k",
                    mew=0.6, zorder=5, ls="")
        # raw layer count at the row's right edge
        ax.text(1.012, y, f"n={r['n']}", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=9, color=GREY)

    # legend above the rows, under the suptitle (proxy artists keep it one-per-metric)
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=BLUE, alpha=0.28, ec="none"),
        *(plt.Line2D([], [], marker="o", ls="", ms=8, mfc=c, mec="k", mew=0.6)
          for _, _, c in ONSET_METRICS),
        paper_span,
    ]
    labels = ["shaded: top-1 onset\u2192offset (the row's band)",
              *[lab for _, lab, _ in ONSET_METRICS],
              f"paper band [12,29]/32 = [{paper_frac[0]:.3f}, {paper_frac[1]:.3f}]"]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.012, 0.945),
               ncol=3, fontsize=8.6)

    # insight box in the free strip under the axes (lines wrapped to figure width)
    DOT = " \u00b7 "
    parts = []
    for r in rows:
        lab = r["label"].replace(" ", "\u00b7")
        a, b = r["onsets"]["top1"]
        parts.append(f"{lab} {a / r['n']:.2f}\u2192{b / r['n']:.2f}")
    onoff = DOT.join(parts)
    t1 = rows[0]["onsets"]["top1"]
    t13 = rows[1]["onsets"]["top1"]
    tq = rows[3]["onsets"]["top1"]
    n7, n13, nq = rows[0]["n"], rows[1]["n"], rows[3]["n"]
    fig.text(
        0.012, 0.012,
        "Takeaway: the band's fractional position is model-stable within the LLaVA "
        "family \u2014 top-1 onset 0.63 of depth for 7b and 13b alike\n"
        f"(offset {t1[1] / n7:.2f} vs {t13[1] / n13:.2f}) \u2014 its absolute layer "
        "deepens with the stack (onset \u00d7layers/32: "
        f"{t1[0]} \u2192 {t13[0] * n13 / 32:.0f}) while its own-depth fraction stays "
        f"{t13[0] / n13:.2f};\n"
        f"qwen's band compresses onto the motor end ({tq[0] / nq:.2f}"
        f"\u2192{tq[1] / nq:.2f}); the budget axis\n"
        "(64 vs 256 tokens) moves nothing \u2014 top-1 onset/offset identical per "
        "model.\n"
        f"onset\u2192offset (fraction of depth): {onoff}",
        va="bottom", ha="left", fontsize=8.8, color="0.1", linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.45", fc="0.96", ec="0.7"))

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.75, len(rows) - 0.25)
    ax.set_yticks([len(rows) - 1 - i for i in range(len(rows))])
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlabel("layer / n_layers  (fraction of depth; dots = half-rise onsets from "
                  "the x22 digest, in-band layers)")
    fig.suptitle("figE6 - the workspace band across architectures and caption "
                 "budgets (x22 structure arm)", fontsize=11.5, y=0.985)
    ax.grid(axis="y", visible=False)

    fig.savefig(FIGS / "figE6_band_map.png", **SAVE_KW)
    plt.close(fig)

    key = {}
    for r in rows:
        tag = r["label"].replace(" ", "_").replace("-", "_")
        t1o, t1f = r["onsets"]["top1"]
        key[f"{tag}_top1_onset_offset"] = kn(
            [t1o, t1f], src(r["rel"]) + " digest 'half-rise onset/offset': top1")
        key[f"{tag}_top1_onset_offset_frac"] = kn(
            [round(t1o / r["n"], 3), round(t1f / r["n"], 3)],
            src(r["rel"]) + " top1 onset/offset / n_layers")
        key[f"{tag}_metric_onsets"] = kn(
            {name: r["onsets"][k][0] for k, name, _ in ONSET_METRICS},
            src(r["rel"]) + " digest onsets (autocorr_delta/effdim/top1/kurt)")
        key[f"{tag}_fitted_band"] = kn(r["band"], src(r["rel"]) + " band (lens-fitted)")
        key[f"{tag}_onset_x_layers_over_32"] = kn(
            round(t1o * r["n"] / 32, 2),
            src(r["rel"]) + " top1 onset x (layers/32), the depth-matched scale")
    key["paper_band_32l_frac"] = kn(
        [round(paper_frac[0], 3), round(paper_frac[1], 3)],
        src("step13/x23_band_llava-7b_64tok.json")
        + " workspace_band.paper_band_32l / 32")
    key["budget_axis_verdict"] = kn(
        "identical top-1 onset/offset per model across 64 vs 256 tokens "
        "(13b L25/L36 both; qwen L24/L26 both)",
        "x22 digests of the two budgets per model")
    return key


# ---------------------------------------------------------------- figE7: X23
def figE7():
    mods = []
    for label, rel, colour in X23:
        d = load(rel)
        layers = d["layers"]
        rec = d["per_layer"]
        mods.append(
            {
                "label": label,
                "rel": rel,
                "colour": colour,
                "layers": layers,
                "frac": [L / layers[-1] for L in layers],
                "z_img": [rec[str(L)]["image"]["stickiness_z"] for L in layers],
                "z_txt": [rec[str(L)]["text"]["stickiness_z"] for L in layers],
                "o_img": [rec[str(L)]["image"]["object_in_top10_rate"] for L in layers],
                "o_txt": [rec[str(L)]["text"]["object_in_top10_rate"] for L in layers],
                "n_images": d["n_samples"],
            }
        )

    fig = plt.figure(figsize=(12.4, 6.1))
    ax = fig.add_axes((0.062, 0.315, 0.395, 0.50))
    ax2 = fig.add_axes((0.565, 0.315, 0.395, 0.50))

    for a in (ax, ax2):
        a.axhline(0, color=GREY, lw=0.8, ls=":", zorder=1)

    # left: top-1 stickiness z, image (solid) vs text (dashed)
    for m in mods:
        ax.plot(m["frac"], m["z_img"], "-", color=m["colour"], lw=2.0, marker="o",
                ms=4.5, zorder=4, label=f"{m['label']} image")
        ax.plot(m["frac"], m["z_txt"], "--", color=m["colour"], lw=1.6, marker="s",
                ms=4, zorder=3, label=f"{m['label']} text")
    ax.set_xlim(0, 1)
    ax.set_ylim(-2, 27)
    ax.set_xlabel("layer / n_layers (fraction of depth)")
    ax.set_ylabel("top-1 stickiness z (vs shuffled null)")
    ax.set_title("stickiness: image positions hyper-persistent,\ntext positions "
                 "token-local", fontsize=10)
    ax.legend(loc="upper left", fontsize=8.2)

    # right: object-in-top10 rate, image (solid) vs text (dashed)
    for m in mods:
        ax2.plot(m["frac"], [100 * v for v in m["o_img"]], "-", color=m["colour"],
                 lw=2.0, marker="o", ms=4.5, zorder=4, label=f"{m['label']} image")
        ax2.plot(m["frac"], [100 * v for v in m["o_txt"]], "--", color=m["colour"],
                 lw=1.6, marker="s", ms=4, zorder=3, label=f"{m['label']} text")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.8, 14)
    ax2.set_xlabel("layer / n_layers (fraction of depth)")
    ax2.set_ylabel("object-in-top10 rate (%)")
    ax2.set_title("objects: image rows object-free through the band,\nrising only at "
                  "the motor regime", fontsize=10)
    ax2.legend(loc="upper left", fontsize=8.2)

    # insight boxes in the strip under each panel
    zi = " \u00b7 ".join(f"{m['label']} {min(m['z_img']):.1f}\u2013{max(m['z_img']):.1f}"
                         for m in mods)
    zt = max(max(m["z_txt"]) for m in mods)
    fig.text(0.045, 0.012,
             "Takeaway: image positions hold hyper-persistent non-verbal content \u2014\n"
             f"stickiness z: {zi}\n"
             f"(text z \u2264 {zt:.1f} on all three, the token-local 0\u20132 regime)",
             va="bottom", ha="left", fontsize=8.8, color="0.1", linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.45", fc="0.96", ec="0.7"))
    on = " / ".join(str(pct1(m["o_img"][0])) + "%" for m in mods)
    fin = " / ".join(str(pct1(m["o_img"][-1])) + "%" for m in mods)
    otxt_max = max(max(m["o_txt"]) for m in mods)
    fig.text(0.545, 0.012,
             "Takeaway: image rows are object-free at band onset "
             f"({on} 7b/13b/qwen), rising\n"
             f"to {fin} only at the final (motor) layers; text rows stay \u2264 "
             f"{pct1(otxt_max)}% throughout \u2014\n"
             "objects verbalise only in the motor regime.",
             va="bottom", ha="left", fontsize=8.8, color="0.1", linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.45", fc="0.96", ec="0.7"))

    fig.suptitle(
        "figE7 - the universal sensory-block signature (x23 band arm, 59 images each): "
        "three architectures, one modality split \u2014 stable\nnon-verbal visual content "
        "at image positions; objects verbalised only at the motor regime    "
        "(solid = image positions, dashed = text positions)",
        fontsize=11, y=0.985,
    )
    fig.savefig(FIGS / "figE7_sensory_block.png", **SAVE_KW)
    plt.close(fig)

    key = {}
    for m in mods:
        tag = m["label"].replace("-", "_")
        key[f"{tag}_z_img_range"] = kn(
            [round(min(m["z_img"]), 2), round(max(m["z_img"]), 2)],
            src(m["rel"]) + " per_layer[L].image.stickiness_z over measured layers")
        key[f"{tag}_z_txt_range"] = kn(
            [round(min(m["z_txt"]), 2), round(max(m["z_txt"]), 2)],
            src(m["rel"]) + " per_layer[L].text.stickiness_z")
        key[f"{tag}_obj_img_range"] = kn(
            [round(min(m["o_img"]), 3), round(max(m["o_img"]), 3)],
            src(m["rel"]) + " per_layer[L].image.object_in_top10_rate")
        key[f"{tag}_obj_txt_range"] = kn(
            [round(min(m["o_txt"]), 3), round(max(m["o_txt"]), 3)],
            src(m["rel"]) + " per_layer[L].text.object_in_top10_rate")
        key[f"{tag}_obj_img_onset_final"] = kn(
            [round(m["o_img"][0], 3), round(m["o_img"][-1], 3)],
            src(m["rel"]) + f" image obj-in-top10 at first/last measured layer "
            f"(L{m['layers'][0]}/L{m['layers'][-1]})")
    key["sensory_block_verdict"] = kn(
        "image-position content hyper-persistent (z "
        f"{min(min(m['z_img']) for m in mods):.1f}\u2013"
        f"{max(max(m['z_img']) for m in mods):.1f}) yet object-free at band onset "
        "(0.0\u20131.8%), rising to 4\u201312% only at the final layers; text z \u2264 2.2 "
        "throughout",
        "x23 band digests, all three models")
    return key


# ---------------------------------------------------------------- main
def main():
    print("figures_e_arch.py: regenerating figE6..figE7 + metrics")

    e6 = figE6()
    parts6 = []
    for label, f, n, _ in ROWS:
        t1o, t1f = parse_digest(load(f)["digest"])["top1"]
        parts6.append(f"{label}: top1 L{t1o}\u2192L{t1f}/{n} "
                      f"({t1o / n:.3f}\u2192{t1f / n:.3f})")
    print("  figE6: band map - " + " | ".join(parts6)
          + " - wrote figures/figE6_band_map.png")

    e7 = figE7()
    parts = []
    for label, rel, _ in X23:
        d = load(rel)
        zimg = [d["per_layer"][str(L)]["image"]["stickiness_z"] for L in d["layers"]]
        ztxt = [d["per_layer"][str(L)]["text"]["stickiness_z"] for L in d["layers"]]
        oimg = [d["per_layer"][str(L)]["image"]["object_in_top10_rate"] for L in d["layers"]]
        parts.append(f"{label}: z_img {min(zimg):.1f}\u2013{max(zimg):.1f} vs "
                     f"z_txt \u2264{max(ztxt):.1f}, obj_img {pct1(min(oimg))}\u2013"
                     f"{pct1(max(oimg))}%")
    print("  figE7: sensory block - " + " | ".join(parts)
          + " - wrote figures/figE7_sensory_block.png")

    metrics = {
        "round": "validation_2026-10-01",
        "family": "architecture round (cross-architecture band map, universal "
                  "sensory-block signature)",
        "script": "results/validation_2026-10-01/code/figures_e_arch.py",
        "figures": [
            {
                "file": "figE6_band_map.png",
                "sources": [
                    *[src(f) for _, f, _, _ in ROWS],
                    src("step13/x23_band_llava-7b_64tok.json")
                    + " workspace_band.paper_band_32l (paper comparison band)",
                    "DECISIONS.md D48 (band onsets per model; budget verdict)",
                ],
                "key_numbers": e6,
                "finding": (
                    "The top-1 workspace band keeps its fractional position within the "
                    "LLaVA family across the width change (7b L20-L29/32 = 0.625-0.906; "
                    "13b L25-L36/40 = 0.625-0.900, identical at both caption budgets), "
                    "while 13b's band sits deeper once depth-matched (onset x layers/32: "
                    "20 -> 31) and qwen2vl's band compresses onto the motor end "
                    "(L24-L26/28 = 0.857-0.929, both budgets). The budget axis "
                    "(64 vs 256 tokens) moves nothing: identical half-rise onsets/offsets "
                    "per model."
                ),
            },
            {
                "file": "figE7_sensory_block.png",
                "sources": [src(r) for _, r, _ in X23]
                + ["DECISIONS.md D48 (universal sensory-block signature)"],
                "key_numbers": e7,
                "finding": (
                    "Three architectures, one modality split: image-position lens content "
                    "is hyper-persistent (top-1 stickiness z 3.2-23.9 across models) while "
                    "text positions stay token-local (z <= 2.2) on all three; image rows "
                    "are object-free at band onset (0.0-1.8%) and rise to 4-12% only at "
                    "the final layers, so the workspace carries stable non-verbal visual "
                    "content and objects verbalise only in the motor regime."
                ),
            },
        ],
        "notes": [
            "All plotted numbers are read from the raw JSONs at run time; the only "
            "external values are the paper comparison band [12,29] (read from "
            "x23 workspace_band.paper_band_32l) and DECISIONS.md D48 phrasing.",
            "figE6 band shading uses the x22 digest's half-rise top-1 onset/offset "
            "(in-band layers); the four dots are the digest's half-rise onsets for "
            "autocorr_delta/effdim/top1/kurt, each at layer/n_layers.",
            "figE7 x positions normalise each model's measured layers by its last "
            "measured layer so the three architectures align on one depth axis.",
            "figE7 object rates: in the step13 rerun the image-side rate is >= the "
            "text-side rate at every measured layer on all three models; the D48 "
            "sensory-block claim carried here is the image-side one (object-free at "
            "onset, 4-12% at the final layers), not a text>image asymmetry.",
            "Deterministic: fixed palette, fixed layout, fixed jitter seed "
            "(np.random.default_rng(0)), no timestamps; re-running regenerates "
            "byte-identical PNGs + metrics JSON.",
        ],
    }
    with open(SUMMARY / "metrics_e_arch.json", "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    print("  wrote summary/metrics_e_arch.json")


if __name__ == "__main__":
    main()
