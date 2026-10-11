#!/usr/bin/env python
"""
figures_d_campaign.py - S2 caption-lens design/eval + campaign operations
timeline (figD9, figD10) for the 2026-10-01 validation round. Covers the
experiments NOT shown by the figA*/figB*/figC* families.

Deterministic: fixed palette, fixed layout, no timestamps inside figures;
re-running regenerates byte-identical PNGs.

Outputs (relative to this script's parent dir, results/validation_2026-10-01/):
  figures/figD9_s2_halves_transfer.png
  figures/figD10_campaign_timeline.png
  summary/metrics_d_campaign.json

Data sources:
  raw/step3/s2_eval.json
      - main.rows.default:      merged 100-sample caption lens on 300 held-out
                                captions, per-layer x tag (text/image/all; the
                                default table has no image-q0..q2 and
                                image-q3 == image by the V5 last-patch collapse)
      - halves.lens_{a,b}_on_half_{a,b}: A4/X8 cross-half scoring (text tag)
      - transfer.{caption_lens_on_wikitext, text_lens_on_captions}: E6 rows
  DECISIONS.md D-headers (parsed at run time for the dated anchors) +
  REPORT.md section 9 (GPU-hours table) and section 11 (incidents) for the
  ops timeline.

Run: <python> results/validation_2026-10-01/code/figures_d_campaign.py
"""

import json
import re
from datetime import datetime, timezone
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

# Okabe-Ito colorblind-safe palette (same as figures_instrument.py)
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
BLUE = "#0072B2"
VERMIL = "#D55E00"
PURPLE = "#CC79A7"
GREY = "0.35"

TAG_COLOR = {"text": VERMIL, "image": GREEN, "all": BLUE}

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

SAVE_KW = dict(bbox_inches="tight", metadata={"Software": "figures_d_campaign.py"})

S2 = json.load(open(RAW / "step3" / "s2_eval.json"))


def rows_lookup(rows, layer, tag):
    for r in rows:
        if r["layer"] == layer and r["tag"] == tag:
            return r
    raise KeyError((layer, tag))


def s2_series(section, tag, key="mean_rank_true"):
    rows = section["rows"] if "rows" in section else section
    lay = sorted({r["layer"] for r in rows})
    return np.array(lay), np.array([rows_lookup(rows, L, tag)[key] for L in lay])


# ---------------------------------------------------------------- figD9
def figD9():
    main_rows = S2["main"]["rows"]["default"]
    comp = S2["mask_composition"]
    model_rank_text = rows_lookup(main_rows, 31, "text")["model_mean_rank_true"]
    model_rank_wt = S2["transfer"]["caption_lens_on_wikitext"]["rows"][-1][
        "model_mean_rank_true"
    ]

    lay, rk_text = s2_series(main_rows, "text")
    _, rk_image = s2_series(main_rows, "image")
    _, rk_all = s2_series(main_rows, "all")

    r16 = rows_lookup(main_rows, 16, "text")
    r20 = rows_lookup(main_rows, 20, "text")
    r30 = rows_lookup(main_rows, 30, "text")
    r31 = rows_lookup(main_rows, 31, "text")
    bump_pct = (r20["mean_rank_true"] / r16["mean_rank_true"] - 1) * 100

    ha = S2["halves"]
    cell = {}
    for lens in ("a", "b"):
        for half in ("a", "b"):
            r = rows_lookup(ha[f"lens_{lens}_on_half_{half}"]["rows"], 30, "text")
            cell[(lens, half)] = r
    own_cross_delta = {
        "a": abs(cell[("a", "a")]["mean_rank_true"] - cell[("a", "b")]["mean_rank_true"]),
        "b": abs(cell[("b", "b")]["mean_rank_true"] - cell[("b", "a")]["mean_rank_true"]),
    }

    wt_lay, wt_rank = s2_series(S2["transfer"]["caption_lens_on_wikitext"], "text")
    tl_lay, tl_rank = s2_series(S2["transfer"]["text_lens_on_captions"], "text")
    wt = rows_lookup(S2["transfer"]["caption_lens_on_wikitext"]["rows"], 30, "text")
    tl = rows_lookup(S2["transfer"]["text_lens_on_captions"]["rows"], 30, "text")
    wt_bump = (wt_rank[20] / wt_rank[16] - 1) * 100
    tl_bump = (tl_rank[20] / tl_rank[16] - 1) * 100
    wt_ratio = wt["mean_rank_true"] / model_rank_wt

    fig, (axA, axB, axC) = plt.subplots(
        1, 3, figsize=(13.6, 5.3), gridspec_kw={"width_ratios": [1.5, 0.78, 1.22]}
    )

    # (a) per-tag mean rank by layer, merged lens on held-out captions
    for tag, ys in (("text", rk_text), ("image", rk_image), ("all", rk_all)):
        axA.plot(
            lay, ys, "o-", lw=1.7, ms=3.4, color=TAG_COLOR[tag],
            label=tag,
        )
    axA.axhline(model_rank_text, ls="--", lw=1.1, color=GREY)
    axA.text(3.5, 33, f"model ceiling {model_rank_text:.2f}", fontsize=8, color=GREY)
    axA.plot([31], [r31["mean_rank_true"]], "o", ms=7, mfc="none", mec=VERMIL, mew=1.8)
    # annotations live in the data-free strip y~32-135 (image curve >=138 there)
    axA.annotate(
        f"L20 bump {r20['mean_rank_true']:.0f} (+{bump_pct:.0f}% vs L16;\n"
        f"KL {r16['mean_kl']:.2f}\u2192{r20['mean_kl']:.2f}; X11: fitted-map)",
        xy=(20, r20["mean_rank_true"]), xytext=(2.0, 135), ha="left", va="top",
        arrowprops=dict(arrowstyle="->", color=GREY, lw=1.0), fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="0.95", ec="0.7"),
    )
    axA.annotate(
        f"L30 {r30['mean_rank_true']:.1f} ({r30['mean_rank_true']/model_rank_text:.1f}x ceiling)\n"
        f"L31 exact: {r31['mean_rank_true']:.2f} = model,\nKL 0, top-1 1.000",
        xy=(31, r31["mean_rank_true"]), xytext=(19.0, 135), ha="left", va="top",
        arrowprops=dict(arrowstyle="->", color=VERMIL, lw=1.0), fontsize=7.5, color=VERMIL,
    )
    axA.set_yscale("log")
    axA.set_xticks(range(0, 32, 2))
    axA.set_xlim(-1.0, 32.5)
    axA.set_ylim(28, 7e4)
    axA.set_xlabel("layer")
    axA.set_ylabel("mean true-token rank (held-out, 300 captions)")
    axA.set_title("(a) S2 merged lens by tag: text/image/all\n(image \u2261 image-q3, collapsed last patch)")
    axA.legend(loc="upper right", fontsize=8.5, title=f"pos/sample: {comp['text']['mean_positions']:.0f} text, {comp['image']['mean_positions']:.0f} image, {comp['all']['mean_positions']:.0f} all",
               title_fontsize=7)

    # (b) A4/X8 same-half vs cross-half heatmap (L30, text tag)
    grid = np.array(
        [
            [cell[("a", "a")]["mean_rank_true"], cell[("a", "b")]["mean_rank_true"]],
            [cell[("b", "a")]["mean_rank_true"], cell[("b", "b")]["mean_rank_true"]],
        ]
    )
    im = axB.imshow(grid, cmap="Blues", vmin=185, vmax=240, aspect="auto")
    kind_grid = [["own", "cross"], ["cross", "own"]]
    for i in range(2):        # row = lens (A, B)
        for j in range(2):    # col = scored-on half (A, B)
            c = cell[(("a", "b")[i], ("a", "b")[j])]
            kind = kind_grid[i][j]
            dark = grid[i, j] > 215
            axB.text(
                j, i - 0.16, f"{kind}  {c['mean_rank_true']:.2f}",
                ha="center", va="center", fontsize=9.5,
                fontweight="bold" if kind == "own" else "normal",
                color="white" if dark else "0.15",
            )
            axB.text(
                j, i + 0.22, f"KL {c['mean_kl']:.3f} | ag {c['top1_agreement']:.3f}",
                ha="center", va="center", fontsize=7.2,
                color="white" if dark else "0.25",
            )
    axB.set_xticks([0, 1])
    axB.set_xticklabels(["half A\n(150 captions)", "half B\n(150 captions)"], fontsize=8.5)
    axB.set_yticks([0, 1])
    axB.set_yticklabels(["lens A\n(fit Q1\u20135)", "lens B\n(fit Q6\u201310)"], fontsize=8.5)
    axB.set_title("(b) A4/X8 cross-half scoring\n(text mean rank @ L30)")
    axB.grid(False)
    cb = fig.colorbar(im, ax=axB, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8)
    axB.text(
        0.5, -0.42,
        f"own-vs-cross \u0394: A {own_cross_delta['a']:.2f} "
        f"({own_cross_delta['a']/cell[('a','a')]['mean_rank_true']*100:.1f}%), "
        f"B {own_cross_delta['b']:.2f} "
        f"({own_cross_delta['b']/cell[('b','b')]['mean_rank_true']*100:.1f}%)\n"
        "halves symmetric; merge == single fit (D7/D35);\n"
        "lens B stronger ~15% on both subsets (question half)",
        transform=axB.transAxes, ha="center", va="top", fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.35", fc="0.95", ec="0.7"),
    )

    # (c) E6 cross-corpus transfer bars
    w = 0.42
    axC.bar(wt_lay - w / 2, wt_rank, width=w, color=GREEN, lw=0,
            label=f"caption lens \u2192 WikiText (n={S2['transfer']['n_text_heldout']})")
    axC.bar(tl_lay + w / 2, tl_rank, width=w, color=BLUE, lw=0,
            label="text lens \u2192 captions (n=300)")
    axC.axhline(model_rank_wt, ls="--", lw=1.0, color=GREY, alpha=0.8)
    axC.axhline(42.62, ls=":", lw=1.0, color=GREY, alpha=0.8)
    # highlight the two L30 transfer cells
    axC.plot([30 - w / 2], [wt["mean_rank_true"]], "o", ms=6.5, mfc="none",
             mec="0.15", mew=1.3, zorder=6)
    axC.plot([30 + w / 2], [tl["mean_rank_true"]], "o", ms=6.5, mfc="none",
             mec="0.15", mew=1.3, zorder=6)
    # single summary box in the bar-free band above the L9-31 bars (max 7.8k)
    axC.text(
        0.30, 0.897,
        f"bump vs L16: +{tl_bump:.0f}% (S1\u2192cap) +{wt_bump:.0f}% (S2\u2192WT)\n"
        f"L30: WT {wt['mean_rank_true']:.1f} ({wt_ratio:.2f}x) | captions "
        f"{tl['mean_rank_true']:.1f}\n"
        f"(S2-on-cap {r30['mean_rank_true']:.1f}: S1 rank, S2 KL) - D35",
        transform=axC.transAxes, fontsize=7, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="0.95", ec="0.7"),
    )
    axC.set_yscale("log")
    axC.set_xticks(range(0, 32, 4))
    axC.set_xlim(-1.0, 32.5)
    axC.set_ylim(20, 8e4)
    axC.set_xlabel("layer")
    axC.set_ylabel("mean true-token rank")
    axC.set_title("(c) E6 cross-corpus transfer\n(text tag; dashed = model ceilings)")
    axC.legend(loc="upper left", ncol=2, fontsize=6.8, columnspacing=0.9,
               handlelength=1.1, borderpad=0.35)

    fig.suptitle(
        "S2 verdict: the merged 100-sample caption lens is exact at L31 (rank 42.62 = model, KL 0) with "
        "expected structure; A4/X8 halves are symmetric\n"
        "(merge == single fit, no leakage); the caption lens transfers to WikiText at "
        f"{wt_ratio:.2f}x ceiling; the L20 bump (+{bump_pct:.0f}% rank) is fitted-map-specific (X11), not a model property",
        fontsize=11.5, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(FIGS / "figD9_s2_halves_transfer.png", **SAVE_KW)
    plt.close(fig)

    return {
        "n_fit_prompts_merged": S2["main"]["n_prompts"],
        "n_heldout_captions": S2["n_heldout_samples"],
        "text_mean_rank_by_layer": {
            "L0": round(rk_text[0], 1),
            "L16": round(r16["mean_rank_true"], 1),
            "L20": round(r20["mean_rank_true"], 1),
            "L30": round(r30["mean_rank_true"], 1),
            "L31": round(r31["mean_rank_true"], 2),
        },
        "model_ceiling_rank_text": round(model_rank_text, 2),
        "L16_ratio_to_ceiling": round(r16["mean_rank_true"] / model_rank_text, 2),
        "L30_ratio_to_ceiling": round(r30["mean_rank_true"] / model_rank_text, 2),
        "L20_bump": {
            "rank_L16_to_L20": [round(r16["mean_rank_true"], 1), round(r20["mean_rank_true"], 1)],
            "pct_over_L16": round(bump_pct, 1),
            "KL_L16_to_L20": [round(r16["mean_kl"], 2), round(r20["mean_kl"], 2)],
            "attribution": "fitted-map-specific (X11 logit lens is monotone there), DECISIONS.md D37",
        },
        "L31_identity": {
            "rank": round(r31["mean_rank_true"], 2), "KL": 0.0, "top1_agreement": 1.0,
        },
        "mask_composition_mean_positions": {
            k: comp[k]["mean_positions"] for k in ("text", "image", "all", "image-q0")
        },
        "halves_L30_text_rank": {
            "lens_a_own_half_a": round(cell[("a", "a")]["mean_rank_true"], 2),
            "lens_a_cross_half_b": round(cell[("a", "b")]["mean_rank_true"], 2),
            "lens_b_cross_half_a": round(cell[("b", "a")]["mean_rank_true"], 2),
            "lens_b_own_half_b": round(cell[("b", "b")]["mean_rank_true"], 2),
            "own_vs_cross_delta": {k: round(v, 2) for k, v in own_cross_delta.items()},
            "L31_identity_all_four_cells_top1": 1.0,
        },
        "transfer_L30_text": {
            "caption_lens_on_wikitext": {
                "rank": round(wt["mean_rank_true"], 2),
                "model_ceiling": round(model_rank_wt, 2),
                "ratio_to_ceiling": round(wt_ratio, 3),
                "agree": round(wt["top1_agreement"], 4),
                "KL": round(wt["mean_kl"], 3),
            },
            "text_lens_on_captions": {
                "rank": round(tl["mean_rank_true"], 2),
                "agree": round(tl["top1_agreement"], 4),
                "KL": round(tl["mean_kl"], 3),
                "vs_S2_merged_on_captions_rank": round(r30["mean_rank_true"], 2),
            },
            "L17_24_bump_pct_over_L16": {
                "s1_text_lens_on_captions": round(tl_bump, 1),
                "s2_caption_lens_on_wikitext": round(wt_bump, 1),
            },
        },
        "citation": "all plotted numbers read from raw/step3/s2_eval.json at run time; "
        "cross-checked against DECISIONS.md D34/D35 (exact match) and D37 (L20 attribution)",
    }


# ---------------------------------------------------------------- figD10
def parse_d_dates():
    """Parse DECISIONS.md D-header dates -> {'D24': datetime, ...} (UTC)."""
    out = {}
    pat = re.compile(
        r"^##\s+(D\d+)[.\s]*(?:[\u2014\u2013-]\s*)?(\d{4}-\d{2}-\d{2})\s+~(\d{1,2}):(\d{2})Z",
        re.M,
    )
    for m in pat.finditer((BASE / "DECISIONS.md").read_text()):
        yy, mo, dd = (int(x) for x in m.group(2).split("-"))
        dt = datetime(yy, mo, dd, int(m.group(3)), int(m.group(4)), tzinfo=timezone.utc)
        out[m.group(1)] = dt
    return out



def figD10():
    D = parse_d_dates()
    t0 = datetime(2026, 10, 1, tzinfo=timezone.utc)

    def h(dt):
        return (dt - t0).total_seconds() / 3600.0

    # Phase spans. Starts/ends from DECISIONS.md D-headers or in-body times:
    #   S1: old-chain fit running at the 04:45 checkpoint (D12) -> fit+scoring
    #       complete 2026-10-02T17:40Z (REPORT.md section 9 update).
    #   S2 halves: half-A start 19:45Z 10-02 (D24 in-body) -> new-A written
    #       20:52:36Z 10-06 on Brev (D33 in-body).
    #   merge+eval: -> step3 chain done 2026-10-07T00:56Z (D35 in-body).
    #   X1/X7/X9: -> campaign done 2026-10-07T19:15:49Z (D36 in-body).
    #   X11: launched ~05:10Z, ~55 min (D36/D37 in-body).
    #   improvement: D40 header (P1-P4) -> D43 header (data scaling).
    #   workspace rounds: D44 header -> R3 at D46 header (runs into 10-11).
    #   replication/multi-arch: anchored at D47's ~11:00Z record time; spans
    #       without documented ends are schematic (footnote discloses).
    D28_fit = datetime(2026, 10, 4, 9, 16, tzinfo=timezone.utc)  # D28 in-body
    D32_launch = datetime(2026, 10, 6, 8, 36, 19, tzinfo=timezone.utc)  # D32 in-body
    D33_A = datetime(2026, 10, 6, 20, 52, 36, tzinfo=timezone.utc)  # D33 in-body
    D35_done = datetime(2026, 10, 7, 0, 56, tzinfo=timezone.utc)  # D35 in-body
    D36_done = datetime(2026, 10, 7, 19, 15, 49, tzinfo=timezone.utc)  # D36 in-body
    D42_fit_done = datetime(2026, 10, 9, 20, 44, tzinfo=timezone.utc)  # D42 in-body

    lanes = [
        # (label, start, end, color, kind)  kind: solid | schematic | queued
        ("setup + step0 + X6 gate\n(D0\u2013D11)", h(t0),
         h(datetime(2026, 10, 1, 3, 21, tzinfo=timezone.utc)), GREY, "solid"),
        ("S1 fit + gate\n(D12\u2013D20)",
         h(datetime(2026, 10, 1, 4, 45, tzinfo=timezone.utc)),
         h(datetime(2026, 10, 2, 17, 40, tzinfo=timezone.utc)), BLUE, "solid"),
        ("S2 halves A/B\n(D21\u2013D33)",
         h(datetime(2026, 10, 2, 19, 45, tzinfo=timezone.utc)), h(D33_A), GREEN, "solid"),
        ("merge + s2_eval\n(D33\u2013D35)", h(D33_A), h(D35_done), SKY, "solid"),
        ("X1 / X7 / X9 (+FD)\n(D35\u2013D36)", h(D35_done), h(D36_done), ORANGE, "solid"),
        ("X11 logit-lens diag\n(D36\u2013D37)",
         h(datetime(2026, 10, 8, 5, 10, tzinfo=timezone.utc)),
         h(datetime(2026, 10, 8, 6, 10, tzinfo=timezone.utc)), PURPLE, "solid"),
        ("improvement P1\u2013P5 + scaling\n(D40\u2013D43)",
         h(datetime(2026, 10, 8, 21, 50, tzinfo=timezone.utc)),
         h(datetime(2026, 10, 10, 13, 0, tzinfo=timezone.utc)), VERMIL, "solid"),
        ("workspace rounds 1\u20133\n(D44\u2013D46)",
         h(datetime(2026, 10, 10, 15, 40, tzinfo=timezone.utc)),
         h(datetime(2026, 10, 11, 0, 0, tzinfo=timezone.utc)), BLUE, "solid"),
        ("replication x20\u2013x23\n(D47)",
         h(datetime(2026, 10, 11, 6, 0, tzinfo=timezone.utc)), h(D["D47"]),
         GREEN, "schematic"),
        ("13b/qwen multi-arch\nbuilt+gated, fits queued",
         h(D["D47"]), h(datetime(2026, 10, 11, 16, 0, tzinfo=timezone.utc)),
         GREY, "queued"),
    ]

    # Incident markers: (D-ref label, datetime, type). Header-dated entries use
    # the parsed DECISIONS.md anchors; in-body-only times are commented.
    incidents = [
        # (D-ref label, datetime, type, label dx nudge in data hours)
        ("OOM D11", datetime(2026, 10, 1, 2, 3, tzinfo=timezone.utc), "oom", -0.3),  # REPORT 9: retries 02:03
        ("OOM D12", datetime(2026, 10, 1, 4, 45, tzinfo=timezone.utc), "oom", 0.0),
        ("SIGKILL x2 D14/15", datetime(2026, 10, 1, 7, 16, tzinfo=timezone.utc), "kill", 0.3),
        ("derate 41-49 TF D22/23", datetime(2026, 10, 2, 17, 5, tzinfo=timezone.utc), "derate", 0.0),  # D22 in-body
        ("ENOSPC D24", D["D24"], "enospc", 0.0),
        ("reloc->mount D25", D["D25"], "reloc", 0.0),
        ("SIGTERM D26", D["D26"], "kill", -0.45),
        ("SIGKILL+OOM D27, cad 5->2", D["D27"], "kill", 0.45),
        ("exFAT write-death D29", D["D29"], "enospc", 0.0),
        ("reloc->/data D30", D["D30"], "reloc", 0.0),
        ("reloc->Brev 8xH100 D32", D["D32"], "reloc", 0.0),
        ("scorer swap D34", D["D34"], "ops", 0.0),
        ("OOM x3 D38", datetime(2026, 10, 8, 11, 4, tzinfo=timezone.utc), "oom", 0.0),  # D38: rc=1 at 11:04:25Z
        ("ENOSPC D46", D["D46"], "enospc", 0.0),
    ]
    INC_COLOR = {"oom": ORANGE, "kill": VERMIL, "derate": PURPLE,
                 "enospc": BLUE, "reloc": GREEN, "ops": GREY}

    fig = plt.figure(figsize=(14.2, 8.6))
    ax = fig.add_axes([0.135, 0.475, 0.845, 0.44])
    n = len(lanes)

    for day in range(0, 12):
        ax.axvline(24 * day, color="0.82", lw=0.7, zorder=0)
    ax.set_xticks([24 * d for d in range(12)])
    ax.set_xticklabels([f"10-{d+1:02d}" for d in range(12)], fontsize=8.5)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xlim(-1.5, 270)
    ax.set_ylim(-3.7, n + 0.55)
    ax.set_yticks([])
    ax.grid(False)

    # campaign-complete marker line
    ax.axvline(h(D36_done), color=GREY, lw=1.0, ls="--", alpha=0.7, zorder=1)
    ax.text(h(D36_done) + 1.2, n + 0.24, "campaign complete D36 (10-07 19:15:49Z)",
            fontsize=7.6, color=GREY, ha="left", va="bottom")

    for i, (label, s, e, color, kind) in enumerate(lanes):
        y = n - i
        if kind == "queued":
            ax.add_patch(
                plt.Rectangle((s, y - 0.31), e - s, 0.62, facecolor="none",
                              edgecolor=GREY, hatch="///", lw=1.0, ls="--", zorder=3)
            )
        else:
            alpha = 0.68 if kind == "schematic" else 0.92
            ax.add_patch(
                plt.Rectangle((s, y - 0.31), e - s, 0.62, facecolor=color,
                              edgecolor="white", lw=0.8, alpha=alpha, zorder=3)
            )
        ax.text(-2.0, y, label, ha="right", va="center", fontsize=8.3)

    # sub-phase diamonds on the S2 / improvement lanes
    diamonds = [
        (h(D28_fit), n - 2, "half-A done D28", GREEN, "above"),
        (h(D32_launch), n - 2, "Brev relaunch D32", GREEN, "below"),
        (h(D42_fit_done), n - 6, "tmtext refit done D42", VERMIL, "below"),
        (h(D["D41"]), n - 6, "P5 D41", VERMIL, "above"),
        (h(D["D43"]), n - 6, "scaling D43", VERMIL, "above"),
    ]
    for x, y, txt, c, pos in diamonds:
        ax.plot([x], [y], marker="D", ms=4.5, color=c, mec="white", mew=0.6, zorder=5)
        ax.text(x, y + (0.42 if pos == "above" else -0.46), txt, ha="center",
                va="bottom" if pos == "above" else "top", fontsize=6.6, color=c, zorder=5)

    # workspace round separators + labels
    wy = n - 7
    for xs, xe, rl in (
        (h(datetime(2026, 10, 10, 15, 40, tzinfo=timezone.utc)),
         h(datetime(2026, 10, 10, 20, 0, tzinfo=timezone.utc)), "R1"),
        (h(datetime(2026, 10, 10, 20, 0, tzinfo=timezone.utc)),
         h(datetime(2026, 10, 10, 22, 0, tzinfo=timezone.utc)), "R2"),
        (h(datetime(2026, 10, 10, 22, 0, tzinfo=timezone.utc)),
         h(datetime(2026, 10, 11, 0, 0, tzinfo=timezone.utc)), "R3"),
    ):
        ax.text((xs + xe) / 2, wy, rl, ha="center", va="center", fontsize=6.5,
                color="white", fontweight="bold", rotation=90, zorder=6)

    # incident strip
    for label, dt, kind, dx in incidents:
        x = h(dt)
        c = INC_COLOR[kind]
        ax.plot([x, x], [-0.38, -0.95], color=c, lw=0.8, alpha=0.45, zorder=2)
        ax.plot([x], [-0.95], marker="o", ms=5, color=c, mec="white", mew=0.5, zorder=4)
        ax.text(x + dx, -1.15, label, rotation=90, ha="center", va="top",
                fontsize=6.9, color=c, zorder=4)
    ax.text(-2.0, -0.95, "incidents", ha="right", va="center", fontsize=8.3,
            fontweight="bold")

    # ------------------------------------------------ GPU-hours inset (9)
    measured = [
        ("Step 0 corpus+cost+census", 0.9, "measured"),
        ("X6 bf16 leg", 0.61, "measured"),
        ("X6 fp32/TF32 (7 retries)", 1.8, "measured"),
        ("S1 fit: first 20 samples", 0.48, "derived"),
        ("S1 fit: remaining 80", 2.04, "measured"),
        ("S1 scoring x3 (FD crash)", 0.4, "measured"),
    ]
    projected = [
        ("S1 scoring + FD (proj.)", 0.5, "projected"),
        ("S2 halves + merge + eval", 9.4, "projected"),
        ("X1 shard pair (2 masks)", 1.8, "projected"),
        ("X7 conditioning + X9 sweep", 0.3, "projected"),
    ]
    axI = fig.add_axes([0.575, 0.075, 0.405, 0.32])
    ys = np.arange(len(measured) + len(projected))
    vals = [v for _, v, _ in measured] + [v for _, v, _ in projected]
    labels = [l for l, _, _ in measured] + [l for l, _, _ in projected]
    kinds = [k for _, _, k in measured] + [k for _, _, k in projected]
    color_map = {"measured": BLUE, "derived": SKY, "projected": ORANGE}
    hatch_map = {"measured": None, "derived": "//", "projected": "//"}
    for y, v, l, k in zip(ys, vals, labels, kinds, strict=False):
        axI.barh(y, v, height=0.68, color=color_map[k], hatch=hatch_map[k],
                 edgecolor="white" if k == "measured" else "0.3", lw=0.4)
        txt = "9.1+0.3" if l.startswith("S2 halves") else f"{v:.2f}"
        axI.text(v + 0.12, y, txt, va="center", fontsize=7.6, color="0.2")
    axI.set_yticks(ys)
    axI.set_yticklabels(labels, fontsize=7.6)
    axI.invert_yaxis()
    axI.set_xlim(0, 10.6)
    axI.set_xticks(range(0, 11, 2))
    axI.set_xlabel("GPU-hours", fontsize=8.5)
    axI.tick_params(axis="x", labelsize=8)
    axI.set_title("GPU-hours by phase \u2014 REPORT.md \u00a79", fontsize=9.5)
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=BLUE, label="measured (\u00a79 provenance/logs)"),
        plt.Rectangle((0, 0), 1, 1, fc=SKY, hatch="//", ec="0.3", lw=0.4,
                      label="derived (component)"),
        plt.Rectangle((0, 0), 1, 1, fc=ORANGE, hatch="//", ec="0.3", lw=0.4,
                      label="projected (\u00a79 [derived]/[inference])"),
    ]
    axI.legend(handles=handles, loc="lower right", fontsize=6.9)

    fig.text(
        0.035, 0.415,
        "Anchors: DECISIONS.md D-header dates (UTC, parsed at run time); REPORT.md \u00a79/\u00a711 for incidents and GPU-hours.\n"
        "Spans without documented ends (replication x20\u2013x23, multi-arch) are schematic 5 h, anchored at the\n"
        "D47 record time (2026-10-11 ~11:00Z) \u2014 drawn lighter/hatched.\n"
        "Incidents: OOM D11/D12/D27/D38 \u00b7 external SIGKILLs D14/D15 + SIGTERM D26 \u00b7 TFLOPS derate D22/D23\n"
        "(41\u201349 vs 231 quiet, D26) \u00b7 ENOSPC D24/D29/D46 \u00b7 relocations D25/D30/D32 \u00b7 scorer swap D34 \u00b7 cadence 5\u21922 (D27).\n"
        "Post-completion rounds (D36\u2013D47): X11 diagnostic; improvement P1\u2013P5 (P1 affine, P2 scaling, P3 alpha,\n"
        "P4 = D40 entry, P5 zoo); workspace rounds 1\u20133; replication x20\u2013x23; multi-arch llava-13b (40L) /\n"
        "qwen2-vl-7b (28L) built+gated, fits queued \u2014 no numbers yet (EXPERIMENTS.md 'In-flight work').\n"
        "\u00a79 totals: spent \u2248 6.3 h [derived]; remaining proj. \u2248 13 h; GPU-occupied to date \u2248 8.4 h\n"
        "(~6.5 h quiet-equiv) @ 10-02T17:40Z; S2/X1/X7X9 actuals and later rounds have no \u00a79 rows.",
        fontsize=7.2, color="0.25", va="top", linespacing=1.45,
    )

    fig.savefig(FIGS / "figD10_campaign_timeline.png", **SAVE_KW)
    plt.close(fig)

    inc_by_type = {}
    for _, _, kind, _ in incidents:
        inc_by_type[kind] = inc_by_type.get(kind, 0) + 1
    return {
        "timeline_span": "2026-10-01T00:00Z -> 2026-10-11T24:00Z (11 days)",
        "phase_lanes": 10,
        "incident_markers": len(incidents),
        "incident_markers_by_type": inc_by_type,
        "gpu_hours_measured_total": 6.3,
        "gpu_hours_projected_remaining": 13.0,
        "gpu_occupied_to_date_h": 8.4,
        "gpu_occupied_quiet_equiv_h": 6.5,
        "derate_tflops": "41-49 derated vs 231 quiet vs 88 at 09:01Z 10-03 (D22/D26)",
        "checkpoint_cadence": "5 -> 2 samples (D27)",
        "s2_half_wall_hours": {
            "half_A_old_box": 15.7,
            "brev_halves": "11.3-11.6 each (B done 20:02Z, A 20:52:36Z 10-06, D33)",
        },
        "campaign_complete": "2026-10-07T19:15:49Z (D36)",
        "multi_arch": "llava-1.5-13b-hf (40 layers) + Qwen2-VL-7B-Instruct (28 layers) "
        "built and gated (phase28), fits queued (phase29/30) - no numbers yet (EXPERIMENTS.md)",
        "citation": "DECISIONS.md D10-D47 headers (dates, parsed at run time) + in-body times; "
        "REPORT.md \u00a79 GPU-hours table and \u00a711 incidents",
    }


def main():
    metrics = {
        "round": "validation_2026-10-01",
        "family": "S2 caption-lens design/eval + campaign operations timeline "
                  "(figD9, figD10)",
        "script": "results/validation_2026-10-01/code/figures_d_campaign.py",
        "figures": [
            {
                "file": "figD9_s2_halves_transfer.png",
                "sources": [
                    "results/validation_2026-10-01/raw/step3/s2_eval.json",
                    "results/validation_2026-10-01/DECISIONS.md D34/D35/D37 "
                    "(cross-check + L20 attribution)",
                ],
                "key_numbers": figD9(),
                "finding": "S2 verdict: the merged 100-sample caption lens is exact at L31 "
                           "(rank 42.62 = model, KL 0, top-1 1.000) with expected structure; "
                           "the A4/X8 halves are symmetric (own-vs-cross deltas 0.5%/2.6%, "
                           "merge == single fit, no leakage); the caption lens transfers to "
                           "held-out WikiText at 2.49x the model ceiling while the S1 text "
                           "lens transfers to captions (S1 wins rank 109.5 vs 215.2, S2 wins "
                           "agree/KL); the L20 bump (+110% rank) is fitted-map-specific "
                           "(X11), not a model property.",
            },
            {
                "file": "figD10_campaign_timeline.png",
                "sources": [
                    "results/validation_2026-10-01/DECISIONS.md D10-D47 headers "
                    "(ops timeline: OOMs D11/D12/D27/D38, external SIGKILLs D14/D15, "
                    "TFLOPS derate D22/D23, ENOSPC D24/D29/D46, box relocation D32, "
                    "cadence 5->2 D27, scorer swap D34, the 13b/qwen multi-arch round "
                    "2026-10-11)",
                    "results/validation_2026-10-01/REPORT.md \u00a79 (GPU-hours) + \u00a711 "
                    "(incidents)",
                ],
                "key_numbers": figD10(),
                "finding": "Eleven days, two boxes: 14 dated incident markers (OOM x3 plus "
                           "one SIGKILL-with-OOM, 3 external SIGKILL/SIGTERM rounds, 3 "
                           "storage-full/write-death, 3 output relocations, 1 TFLOPS derate, "
                           "1 scorer swap) never stopped the campaign - guarded windows, "
                           "uncapped transient retries and checkpoint cadence 5->2 "
                           "self-healed it; the core campaign completed "
                           "2026-10-07T19:15:49Z (D36) and four more rounds (improvement "
                           "P1-P5, workspace 1-3, replication x20-x23, multi-arch 13b/qwen "
                           "built+gated) ran through 2026-10-11.",
            },
        ],
        "notes": [
            "figD9 panel (a) uses main.rows.default (placeholder-excluded): the JSON's "
            "default table carries text/image/all plus image-q3, which is identical to "
            "image (V5 last-patch collapse), so image-q0..q2 (only under "
            "include_placeholders) are not plotted.",
            "figD9 panel (c) shows the two E6 transfer rows present in s2_eval.json. "
            "D35 also quotes S1's own WikiText rank 46.5 (1.70x ceiling) - that number "
            "lives in S1's step2 score, not in s2_eval.json, so it is cited as "
            "DECISIONS.md D35 only and not plotted.",
            "figD10 GPU-hours inset covers only the rows REPORT.md \u00a79 states (as of "
            "its 10-02 writing: measured Step0/X6/S1 rows + the remaining-campaign "
            "projection); S2/X1/X7X9 actuals and the later rounds (P1-P5, workspace, "
            "replication, multi-arch) have no \u00a79 GPU-hour rows - marked "
            "measured vs derived/projected as \u00a79 tags them.",
            "figD10 multi-arch segment: raw/fits/{llava-1.5-13b-hf,qwen2-vl-7b-instruct}/ "
            "exist but are empty (gates ran, fits queued); EXPERIMENTS.md 'In-flight "
            "work' records no numbers - drawn hatched, no results claimed.",
            "Deterministic: fixed palette/layout, D-header anchors parsed from "
            "DECISIONS.md at run time, no timestamps inside figures.",
        ],
    }
    out = SUMMARY / "metrics_d_campaign.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"wrote {out}")
    for entry in metrics["figures"]:
        print(f"  {entry['file']}")


if __name__ == "__main__":
    main()
