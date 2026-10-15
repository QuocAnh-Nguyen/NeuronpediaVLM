# Throwaway analyzer for the J-lens improvement experiments (Phase 4).
# Usage: python3 tmp/analyze_improvement.py   (expects the pulled JSONs under /tmp/brev/)
import json
import math
from pathlib import Path

BR = Path("/tmp/brev")
out = []


def idx(rows):
    return {(r["layer"], r["tag"]): r for r in rows}


zoo = json.load(open(BR / "lens_zoo.json"))
old = json.load(open(BR / "s2_eval.json"))

biased_path = BR / "lens_zoo_biased.json"
if biased_path.exists():
    zm = idx(json.load(open(biased_path))["zoo"]["merged"]["rows"])
    bias_src = "with bias"
else:
    zm = idx(zoo["zoo"]["merged"]["rows"])
    bias_src = "unbiased fallback (biased run pending)"
om = idx(old["main"]["rows"]["default"])
out.append(f"== BIAS A/B (merged lens, tag=text, 300 heldout; nb = historical no-bias, b = {bias_src}) ==")
out.append(f"{'L':>3} {'rank_nb':>9} {'rank_b':>9} {'d%':>7} {'med_b':>8} {'top10_b':>8} {'kl_nb':>7} {'kl_b':>7}")
for L in (0, 8, 12, 16, 20, 24, 27, 30):
    a, b = om.get((L, "text")), zm.get((L, "text"))
    if not a or not b:
        continue
    d = 100 * (b["mean_rank_true"] / a["mean_rank_true"] - 1)
    med = b.get("median_rank_true", float("nan"))
    t10 = b.get("top10_agreement", float("nan"))
    out.append(f"{L:>3} {a['mean_rank_true']:>9.2f} {b['mean_rank_true']:>9.2f} {d:>+6.1f}% "
               f"{med:>8.1f} {t10:>8.3f} {a['mean_kl']:>7.3f} {b['mean_kl']:>7.3f}")
if (20, "text") in om and (16, "text") in om and (20, "text") in zm and (16, "text") in zm:
    ba = om[(20, "text")]["mean_rank_true"] / om[(16, "text")]["mean_rank_true"]
    bb = zm[(20, "text")]["mean_rank_true"] / zm[(16, "text")]["mean_rank_true"]
    out.append(f"L20/L16 bump ratio: no-bias {ba:.2f}x -> with-bias {bb:.2f}x "
               f"({'reduced' if bb < ba else 'NOT reduced'})")
out.append("")
out.append("== SCALING (LQS + rank_ratio vs n_prompts; with bias where present) ==")
for name, d in zoo["zoo"].items():
    rr = d.get("rank_ratio", {})
    rrs = " ".join(f"L{L}:{rr[str(L)]:.2f}" for L in (0, 16, 24, 30) if str(L) in rr)
    out.append(f"{name:<10} n={d['n_prompts']:<4} LQS={d['lqs']:+.4f}  {rrs}")
out.append("")
out.append("== LOGIT-LENS BASELINE (context) ==")
bl = idx(zoo["logit_baseline"]["rows"])
for L in (0, 16, 20, 24, 30):
    r = bl.get((L, "text"))
    if r:
        out.append(f"  L{L} rank={r['mean_rank_true']:.2f} med={r.get('median_rank_true', float('nan')):.1f}")

x9b_path = BR / "x9b_alpha_sweep.json"
if x9b_path.exists():
    x9b = json.load(open(x9b_path))
    out.append("")
    out.append("== X9B ALPHA SWEEP ==")
    out.append("minimal effective strength (change_rate >= 0.2): " + json.dumps(x9b.get("minimal_effective", {})))
    summ = x9b.get("summary", {})
    for mode in ("add", "ablate", "swap"):
        cells = summ.get(mode, {})
        if not cells:
            continue
        for layer, by_strength in sorted(cells.items()):
            parts = []
            for s, cell in sorted(by_strength.items(), key=lambda kv: float(kv[0])):
                parts.append(f"k/α={s}:chg={cell['change_rate']:.2f}/dir={cell['directed_rate']:.2f}")
            out.append(f"  {mode}@L{layer}: " + "  ".join(parts))

Path(BR / "improvement_digest.txt").write_text("\n".join(out) + "\n")
print("ok", len(out))
