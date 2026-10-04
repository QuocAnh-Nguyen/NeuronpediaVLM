/**
 * Color ramps for the attribution heatmap and the signed delta bars, keyed to
 * the demo's design tokens (deep teal / coral / on-dark delta pair).
 *
 * The heat overlay is a diverging map around zero (per-grid z-score) or a
 * sequential coral ramp above the range minimum (raw scale): deviation is
 * carried by hue (teal = below center, coral = above) and by alpha, so it
 * stays readable over an arbitrary photo. Kept dependency-free.
 */

export type RGB = [number, number, number];

/** #5db8a6 — below-center deviation. */
const TEAL: RGB = [93, 184, 166];
/** #cc785c — above-center deviation. */
const CORAL: RGB = [204, 120, 92];
/** On-dark delta pair: #7fd39a (positive) / #e8846f (negative). */
const DELTA_POS: RGB = [127, 211, 154];
const DELTA_NEG: RGB = [232, 132, 111];

/**
 * Translucent overlay color for a raw heat value; `alpha` is the maximum
 * opacity (small deviations stay faint, extremes approach `alpha`).
 */
export function heatColor(v: number, vmin: number, vmax: number, alpha = 0.8): string {
  const center = vmin < 0 && vmax > 0 ? 0 : vmin;
  const side = Math.max(vmax - center, center - vmin, 1e-12);
  const t = Math.max(-1, Math.min(1, (v - center) / side));
  const hue = t < 0 ? TEAL : CORAL;
  const a = alpha * (0.2 + 0.8 * Math.abs(t));
  return `rgba(${hue[0]}, ${hue[1]}, ${hue[2]}, ${a})`;
}

/** CSS `linear-gradient` of the same ramp, for the legend bar. */
export function heatGradient(vmin: number, vmax: number, stops = 24): string {
  const n = Math.max(2, stops);
  const span = vmax - vmin || 1;
  const parts: string[] = [];
  for (let i = 0; i < n; i += 1) {
    const v = vmin + span * (i / (n - 1));
    parts.push(`${heatColor(v, vmin, vmax, 1)} ${((100 * i) / (n - 1)).toFixed(1)}%`);
  }
  return `linear-gradient(90deg, ${parts.join(', ')})`;
}

/**
 * Color for a signed value (knockout log-prob deltas): negative = warm red,
 * positive = green; magnitude scales opacity up to `alpha`.
 */
export function signedColor(delta: number, maxAbs: number, alpha = 0.72): string {
  const m = Math.max(1e-9, Math.abs(maxAbs));
  const t = Math.max(0, Math.min(1, Math.abs(delta) / m));
  const hue = delta >= 0 ? DELTA_POS : DELTA_NEG;
  return `rgba(${hue[0]}, ${hue[1]}, ${hue[2]}, ${0.12 + alpha * t})`;
}

/** Format a heat value compactly but readably (3 significant digits). */
export function formatHeat(v: number): string {
  if (!Number.isFinite(v)) return 'n/a';
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 100) return v.toFixed(1);
  if (a >= 1) return v.toFixed(3);
  if (a >= 0.001) return v.toFixed(4);
  return v.toExponential(2);
}
