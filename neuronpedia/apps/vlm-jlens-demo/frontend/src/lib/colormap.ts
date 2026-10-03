/**
 * A small, dependency-free viridis-like colormap plus helpers for the
 * attribution heatmap and the signed delta bars.
 *
 * The 11 anchor colors are the classic viridis stops (matplotlib's `viridis`
 * sampled at 0.0, 0.1, ... 1.0); intermediate values are linearly interpolated
 * in sRGB. Good enough for a smooth perceptual ramp on the 24x24 grid.
 */

export type RGB = [number, number, number];

export const VIRIDIS_STOPS: RGB[] = [
  [68, 1, 84], // #440154
  [72, 40, 120], // #482878
  [62, 73, 137], // #3e4989
  [49, 104, 142], // #31688e
  [38, 130, 142], // #26828e
  [31, 158, 137], // #1f9e89
  [53, 183, 121], // #35b779
  [110, 206, 88], // #6ece58
  [181, 222, 43], // #b5de2b
  [253, 231, 37], // #fde725
  [253, 231, 37],
];

function clamp01(t: number): number {
  return t < 0 ? 0 : t > 1 ? 1 : t;
}

/** Viridis color for t in [0, 1] (values outside are clamped). */
export function viridis(t: number): RGB {
  const x = clamp01(t) * (VIRIDIS_STOPS.length - 1);
  const i = Math.min(VIRIDIS_STOPS.length - 2, Math.floor(x));
  const f = x - i;
  const a = VIRIDIS_STOPS[i];
  const b = VIRIDIS_STOPS[i + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

/** `rgba(r,g,b,a)` CSS string for an RGB triple. */
export function rgbaCss(c: RGB, alpha = 1): string {
  return `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${alpha})`;
}

/** Linear normalization of `v` into [0, 1] given [vmin, vmax]. */
export function normalize(v: number, vmin: number, vmax: number): number {
  if (!Number.isFinite(v)) return 0;
  if (!(vmax > vmin)) return 0;
  return clamp01((v - vmin) / (vmax - vmin));
}

/** Viridis CSS color for a raw heat value. */
export function heatColor(v: number, vmin: number, vmax: number, alpha = 0.72): string {
  return rgbaCss(viridis(normalize(v, vmin, vmax)), alpha);
}

/** CSS `linear-gradient` spanning the full viridis ramp (for legends). */
export function viridisGradient(stops = VIRIDIS_STOPS.length): string {
  const parts: string[] = [];
  const n = Math.max(2, stops);
  for (let i = 0; i < n; i += 1) {
    const t = i / (n - 1);
    const c = viridis(t);
    parts.push(`rgb(${c[0]}, ${c[1]}, ${c[2]}) ${(t * 100).toFixed(1)}%`);
  }
  return `linear-gradient(90deg, ${parts.join(', ')})`;
}

/**
 * Color for a signed value (knockout log-prob deltas): negative = red,
 * positive = green, magnitude scaled by |maxAbs|.
 */
export function signedColor(delta: number, maxAbs: number, alpha = 0.55): string {
  const m = Math.max(1e-9, Math.abs(maxAbs));
  const t = clamp01(Math.abs(delta) / m);
  const rgb: RGB = delta >= 0 ? [74, 222, 128] : [248, 113, 113];
  return rgbaCss(rgb, 0.12 + 0.75 * t * alpha + 0.1);
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
