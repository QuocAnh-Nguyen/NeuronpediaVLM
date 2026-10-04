/**
 * Plain-SVG line chart of a tracked token's lens probability vs layer.
 * No chart library: one path, dots with `<title>` tooltips, axis ticks.
 */

export interface LayerPoint {
  layer: number;
  prob: number;
  rank?: number;
}

export interface LayerChartProps {
  points: LayerPoint[];
  color?: string;
  height?: number;
  /** Chart heading, e.g. the tracked token. */
  title?: string;
}

const W = 560;
const M = { top: 14, right: 18, bottom: 28, left: 52 };

export function LayerChart({ points, color = 'var(--primary)', height = 190, title }: LayerChartProps) {
  if (points.length === 0) return <div className="empty small">no tracked token — select one to see its per-layer curve</div>;
  const minX = Math.min(...points.map((p) => p.layer));
  const maxX = Math.max(...points.map((p) => p.layer));
  const maxY = Math.max(...points.map((p) => p.prob), 1e-6) * 1.08;
  const innerW = W - M.left - M.right;
  const innerH = height - M.top - M.bottom;
  const sx = (layer: number) => M.left + (maxX > minX ? ((layer - minX) / (maxX - minX)) * innerW : innerW / 2);
  const sy = (prob: number) => M.top + innerH - Math.max(0, Math.min(1, prob / maxY)) * innerH;
  const path = points.map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${sx(p.layer).toFixed(2)} ${sy(p.prob).toFixed(2)}`).join(' ');
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * maxY);
  const step = Math.max(1, Math.ceil(points.length / 12));
  const last = points[points.length - 1];
  return (
    <div className="chart-wrap">
      {title ? <div className="chart-title small muted">{title}</div> : null}
      <svg viewBox={`0 0 ${W} ${height}`} className="chart" role="img" aria-label={`lens probability vs layer for ${title ?? 'tracked token'}`}>
        {ticks.map((t, idx) => (
          <g key={idx}>
            <line x1={M.left} x2={W - M.right} y1={sy(t)} y2={sy(t)} className="chart-grid" />
            <text x={M.left - 8} y={sy(t) + 4} textAnchor="end" className="chart-tick mono">
              {t.toFixed(t >= 0.1 || t === 0 ? 2 : 3)}
            </text>
          </g>
        ))}
        <line x1={M.left} x2={M.left} y1={M.top} y2={height - M.bottom} className="chart-axis" />
        <line x1={M.left} x2={W - M.right} y1={height - M.bottom} y2={height - M.bottom} className="chart-axis" />
        {points.map((p, idx) =>
          idx % step === 0 || idx === points.length - 1 ? (
            <text key={`x${p.layer}`} x={sx(p.layer)} y={height - M.bottom + 16} textAnchor="middle" className="chart-tick mono">
              {p.layer}
            </text>
          ) : null,
        )}
        <path d={path} className="chart-line" style={{ stroke: color }} />
        {points.map((p) => (
          <circle key={p.layer} cx={sx(p.layer)} cy={sy(p.prob)} r={3} className="chart-dot" style={{ fill: color }}>
            <title>{`L${p.layer}: p=${p.prob.toFixed(4)}${p.rank !== undefined ? ` · rank #${p.rank}` : ''}`}</title>
          </circle>
        ))}
        <text x={W - M.right} y={M.top + 2} textAnchor="end" className="chart-tick mono">
          last p={last.prob.toFixed(4)}
        </text>
      </svg>
      <div className="muted small">x = layer index, y = lens probability of the tracked token</div>
    </div>
  );
}
