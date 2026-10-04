import { useEffect, useRef } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';
import { heatColor } from '../lib/colormap';
import type { GridShape } from '../lib/types';

export interface HeatmapGridProps {
  /** Row-major scores (grid[row][col]); null before attribution is fetched. */
  values: number[][] | null;
  grid: GridShape;
  vmin: number;
  vmax: number;
  /** CSS pixel size of the overlay (= displayed image box). */
  width: number;
  height: number;
  alpha?: number;
  showQuarters?: boolean;
  quarterOfPatch?: number[] | null;
  selected: Set<number>;
  focus: number | null;
  onHover?: (idx: number | null, x: number, y: number) => void;
  onPaintStart?: (idx: number) => void;
  onPaintMove?: (idx: number) => void;
  onPaintEnd?: () => void;
}

const QUARTER_LABELS = ['Q0', 'Q1', 'Q2', 'Q3'];

/**
 * Canvas overlay for the 24x24 patch attribution grid, drawn on top of the
 * session image. Handles cell hit-testing, hover reporting and click-drag
 * painting; the parent owns selection state.
 */
export function HeatmapGrid({
  values,
  grid,
  vmin,
  vmax,
  width,
  height,
  alpha = 0.72,
  showQuarters = false,
  quarterOfPatch = null,
  selected,
  focus,
  onHover,
  onPaintStart,
  onPaintMove,
  onPaintEnd,
}: HeatmapGridProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const painting = useRef(false);
  const [cols, rows] = grid;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(width * dpr));
    canvas.height = Math.max(1, Math.round(height * dpr));
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const cw = width / cols;
    const ch = height / rows;

    if (values) {
      for (let r = 0; r < rows; r += 1) {
        for (let c = 0; c < cols; c += 1) {
          const v = values[r]?.[c];
          if (v === undefined) continue;
          ctx.fillStyle = heatColor(v, vmin, vmax, alpha);
          ctx.fillRect(c * cw, r * ch, cw + 0.5, ch + 0.5);
        }
      }
    } else {
      ctx.strokeStyle = 'rgba(20, 20, 19, 0.25)';
      ctx.lineWidth = 1;
      for (let r = 0; r <= rows; r += 1) {
        ctx.beginPath();
        ctx.moveTo(0, r * ch);
        ctx.lineTo(width, r * ch);
        ctx.stroke();
      }
      for (let c = 0; c <= cols; c += 1) {
        ctx.beginPath();
        ctx.moveTo(c * cw, 0);
        ctx.lineTo(c * cw, height);
        ctx.stroke();
      }
    }

    if (showQuarters) {
      ctx.save();
      ctx.strokeStyle = 'rgba(250, 249, 245, 0.75)';
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(width / 2, 0);
      ctx.lineTo(width / 2, height);
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.stroke();
      ctx.restore();
      if (quarterOfPatch && quarterOfPatch.length >= cols * rows) {
        ctx.font = '500 11px "JetBrains Mono", ui-monospace, Menlo, monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        const qw = width / 2;
        const qh = height / 2;
        QUARTER_LABELS.forEach((label, q) => {
          const x = (q % 2) * qw + 4;
          const y = Math.floor(q / 2) * qh + 4;
          ctx.fillStyle = 'rgba(24, 23, 21, 0.72)';
          const metrics = ctx.measureText(label);
          ctx.fillRect(x - 2, y - 1, metrics.width + 4, 14);
          ctx.fillStyle = 'rgba(250, 249, 245, 0.92)';
          ctx.fillText(label, x, y);
        });
      }
    }

    ctx.lineWidth = 2;
    selected.forEach((idx) => {
      const r = Math.floor(idx / cols);
      const c = idx % cols;
      if (r < 0 || r >= rows || c < 0 || c >= cols) return;
      ctx.strokeStyle = 'rgba(204, 120, 92, 0.95)';
      ctx.strokeRect(c * cw + 1, r * ch + 1, Math.max(1, cw - 2), Math.max(1, ch - 2));
    });
    if (focus !== null) {
      const r = Math.floor(focus / cols);
      const c = focus % cols;
      ctx.strokeStyle = 'rgba(250, 249, 245, 0.95)';
      ctx.lineWidth = 2;
      ctx.strokeRect(c * cw + 1, r * ch + 1, Math.max(1, cw - 2), Math.max(1, ch - 2));
    }
  }, [values, cols, rows, vmin, vmax, width, height, alpha, showQuarters, quarterOfPatch, selected, focus]);

  function cellAt(e: ReactPointerEvent<HTMLCanvasElement>): { idx: number; x: number; y: number } | null {
    const rect = e.currentTarget.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    if (x < 0 || y < 0 || x >= rect.width || y >= rect.height) return null;
    const c = Math.min(cols - 1, Math.max(0, Math.floor((x / rect.width) * cols)));
    const r = Math.min(rows - 1, Math.max(0, Math.floor((y / rect.height) * rows)));
    return { idx: r * cols + c, x, y };
  }

  return (
    <canvas
      ref={canvasRef}
      className="heat-canvas"
      onPointerDown={(e) => {
        const hit = cellAt(e);
        if (!hit) return;
        e.currentTarget.setPointerCapture(e.pointerId);
        painting.current = true;
        onPaintStart?.(hit.idx);
      }}
      onPointerMove={(e) => {
        const hit = cellAt(e);
        if (painting.current) {
          if (hit) onPaintMove?.(hit.idx);
        }
        onHover?.(hit ? hit.idx : null, hit?.x ?? 0, hit?.y ?? 0);
      }}
      onPointerUp={() => {
        if (!painting.current) return;
        painting.current = false;
        onPaintEnd?.();
      }}
      onPointerCancel={() => {
        if (!painting.current) return;
        painting.current = false;
        onPaintEnd?.();
      }}
      onPointerLeave={() => {
        onHover?.(null, 0, 0);
      }}
    />
  );
}
