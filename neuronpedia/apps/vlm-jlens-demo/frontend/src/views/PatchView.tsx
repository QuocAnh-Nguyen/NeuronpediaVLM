import { useEffect, useMemo, useRef, useState } from 'react';
import { attribution, errorMessage, lens, pollJob } from '../lib/api';
import { formatHeat, viridisGradient } from '../lib/colormap';
import type {
  AttributionMetric,
  AttributionResult,
  GenerateResult,
  GridShape,
  Job,
  LensResult,
  Meta,
  SessionResponse,
} from '../lib/types';
import { HeatmapGrid } from '../components/HeatmapGrid';
import { JobProgress } from '../components/JobProgress';
import { TopKTable } from '../components/TopKTable';

const METRICS: Array<{ value: AttributionMetric; hint: string }> = [
  { value: 'lens_prob', hint: 'prob the lens gives the target token from this patch' },
  { value: 'lens_logit', hint: 'raw lens logit — less saturated than probability' },
  { value: 'attn_rollout', hint: 'attention-rollout mass reaching this patch' },
];

const QUARTER_NAMES = ['Q0 top-left', 'Q1 top-right', 'Q2 bottom-left', 'Q3 bottom-right'];

export interface PatchViewProps {
  meta: Meta | null;
  session: SessionResponse | null;
  imageB64: string | null;
  generated: GenerateResult | null;
  /** Token to attribute by default (flagged token or selected caption token). */
  defaultToken: string;
  selectedPatches: Set<number>;
  onSelectedPatchesChange: (next: Set<number>) => void;
}

/**
 * View 3 — patch attribution: canvas heatmap over the session image, brush
 * painting of patches, and a per-patch lens readout. The painted selection is
 * lifted to the app and reused by the knockout tool in the Steer tab.
 */
export function PatchView({
  meta,
  session,
  imageB64,
  generated,
  defaultToken,
  selectedPatches,
  onSelectedPatchesChange,
}: PatchViewProps) {
  const [layer, setLayer] = useState(0);
  const [metric, setMetric] = useState<AttributionMetric>('lens_prob');
  const [rollout, setRollout] = useState(true);
  const [token, setToken] = useState(defaultToken);
  const [showQuarters, setShowQuarters] = useState(true);
  const [scaleMode, setScaleMode] = useState<'zscore' | 'raw'>('zscore');
  const [attrJob, setAttrJob] = useState<Job<AttributionResult> | null>(null);
  const [attr, setAttr] = useState<AttributionResult | null>(null);
  const [attrErr, setAttrErr] = useState<string | null>(null);
  const [patchJob, setPatchJob] = useState<Job<LensResult> | null>(null);
  const [patchLens, setPatchLens] = useState<LensResult | null>(null);
  const [patchErr, setPatchErr] = useState<string | null>(null);
  const [focus, setFocus] = useState<number | null>(null);
  const [hover, setHover] = useState<{ idx: number; x: number; y: number } | null>(null);
  const [box, setBox] = useState({ w: 0, h: 0 });
  const imgRef = useRef<HTMLImageElement | null>(null);
  const strokeRef = useRef<{ idx: number; wasSelected: boolean; moved: boolean } | null>(null);
  const patchReqRef = useRef(0);
  const [tokenEdited, setTokenEdited] = useState(false);

  const sourceLayers = meta?.lens.source_layers ?? [];
  const imageSpan = session?.image ?? null;
  const grid: GridShape = imageSpan?.grid ?? meta?.model.grid ?? [24, 24];
  const [cols, rows] = grid;
  const patchCount = imageSpan?.count ?? cols * rows;
  const lensOk = meta?.lens.available ?? false;
  const maxLayer = Math.max(0, (meta?.model.n_layers ?? 1) - 1);

  useEffect(() => {
    if (!tokenEdited && defaultToken) setToken(defaultToken);
  }, [defaultToken, tokenEdited]);

  useEffect(() => {
    if (sourceLayers.length > 0 && !sourceLayers.includes(layer)) setLayer(sourceLayers[sourceLayers.length - 1]);
  }, [sourceLayers, layer]);

  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;
    const measure = () => {
      const w = img.clientWidth;
      const h = img.clientHeight;
      setBox((prev) => (prev.w === w && prev.h === h ? prev : { w, h }));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(img);
    return () => ro.disconnect();
  }, [imageB64]);

  function addPatches(ids: number[]) {
    if (ids.length === 0) return;
    const next = new Set(selectedPatches);
    for (const id of ids) next.add(id);
    onSelectedPatchesChange(next);
  }

  function removePatches(ids: number[]) {
    const next = new Set(selectedPatches);
    for (const id of ids) next.delete(id);
    onSelectedPatchesChange(next);
  }

  function selectTop64() {
    if (!attr) return;
    const flat: Array<{ idx: number; v: number }> = [];
    attr.grid.forEach((row, r) => {
      row.forEach((v, c) => flat.push({ idx: r * cols + c, v }));
    });
    flat.sort((a, b) => b.v - a.v);
    onSelectedPatchesChange(new Set(flat.slice(0, 64).map((x) => x.idx)));
  }

  async function runAttribution() {
    if (!session || !token.trim()) return;
    setAttrErr(null);
    setAttrJob(null);
    try {
      const { job_id } = await attribution({
        session_id: session.session_id,
        layer,
        token: token.trim(),
        metric,
        rollout,
      });
      const done = await pollJob<AttributionResult>(job_id, setAttrJob);
      if (done.result) setAttr(done.result);
    } catch (e) {
      setAttrErr(errorMessage(e));
    }
  }

  async function fetchPatchLens(idx: number) {
    if (!session) return;
    const reqId = patchReqRef.current + 1;
    patchReqRef.current = reqId;
    setPatchErr(null);
    setPatchJob(null);
    try {
      const { job_id } = await lens({
        session_id: session.session_id,
        targets: [{ kind: 'patch', i: idx }],
        topk: 8,
        track: token.trim() ? [token.trim()] : [],
      });
      const done = await pollJob<LensResult>(job_id, setPatchJob);
      if (patchReqRef.current === reqId && done.result) setPatchLens(done.result);
    } catch (e) {
      if (patchReqRef.current === reqId) setPatchErr(errorMessage(e));
    }
  }

  // Default view is the per-grid z-score (shape of the attribution), with the
  // raw metric one click away.
  const display = useMemo(() => {
    if (!attr) return null;
    if (scaleMode === 'raw') return { values: attr.grid, vmin: attr.vmin, vmax: attr.vmax };
    const vals = attr.grid.flat();
    const mean = vals.reduce((a, b) => a + b, 0) / Math.max(1, vals.length);
    const sd = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, vals.length)) || 1;
    const z = attr.grid.map((row) => row.map((v) => (v - mean) / sd));
    const lim = Math.max(1e-6, ...z.flat().map((v) => Math.abs(v)));
    return { values: z, vmin: -lim, vmax: lim };
  }, [attr, scaleMode]);

  if (!session) {
    return (
      <div className="panel">
        <div className="panel-body empty">
          Create a session in the <b>Session</b> tab first — then attribute caption tokens to image patches here.
        </div>
      </div>
    );
  }

  const hoverQuarter = hover && imageSpan ? imageSpan.quarter_of_patch[hover.idx] : undefined;
  const hoverRank = hover && attr?.grid_rank ? attr.grid_rank[Math.floor(hover.idx / cols)]?.[hover.idx % cols] : undefined;
  const focusQuarter = focus !== null && imageSpan ? imageSpan.quarter_of_patch[focus] : undefined;
  const patchTarget = patchLens?.targets[0] ?? null;
  const selection = [...selectedPatches].sort((a, b) => a - b);

  return (
    <div className="view-grid patch-view">
      <section className="panel">
        <header className="panel-header">
          <h2>patch attribution</h2>
          <span className="muted small">
            {cols}×{rows} = {patchCount} patches · patch index 0..{patchCount - 1} row-major
          </span>
        </header>
        <div className="panel-body">
          <div className="attr-controls">
            <label className="field">
              <span>
                layer <b className="mono">{layer}</b>
              </span>
              <input
                type="range"
                min={0}
                max={maxLayer}
                value={layer}
                onChange={(e) => setLayer(Number(e.target.value))}
                disabled={!meta}
              />
              <span className="muted small mono">
                {sourceLayers.length > 0 ? `lens layers ${sourceLayers.join(', ')}` : 'no lens layers reported'}
              </span>
            </label>
            <label className="field">
              <span>metric</span>
              <select value={metric} onChange={(e) => setMetric(e.target.value as AttributionMetric)}>
                {METRICS.map((m) => (
                  <option key={m.value} value={m.value}>
                    {m.value}
                  </option>
                ))}
              </select>
              <span className="muted small">{METRICS.find((m) => m.value === metric)?.hint}</span>
            </label>
            <label className="field inline">
              <input type="checkbox" checked={rollout} onChange={(e) => setRollout(e.target.checked)} />
              <span>attention rollout</span>
            </label>
            <label className="field">
              <span>target token</span>
              <input
                className="mono"
                list="caption-token-list"
                value={token}
                placeholder="e.g. garden"
                onChange={(e) => {
                  setTokenEdited(true);
                  setToken(e.target.value);
                }}
              />
              <datalist id="caption-token-list">
                {generated?.tokens.map((t) => (
                  <option key={t.i} value={t.str}>{`caption i=${t.i}`}</option>
                ))}
              </datalist>
            </label>
            <button type="button" className="btn primary" onClick={runAttribution} disabled={!lensOk || !token.trim()}>
              compute attribution
            </button>
          </div>
          <JobProgress job={attrJob} label="attribution" error={attrErr} />

          <div className="heat-wrap">
            {imageB64 ? (
              <img
                ref={imgRef}
                className="heat-img"
                src={`data:image/png;base64,${imageB64}`}
                alt="session"
                onLoad={(e) => {
                  const el = e.currentTarget;
                  setBox({ w: el.clientWidth, h: el.clientHeight });
                }}
              />
            ) : (
              <div className="empty small">
                session image not attached (it was created elsewhere) — recreate a session here to see the overlay
              </div>
            )}
            {imageB64 && box.w > 0 && box.h > 0 ? (
              <HeatmapGrid
                values={display?.values ?? null}
                grid={grid}
                vmin={display?.vmin ?? 0}
                vmax={display?.vmax ?? 1}
                width={box.w}
                height={box.h}
                showQuarters={showQuarters}
                quarterOfPatch={imageSpan?.quarter_of_patch ?? null}
                selected={selectedPatches}
                focus={focus}
                onHover={(idx, x, y) => setHover(idx === null ? null : { idx, x, y })}
                onPaintStart={(idx) => {
                  const wasSelected = selectedPatches.has(idx);
                  strokeRef.current = { idx, wasSelected, moved: false };
                  setFocus(idx);
                  if (!wasSelected) addPatches([idx]);
                }}
                onPaintMove={(idx) => {
                  if (!strokeRef.current) return;
                  if (idx !== strokeRef.current.idx) strokeRef.current.moved = true;
                  addPatches([idx]);
                }}
                onPaintEnd={() => {
                  const stroke = strokeRef.current;
                  strokeRef.current = null;
                  if (!stroke) return;
                  if (!stroke.moved) {
                    if (stroke.wasSelected) removePatches([stroke.idx]);
                    else void fetchPatchLens(stroke.idx);
                  } else {
                    void fetchPatchLens(stroke.idx);
                  }
                }}
              />
            ) : null}
            {hover ? (
              <div className="heat-tooltip mono" style={{ left: hover.x + 12, top: hover.y + 12 }}>
                patch {hover.idx} · r{Math.floor(hover.idx / cols)} c{hover.idx % cols} ·{' '}
                {attr ? formatHeat(attr.grid[Math.floor(hover.idx / cols)]?.[hover.idx % cols] ?? NaN) : '—'}
                {hoverRank !== undefined ? ` · rank #${hoverRank}` : ''}
                {hoverQuarter !== undefined ? ` · ${QUARTER_NAMES[hoverQuarter]}` : ''}
              </div>
            ) : null}
          </div>

          <div className="row wrap">
            <label className="field inline">
              <input type="checkbox" checked={showQuarters} onChange={(e) => setShowQuarters(e.target.checked)} />
              <span>quarter outlines</span>
            </label>
            {display ? (
              <>
                <span className="mode-toggle" role="group" aria-label="color scale">
                  <button
                    type="button"
                    className={`btn ghost small${scaleMode === 'zscore' ? ' active' : ''}`}
                    onClick={() => setScaleMode('zscore')}
                    title="per-grid z-score (mean 0, symmetric scale) — shows the shape of the attribution"
                  >
                    z-score
                  </button>
                  <button
                    type="button"
                    className={`btn ghost small${scaleMode === 'raw' ? ' active' : ''}`}
                    onClick={() => setScaleMode('raw')}
                    title="raw metric value, scaled by the reported vmin/vmax"
                  >
                    raw
                  </button>
                </span>
                <span className="legend">
                  <span className="mono small">{formatHeat(display.vmin)}</span>
                  <span className="legend-bar" style={{ background: viridisGradient() }} />
                  <span className="mono small">{formatHeat(display.vmax)}</span>
                </span>
              </>
            ) : null}
            <span className="muted small">
              {attr
                ? `${attr.metric} · layer ${attr.layer} · rollout ${attr.rollout ? 'on' : 'off'} · scale ${
                    scaleMode === 'zscore' ? 'z-score (per grid)' : 'raw'
                  }`
                : 'no attribution yet'}
            </span>
          </div>
          {attr ? (
            <div className="row wrap">
              {attr.quarters.map((q, i) => (
                <span key={i} className="meta-chip mono" title={`mean score · ${QUARTER_NAMES[i]}`}>
                  {QUARTER_NAMES[i].slice(0, 2)} {formatHeat(q)}
                </span>
              ))}
            </div>
          ) : null}
          <div className="row wrap">
            <span className="badge">{selectedPatches.size} selected</span>
            <button
              type="button"
              className="btn ghost small"
              onClick={() => onSelectedPatchesChange(new Set())}
              disabled={selectedPatches.size === 0}
            >
              clear
            </button>
            <button type="button" className="btn ghost small" onClick={selectTop64} disabled={!attr}>
              select top-64 by heat
            </button>
            <span className="muted small">
              click or drag on the grid to paint patches (click a painted patch again to remove it); the selection feeds the
              knockout tool
            </span>
          </div>
        </div>
      </section>

      <section className="panel">
        <header className="panel-header">
          <h2>patch readout</h2>
          <span className="muted small">
            {focus !== null
              ? `patch ${focus} · r${Math.floor(focus / cols)} c${focus % cols}${focusQuarter !== undefined ? ` · ${QUARTER_NAMES[focusQuarter]}` : ''}`
              : 'click a patch to inspect it'}
          </span>
        </header>
        <div className="panel-body">
          <div className="row wrap">
            <button
              type="button"
              className="btn"
              onClick={() => {
                if (focus !== null) void fetchPatchLens(focus);
              }}
              disabled={focus === null || !lensOk}
            >
              read out focused patch
            </button>
            <span className="muted small">
              lens target <span className="mono">kind=patch</span> · tracked{' '}
              <span className="mono">{token.trim() || '—'}</span>
            </span>
          </div>
          <JobProgress job={patchJob} label="patch lens" error={patchErr} />
          {patchTarget ? (
            <TopKTable
              target={patchTarget}
              tracked={token.trim() ? [token.trim()] : []}
              nLayers={meta?.model.n_layers}
            />
          ) : (
            <div className="empty small">no patch readout yet — click a patch on the heatmap</div>
          )}
          <div className="row wrap">
            <span className="muted small">selected patches ({selection.length})</span>
            {selection.slice(0, 48).map((idx) => (
              <button
                key={idx}
                type="button"
                className="chip mono"
                title="remove from selection"
                onClick={() => removePatches([idx])}
              >
                {idx} ×
              </button>
            ))}
            {selection.length > 48 ? <span className="muted small">+{selection.length - 48} more</span> : null}
          </div>
        </div>
      </section>
    </div>
  );
}
