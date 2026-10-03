import { FIXTURES_MODE } from '../lib/api';
import type { Meta } from '../lib/types';

export interface MetaBarProps {
  meta: Meta | null;
  error: string | null;
  loading: boolean;
  onRetry: () => void;
}

function trainedSamples(meta: Meta): { max: number; breakdown: string } {
  const entries = Object.entries(meta.lens.n_prompts);
  if (entries.length === 0) return { max: 0, breakdown: 'no prompt counts reported' };
  const max = entries.reduce((acc, [, n]) => Math.max(acc, n), 0);
  const breakdown = entries.map(([mask, n]) => `${mask}: ${n}`).join(' · ');
  return { max, breakdown };
}

/**
 * Top status strip: model / lens provenance, fixture and mock badges, job and
 * GPU counters. Renders the backend-offline banner when `/api/meta` failed.
 */
export function MetaBar({ meta, error, loading, onRetry }: MetaBarProps) {
  if (!meta) {
    return (
      <div className="meta-bar">
        <div className="banner error">
          <div className="banner-head">
            <b>Backend offline</b>
            {loading ? <span className="muted"> — connecting…</span> : null}
          </div>
          {error ? <div className="mono small">{error}</div> : null}
          <div className="hint small">
            Expected at <code>http://127.0.0.1:8787</code> (the dev server proxies <code>/api</code> there; the backend
            README documents how to serve the built <code>dist/</code>). Or explore offline with{' '}
            <code>?fixtures=1</code>.
          </div>
          <button type="button" className="btn" onClick={onRetry} disabled={loading}>
            {loading ? 'connecting…' : 'retry'}
          </button>
        </div>
      </div>
    );
  }
  const { max, breakdown } = trainedSamples(meta);
  const gpu = meta.gpu;
  return (
    <div className="meta-bar">
      {FIXTURES_MODE ? (
        <span className="meta-chip chip-fixtures" title="All API calls are served from src/fixtures — no backend involved">
          fixtures
        </span>
      ) : null}
      <span className="meta-chip" title="backend mode">
        mode <b>{meta.mode}</b>
      </span>
      <span className="meta-chip" title="model">
        {meta.model.name}
      </span>
      <span className="meta-chip mono" title="device / dtype">
        {meta.device} · {meta.dtype}
      </span>
      <span
        className="meta-chip mono"
        title={`layers · width · image grid · vocabulary`}
      >
        {meta.model.n_layers}L · d{meta.model.d_model} · {meta.model.grid[0]}×{meta.model.grid[1]} · vocab{' '}
        {meta.model.vocab_size}
      </span>
      {meta.lens.mock ? (
        <span className="meta-chip chip-mock" title="The lens weights are mock/synthetic — readouts are not from a trained Jacobian">
          MOCK LENS
        </span>
      ) : null}
      <span
        className={`meta-chip ${meta.lens.available ? 'chip-ok' : 'chip-error'}`}
        title={[`dir: ${meta.lens.dir}`, breakdown, meta.lens.notes ?? ''].filter(Boolean).join('\n')}
      >
        lens {meta.lens.available ? 'available' : 'unavailable'} · {meta.lens.source_layers.length} layers · masks{' '}
        {meta.lens.masks.join('/')}
      </span>
      <span className="meta-chip" title={`prompts per mask — ${breakdown}`}>
        lens trained on <b className="mono">{max}</b>/50 samples
      </span>
      {meta.lens.error ? (
        <span className="meta-chip chip-error" title={meta.lens.error}>
          lens error
        </span>
      ) : null}
      <span className="meta-chip mono" title="active / queued backend jobs">
        jobs {meta.jobs.active}↑ {meta.jobs.queued}…
      </span>
      {gpu ? (
        <span className="meta-chip mono" title="GPU memory free / total">
          GPU {(gpu.free_mb / 1024).toFixed(1)}/{(gpu.total_mb / 1024).toFixed(1)} GiB
        </span>
      ) : (
        <span className="meta-chip muted mono">no GPU</span>
      )}
      <button type="button" className="btn ghost small" onClick={onRetry} disabled={loading} title="refresh /api/meta">
        {loading ? '…' : 'refresh'}
      </button>
    </div>
  );
}
