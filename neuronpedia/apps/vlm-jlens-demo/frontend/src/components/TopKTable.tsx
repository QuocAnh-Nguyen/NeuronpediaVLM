import { displayToken } from './TokenChips';
import type { LensTargetResult, LensTopKEntry, LensTrackedEntry } from '../lib/types';

/** Cell shading for the readout matrix. */
export type TopKColorMode = 'prob' | 'rank';

/** Hint attached to the model's own logits (model row and the final lens row). */
const MODEL_HINT = "the model's own logits - not lens evidence";

export interface TopKTableProps {
  target: LensTargetResult;
  /** Token strings to highlight (the tracked tokens sent to POST /api/lens). */
  tracked?: string[];
  /** Clicking a token cell fills the caller's token input. */
  onPickToken?: (str: string) => void;
  /** Total model layers; the lens row at `n_layers - 1` is labelled like the model row. */
  nLayers?: number;
  /** `prob` shades by the layer's top-1 share (default); `rank` shades by 1-based rank. */
  colorMode?: TopKColorMode;
}

interface ReadoutRow {
  key: string;
  label: string;
  cls: string;
  tag?: string;
  hint?: string;
  topk: LensTopKEntry[];
  trackedMap: Record<string, LensTrackedEntry>;
}

/**
 * Lens readout matrix: one row per layer, one column per top-k rank slot.
 * Each cell shows the token and its probability as a bar; cells matching a
 * tracked token are highlighted, and the rightmost column summarizes the
 * tracked token's rank / probability for every layer. The model's own logits
 * (`model_row`) are appended as a visually distinct, non-lens row, and the
 * final lens layer — which reproduces them — is labelled the same way.
 */
export function TopKTable({ target, tracked = [], onPickToken, nLayers, colorMode = 'prob' }: TopKTableProps) {
  const rows = target.per_layer;
  if (rows.length === 0) return <div className="empty small">no layers in this readout</div>;
  const k = rows.reduce((acc, row) => Math.max(acc, row.topk.length), 0);
  const primary = tracked[0];
  const finalLayer = nLayers !== undefined ? nLayers - 1 : rows[rows.length - 1].layer;
  const readoutRows: ReadoutRow[] = rows.map((row) => ({
    key: `L${row.layer}`,
    label: `L${row.layer}`,
    cls: `lens-row${row.layer === finalLayer ? ' model-layer-row' : ''}`,
    ...(row.layer === finalLayer ? { tag: '= model output', hint: MODEL_HINT } : {}),
    topk: row.topk,
    trackedMap: row.tracked,
  }));
  readoutRows.push({
    key: 'model',
    label: 'model output',
    cls: 'model-row',
    hint: MODEL_HINT,
    topk: target.model_row?.topk ?? [],
    trackedMap: target.model_row?.tracked ?? {},
  });
  return (
    <div className="table-wrap">
      <table className="table topk-table">
        <thead>
          <tr>
            <th className="layer-cell">layer</th>
            {Array.from({ length: k }, (_, j) => (
              <th key={j} className="mono">
                top-{j + 1}
              </th>
            ))}
            {primary ? <th className="tracked-cell">tracked {displayToken(primary)}</th> : null}
          </tr>
        </thead>
        <tbody>
          {readoutRows.map((row) => {
            const rowMax = row.topk.reduce((acc, e) => Math.max(acc, e.prob), 1e-9);
            const trackedEntry = primary ? row.trackedMap[primary] : undefined;
            return (
              <tr key={row.key} className={row.cls} title={row.hint}>
                <td className="layer-cell mono">
                  {row.label}
                  {row.tag ? <span className="row-tag">{row.tag}</span> : null}
                </td>
                {Array.from({ length: k }, (_, j) => {
                  const entry = row.topk[j];
                  if (!entry) return <td key={j} className="empty-cell" />;
                  const hit = tracked.includes(entry.str);
                  const pct =
                    colorMode === 'rank'
                      ? Math.max(4, Math.round(((k - entry.rank + 1) / k) * 100))
                      : Math.max(0, Math.min(100, (entry.prob / rowMax) * 100));
                  return (
                    <td key={j} className={`tok-cell-wrap${hit ? ' hit' : ''}`}>
                      <button
                        type="button"
                        className="tok-cell"
                        style={{
                          background: `linear-gradient(90deg, rgba(109, 124, 255, 0.42) ${pct}%, rgba(255, 255, 255, 0.03) ${pct}%)`,
                        }}
                        title={`${entry.str} · prob=${entry.prob.toFixed(4)} · logit=${entry.logit.toFixed(2)} · rank=${entry.rank}`}
                        onClick={onPickToken ? () => onPickToken(entry.str) : undefined}
                        disabled={!onPickToken}
                      >
                        <span className="tok-str">{displayToken(entry.str)}</span>
                        <span className="tok-prob mono">{entry.prob.toFixed(3)}</span>
                      </button>
                    </td>
                  );
                })}
                {primary ? (
                  <td className={`tracked-cell${trackedEntry && trackedEntry.rank <= k ? ' in-topk' : ''}`}>
                    {trackedEntry ? (
                      <span
                        className="tracked-summary mono"
                        title={`rank=${trackedEntry.rank} · prob=${trackedEntry.prob.toFixed(4)} · logit=${trackedEntry.logit.toFixed(2)}`}
                      >
                        <span className="rank-badge">#{trackedEntry.rank}</span>
                        <span className="tok-prob">{trackedEntry.prob.toFixed(4)}</span>
                        <span
                          className="mini-bar"
                          style={{ width: `${Math.max(2, Math.min(100, trackedEntry.prob * 100))}%` }}
                        />
                      </span>
                    ) : (
                      <span className="muted small">n/a</span>
                    )}
                  </td>
                ) : null}
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="muted small table-note">
        {colorMode === 'rank'
          ? 'bar = 1-based rank share within the row (rank 1 = widest)'
          : "bar = probability relative to that layer's top-1"}{' '}
        · highlighted cells match the tracked token · <span className="mono">model output</span> = {MODEL_HINT}
      </div>
    </div>
  );
}
