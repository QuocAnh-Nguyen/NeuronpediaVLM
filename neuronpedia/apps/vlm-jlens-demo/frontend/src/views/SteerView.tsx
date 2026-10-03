import { useEffect, useState } from 'react';
import { errorMessage, knockout, pollJob, steer } from '../lib/api';
import { signedColor } from '../lib/colormap';
import type {
  Flag,
  GenerateResult,
  Job,
  KnockoutMode,
  KnockoutOutcome,
  KnockoutResult,
  Meta,
  SessionResponse,
  SteerEdit,
  SteerMode,
  SteerPositions,
  SteerRequest,
  SteerResult,
} from '../lib/types';
import { CaptionDiff } from '../components/CaptionDiff';
import { JobProgress } from '../components/JobProgress';
import { TokenChips, displayToken } from '../components/TokenChips';

const MODE_HINTS: Record<SteerMode, string> = {
  add: 'push the residual stream toward the token’s lens direction',
  ablate: 'remove the token’s lens direction',
  swap: 'replace the source token’s lens direction with the target’s',
};

const KO_MODES: KnockoutMode[] = ['zero', 'mean'];

/** Quick alpha picks (residual-relative multiplier). */
const ALPHA_PRESETS = [0.5, 1, 2, 3];

function describeEdit(e: SteerEdit): string {
  const pos = Array.isArray(e.positions) ? `${e.positions.length} position(s)` : e.positions;
  const core = e.mode === 'swap' ? `${e.source_token ?? '?'} → ${e.token ?? '?'}` : `${e.token ?? '?'}`;
  return `L${e.layer} · ${e.mode} · ${core} · α=${e.alpha} · ${pos}`;
}

export interface SteerViewProps {
  meta: Meta | null;
  session: SessionResponse | null;
  generated: GenerateResult | null;
  flags: Flag[];
  /** Token to steer by default (flagged token or selected caption token). */
  defaultToken: string;
  selectedPatches: Set<number>;
  onSelectedPatchesChange: (next: Set<number>) => void;
}

/**
 * View 4 — causal interventions. Left panel applies a lens-vector steer edit
 * (add / ablate / swap, with alpha and positions) and diffs the caption before
 * vs after. Right panel knocks out the patches painted in the Patch tab and
 * reports the per-token log-probability deltas.
 */
export function SteerView({
  meta,
  session,
  generated,
  flags,
  defaultToken,
  selectedPatches,
  onSelectedPatchesChange,
}: SteerViewProps) {
  const [layer, setLayer] = useState(0);
  const [mode, setMode] = useState<SteerMode>('add');
  const [token, setToken] = useState(defaultToken);
  const [sourceToken, setSourceToken] = useState('');
  const [alpha, setAlpha] = useState(2);
  const [positions, setPositions] = useState<SteerPositions>('last');
  const [maxNew, setMaxNew] = useState(0);
  const [steerJob, setSteerJob] = useState<Job<SteerResult> | null>(null);
  const [steerResult, setSteerResult] = useState<SteerResult | null>(null);
  const [steerErr, setSteerErr] = useState<string | null>(null);
  const [koMode, setKoMode] = useState<KnockoutMode>('zero');
  const [koMaxNew, setKoMaxNew] = useState(0);
  const [hideZero, setHideZero] = useState(true);
  const [koJob, setKoJob] = useState<Job<KnockoutResult> | null>(null);
  const [koResult, setKoResult] = useState<KnockoutResult | null>(null);
  const [koErr, setKoErr] = useState<string | null>(null);
  const [tokenEdited, setTokenEdited] = useState(false);


  const sourceLayers = meta?.lens.source_layers ?? [];
  const lensOk = meta?.lens.available ?? false;
  const maxLayer = Math.max(0, (meta?.model.n_layers ?? 1) - 1);
  const maxTokens = meta?.capabilities.max_new_tokens ?? 64;

  const steerModeOptions =
    meta?.capabilities.steer_modes && meta.capabilities.steer_modes.length > 0
      ? meta.capabilities.steer_modes
      : (['add', 'ablate', 'swap'] as SteerMode[]);

  useEffect(() => {
    if (!tokenEdited && defaultToken) setToken(defaultToken);
  }, [defaultToken, tokenEdited]);

  useEffect(() => {
    if (sourceLayers.length > 0 && !sourceLayers.includes(layer)) setLayer(sourceLayers[sourceLayers.length - 1]);
  }, [sourceLayers, layer]);


  const selection = [...selectedPatches].sort((a, b) => a - b);
  const canApply =
    Boolean(session) &&
    lensOk &&
    sourceLayers.length > 0 &&
    (mode === 'swap' ? token.trim().length > 0 && sourceToken.trim().length > 0 : token.trim().length > 0);

  async function applySteer() {
    if (!session) return;
    setSteerErr(null);
    setSteerJob(null);
    const body: SteerRequest = {
      session_id: session.session_id,
      layer,
      mode,
      alpha,
      positions,
      token: token.trim(),
      ...(mode === 'swap' ? { source_token: sourceToken.trim() } : {}),
      ...(maxNew > 0 ? { max_new_tokens: maxNew } : {}),
    };
    try {
      const { job_id } = await steer(body);
      const done = await pollJob<SteerResult>(job_id, setSteerJob);
      if (done.result) setSteerResult(done.result);
    } catch (e) {
      setSteerErr(errorMessage(e));
    }
  }

  async function runKnockout() {
    if (!session || selection.length === 0) return;
    setKoErr(null);
    setKoJob(null);
    try {
      const { job_id } = await knockout({
        session_id: session.session_id,
        patches: selection,
        mode: koMode,
        ...(koMaxNew > 0 ? { max_new_tokens: koMaxNew } : {}),
      });
      const done = await pollJob<KnockoutResult>(job_id, setKoJob);
      if (done.result) setKoResult(done.result);
    } catch (e) {
      setKoErr(errorMessage(e));
    }
  }

  if (!session) {
    return (
      <div className="panel">
        <div className="panel-body empty">
          Create a session in the <b>Session</b> tab first — then steer lens directions and knock out patches here.
        </div>
      </div>
    );
  }

  const deltas = koResult ? [...koResult.deltas].sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta)) : [];
  const shownDeltas = hideZero ? deltas.filter((d) => d.delta !== 0) : deltas;
  const maxAbs = deltas.reduce((acc, d) => Math.max(acc, Math.abs(d.delta)), 0);

  const unstableEdit = steerResult?.diagnostics.some((d) => d.cond !== null && d.cond > 1e3) ?? false;
  const outcomeByToken = new Map<string, KnockoutOutcome>(
    (koResult?.outcomes ?? []).map((o: KnockoutOutcome): [string, KnockoutOutcome] => [o.token, o]),
  );

  return (
    <div className="view-grid">
      <section className="panel">
        <header className="panel-header">
          <h2>lens-vector steering</h2>
          {!lensOk ? <span className="badge warn">lens unavailable</span> : <span className="muted small">edit the residual stream, then re-decode</span>}
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
              <span>mode</span>
              <select value={mode} onChange={(e) => setMode(e.target.value as SteerMode)}>
                {steerModeOptions.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
              <span className="muted small">{MODE_HINTS[mode]}</span>
            </label>
            <label className="field">
              <span>token</span>
              <input
                className="mono"
                list="steer-token-list"
                value={token}
                placeholder="e.g. garden"
                onChange={(e) => {
                  setTokenEdited(true);
                  setToken(e.target.value);
                }}
              />
              <datalist id="steer-token-list">
                {generated?.tokens.map((t) => (
                  <option key={t.i} value={t.str}>{`caption i=${t.i}`}</option>
                ))}
              </datalist>
            </label>
            {mode === 'swap' ? (
              <label className="field">
                <span>source_token</span>
                <input
                  className="mono"
                  value={sourceToken}
                  placeholder="direction to replace"
                  onChange={(e) => setSourceToken(e.target.value)}
                />
              </label>
            ) : null}
            <label className="field">
              <span>
                alpha <b className="mono">{alpha.toFixed(2)}</b>
              </span>
              <input type="range" min={0} max={8} step={0.25} value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} />
              <span className="mode-toggle" role="group" aria-label="alpha presets">
                {ALPHA_PRESETS.map((k) => (
                  <button
                    key={k}
                    type="button"
                    className={`btn ghost small${alpha === k ? ' active' : ''}`}
                    onClick={() => setAlpha(k)}
                    title={`set residual-relative alpha to ${k}`}
                  >
                    {k}
                  </button>
                ))}
              </span>
              <span className="muted small">
                residual-relative alpha: the applied edit is scaled by the residual norm at each position
              </span>
            </label>
            <label className="field">
              <span>positions</span>
              <select value={positions} onChange={(e) => setPositions(e.target.value as SteerPositions)}>
                <option value="last">last</option>
                <option value="all">all</option>
              </select>
              <span className="muted small">which generated positions the edit applies to</span>
            </label>
            <label className="field">
              <span>max_new_tokens</span>
              <input
                type="number"
                className="mono"
                min={0}
                max={maxTokens}
                value={maxNew}
                onChange={(e) => setMaxNew(Math.max(0, Math.min(maxTokens, Number(e.target.value) || 0)))}
              />
              <span className="muted small">0 = backend default</span>
            </label>
            <button type="button" className="btn primary" onClick={applySteer} disabled={!canApply}>
              apply steering
            </button>
          </div>
          {flags.length > 0 ? (
            <div className="row wrap">
              <span className="muted small">flagged:</span>
              {flags.map((f) => (
                <button key={f.str} type="button" className="chip mono" onClick={() => setToken(f.str)}>
                  {f.str}
                </button>
              ))}
            </div>
          ) : null}
          <JobProgress job={steerJob} label="steer" error={steerErr} />
          {steerResult ? (
            <div className="generated-block">
              <div className="muted small mono">
                {steerResult.edits.map(describeEdit).join('  |  ') || 'no edits echoed'} · {steerResult.n_edit_forwards}{' '}
                edit-forward pass(es)
              </div>
              {steerResult.diagnostics.length > 0 ? (
                <>
                  <div className="row wrap">
                    <span className="muted small">edit diagnostics</span>
                    {unstableEdit ? <span className="badge err">nearly collinear pair - swap unstable</span> : null}
                  </div>
                  <div className="table-wrap">
                    <table className="table diag-table">
                      <thead>
                        <tr>
                          <th>layer</th>
                          <th>mode</th>
                          <th className="mono">v_norm</th>
                          <th className="mono">h_norm</th>
                          <th className="mono">cond</th>
                        </tr>
                      </thead>
                      <tbody>
                        {steerResult.diagnostics.map((d, i) => (
                          <tr
                            key={`${d.layer}-${d.mode}-${i}`}
                            className={d.cond !== null && d.cond > 1e3 ? 'row-warn' : undefined}
                          >
                            <td className="mono">L{d.layer}</td>
                            <td className="mono">{d.mode}</td>
                            <td className="mono">{d.v_norm.toFixed(3)}</td>
                            <td className="mono">{d.h_norm === null ? '—' : d.h_norm.toFixed(3)}</td>
                            <td
                              className="mono"
                              title={d.cond !== null && d.cond > 1e3 ? 'nearly collinear pair - swap unstable' : undefined}
                            >
                              {d.cond === null ? '—' : d.cond.toFixed(1)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <div className="muted small table-note">
                      v_norm = norm of the applied lens vector · h_norm = residual norm at the edit site (— = not measured) ·
                      cond = condition number of the 2-direction edit basis; cond &gt; 1e3 means the pair is nearly collinear
                      and the swap is numerically unstable
                    </div>
                  </div>
                </>
              ) : null}
              <CaptionDiff before={steerResult.caption_before} after={steerResult.caption_after} labels={['before', 'after']} />
              <div className="muted small">tokens (after)</div>
              <TokenChips tokens={steerResult.tokens_after} showIndices />
            </div>
          ) : (
            <div className="empty small">no steering run yet</div>
          )}
        </div>
      </section>

      <section className="panel">
        <header className="panel-header">
          <h2>causal knockout</h2>
          <span className="badge">{selection.length} patches</span>
        </header>
        <div className="panel-body">
          <div className="attr-controls">
            <label className="field">
              <span>mode</span>
              <select value={koMode} onChange={(e) => setKoMode(e.target.value as KnockoutMode)}>
                {KO_MODES.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
              <span className="muted small">zero = blank the patch features · mean = replace with the prompt mean</span>
            </label>
            <label className="field">
              <span>max_new_tokens</span>
              <input
                type="number"
                className="mono"
                min={0}
                max={maxTokens}
                value={koMaxNew}
                onChange={(e) => setKoMaxNew(Math.max(0, Math.min(maxTokens, Number(e.target.value) || 0)))}
              />
            </label>
            <button type="button" className="btn primary" onClick={runKnockout} disabled={selection.length === 0}>
              run knockout
            </button>
          </div>
          <div className="row wrap">
            <span className="muted small">painted patches:</span>
            {selection.slice(0, 40).map((idx) => (
              <button
                key={idx}
                type="button"
                className="chip mono"
                title="remove from selection"
                onClick={() => onSelectedPatchesChange(new Set(selection.filter((x) => x !== idx)))}
              >
                {idx} ×
              </button>
            ))}
            {selection.length > 40 ? <span className="muted small">+{selection.length - 40} more</span> : null}
            {selection.length === 0 ? (
              <span className="muted small">paint patches in the Patch tab to enable knockout</span>
            ) : (
              <button type="button" className="btn ghost small" onClick={() => onSelectedPatchesChange(new Set())}>
                clear
              </button>
            )}
          </div>
          <JobProgress job={koJob} label="knockout" error={koErr} />
          {koResult ? (
            <div className="generated-block">
              <div className="muted small mono">
                {koResult.patches.length} patches · mode {koResult.mode} · caption after
              </div>
              <CaptionDiff before={koResult.baseline_caption} after={koResult.caption_after} labels={['baseline', 'knocked out']} />
              <div className="row wrap">
                <span className="muted small">
                  per-token log-probability change on the baseline continuation (positive = knocked-out patches made the
                  token <i>more</i> likely)
                </span>
                <label className="field inline">
                  <input type="checkbox" checked={hideZero} onChange={(e) => setHideZero(e.target.checked)} />
                  <span>hide zero deltas</span>
                </label>
              </div>
              <div className="table-wrap">
                <table className="table ko-table">
                  <thead>
                    <tr>
                      <th>token</th>
                      <th className="mono">logprob before</th>
                      <th className="mono">logprob after</th>
                      <th className="mono">Δ</th>
                      <th>magnitude</th>
                      <th title={koResult.classification_note}>outcome</th>
                    </tr>
                  </thead>
                  <tbody>
                    {shownDeltas.map((d) => {
                      const outcome = outcomeByToken.get(d.token);
                      return (
                        <tr key={d.token} className={outcome ? `outcome-row ${outcome.class}` : undefined}>
                          <td className="mono">{displayToken(d.token)}</td>
                          <td className="mono">{d.logprob_before.toFixed(3)}</td>
                          <td className="mono">{d.logprob_after.toFixed(3)}</td>
                          <td className={`mono delta ${d.delta >= 0 ? 'pos' : 'neg'}`}>
                            {d.delta >= 0 ? '+' : ''}
                            {d.delta.toFixed(3)}
                          </td>
                          <td>
                            <span className="delta-track" title={`|Δ| = ${Math.abs(d.delta).toFixed(4)}`}>
                              <span
                                className={`delta-fill ${d.delta >= 0 ? 'pos' : 'neg'}`}
                                style={{
                                  background: signedColor(d.delta, maxAbs),
                                  width: `${maxAbs > 0 ? Math.max(2, (Math.abs(d.delta) / maxAbs) * 50) : 0}%`,
                                }}
                              />
                            </span>
                          </td>
                          <td>
                            {outcome ? (
                              <span className={`outcome-tag ${outcome.class}`} title={koResult.classification_note}>
                                {outcome.class}
                              </span>
                            ) : (
                              <span className="muted small">—</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                {shownDeltas.length === 0 ? <div className="empty small">no non-zero deltas reported</div> : null}
                <div className="muted small table-note" title={koResult.classification_note}>
                  {koResult.classification_note}
                </div>
              </div>
              <details>
                <summary className="small muted">tokens after knockout</summary>
                <TokenChips tokens={koResult.tokens_after} showIndices />
              </details>
            </div>
          ) : (
            <div className="empty small">no knockout run yet</div>
          )}
        </div>
      </section>
    </div>
  );
}
