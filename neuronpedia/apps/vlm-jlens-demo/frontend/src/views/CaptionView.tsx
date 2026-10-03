import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createSession, errorMessage, generate, lens, pollJob } from '../lib/api';
import type {
  Flag,
  GenerateResult,
  Job,
  LensResult,
  LensTargetSpec,
  LensTrackedEntry,
  Meta,
  SessionResponse,
} from '../lib/types';
import { JobProgress } from '../components/JobProgress';
import { LayerChart } from '../components/LayerChart';
import type { LayerPoint } from '../components/LayerChart';
import { TokenChips, displayToken, plainToken } from '../components/TokenChips';
import type { ChipToken } from '../components/TokenChips';
import { TopKTable } from '../components/TopKTable';
import type { TopKColorMode } from '../components/TopKTable';

/** Result of the no-image twin comparison. */
interface TwinState {
  session: SessionResponse;
  generated: GenerateResult;
  /** Image session's readout over the same token set, for the per-token comparison. */
  baselineLens: LensResult;
  /** Twin (no_image) session's readout. */
  lens: LensResult;
}

/** Last-layer tracked entry for `str`, from the first target that tracked it. */
function lastTrackedProb(res: LensResult, str: string): LensTrackedEntry | null {
  for (const tgt of res.targets) {
    for (let i = tgt.per_layer.length - 1; i >= 0; i -= 1) {
      const entry = tgt.per_layer[i].tracked[str];
      if (entry) return entry;
    }
  }
  return null;
}

export interface CaptionViewProps {
  meta: Meta | null;
  session: SessionResponse | null;
  generated: GenerateResult | null;
  selectedTokenIndex: number | null;
  onSelectToken: (token: ChipToken) => void;
  flags: Flag[];
  onFlagsChange: (flags: Flag[]) => void;
}

/**
 * View 2 — generated-caption token chips. Clicking a token runs
 * POST /api/lens for that `gen` target with topk tokens and the token itself
 * tracked; the result renders as a layer × top-k matrix plus a tracked
 * probability curve. Flagged tokens carry researcher notes and export as text.
 */
export function CaptionView({
  meta,
  session,
  generated,
  selectedTokenIndex,
  onSelectToken,
  flags,
  onFlagsChange,
}: CaptionViewProps) {
  const [job, setJob] = useState<Job<LensResult> | null>(null);
  const [result, setResult] = useState<LensResult | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [targetIdx, setTargetIdx] = useState(0);
  const [topk, setTopk] = useState(8);
  const [copied, setCopied] = useState(false);
  const [colorModeChoice, setColorModeChoice] = useState<'auto' | TopKColorMode>('auto');
  const [twin, setTwin] = useState<TwinState | null>(null);
  const [twinJob, setTwinJob] = useState<Job<unknown> | null>(null);
  const [twinErr, setTwinErr] = useState<string | null>(null);
  const twinReqRef = useRef(0);
  const reqRef = useRef(0);

  const selectedToken = useMemo(
    () => generated?.tokens.find((t) => t.i === selectedTokenIndex) ?? null,
    [generated, selectedTokenIndex],
  );
  const tracked = selectedToken ? [selectedToken.str] : [];
  const lensOk = meta?.lens.available ?? false;

  const runLens = useCallback(
    async (targets: LensTargetSpec[], trackTokens: string[]) => {
      if (!session) return;
      const reqId = reqRef.current + 1;
      reqRef.current = reqId;
      setErr(null);
      setJob(null);
      try {
        const { job_id } = await lens({
          session_id: session.session_id,
          targets,
          topk,
          track: trackTokens,
        });
        const done = await pollJob<LensResult>(job_id, setJob);
        if (reqRef.current === reqId && done.result) {
          setResult(done.result);
          setTargetIdx(0);
        }
      } catch (e) {
        if (reqRef.current === reqId) setErr(errorMessage(e));
      }
    },
    [session, topk],
  );

  useEffect(() => {
    if (!session || !selectedToken) return;
    void runLens([{ kind: 'gen', i: selectedToken.i }], [selectedToken.str]);
  }, [session, selectedToken, runLens]);

  const target = result ? result.targets[Math.min(targetIdx, result.targets.length - 1)] ?? null : null;
  const points: LayerPoint[] = [];
  if (target && tracked.length > 0) {
    for (const row of target.per_layer) {
      const entry = row.tracked[tracked[0]];
      points.push({ layer: row.layer, prob: entry?.prob ?? 0, rank: entry?.rank });
    }
  }
  const ranks = points.map((p) => p.rank).filter((r): r is number => r !== undefined);

  // Rank shading is the default whenever the matrix compares several layers.
  const colorMode: TopKColorMode =
    colorModeChoice === 'auto' ? (target && target.per_layer.length > 1 ? 'rank' : 'prob') : colorModeChoice;
  const compareTokens = flags.length > 0 ? flags.map((f) => f.str) : tracked;

  function readAll() {
    if (!generated) return;
    void runLens(
      generated.tokens.map((t) => ({ kind: 'gen', i: t.i })),
      tracked,
    );
  }

  function trackedInfo(str: string): LensTrackedEntry | null {
    if (!result) return null;
    const tgt = result.targets[Math.min(targetIdx, result.targets.length - 1)];
    if (!tgt) return null;
    for (let i = tgt.per_layer.length - 1; i >= 0; i -= 1) {
      const entry = tgt.per_layer[i].tracked[str];
      if (entry) return entry;
    }
    return null;
  }

  async function runTwin() {
    if (!session || !generated) return;
    const reqId = twinReqRef.current + 1;
    twinReqRef.current = reqId;
    setTwinErr(null);
    setTwinJob(null);
    try {
      const twinSession = await createSession({ prompt: session.prompt_text, variant: 'no_image' });
      const trackTokens = [...new Set([...flags.map((f) => f.str), ...tracked])];
      // Image session over the same token set, so both variants are comparable.
      const baseRef = await lens({
        session_id: session.session_id,
        targets: generated.tokens.map((t) => ({ kind: 'gen', i: t.i })),
        topk,
        track: trackTokens,
      });
      const baseDone = await pollJob<LensResult>(baseRef.job_id, setTwinJob);
      const genRef = await generate({
        session_id: twinSession.session_id,
        max_new_tokens: Math.max(1, generated.tokens.length),
      });
      const genDone = await pollJob<GenerateResult>(genRef.job_id, setTwinJob);
      if (!baseDone.result || !genDone.result) throw new Error('no-image twin: missing job result');
      const twinGen = genDone.result;
      const twinRef = await lens({
        session_id: twinSession.session_id,
        targets: twinGen.tokens.map((t) => ({ kind: 'gen', i: t.i })),
        topk,
        track: trackTokens,
      });
      const twinDone = await pollJob<LensResult>(twinRef.job_id, setTwinJob);
      if (reqId !== twinReqRef.current) return;
      if (!twinDone.result) throw new Error('no-image twin: lens readout failed');
      setTwin({ session: twinSession, generated: twinGen, baselineLens: baseDone.result, lens: twinDone.result });
    } catch (e) {
      if (reqId === twinReqRef.current) setTwinErr(errorMessage(e));
    }
  }
  const exportText = useMemo(() => {
    const lines = [
      '# VLM J-Lens flagged tokens',
      `# session: ${session?.session_id ?? 'none'}`,
      `# model: ${meta?.model.name ?? '?'} · lens: ${meta?.lens.dir ?? '?'}`,
      '# token\trank@lastLayer\tprob@lastLayer\tnote',
    ];
    for (const f of flags) {
      const info = trackedInfo(f.str);
      lines.push([plainToken(f.str), info ? info.rank : '—', info ? info.prob.toFixed(4) : '—', f.note].join('\t'));
    }
    return lines.join('\n');
  }, [flags, result, targetIdx, session, meta]);

  async function copyExport() {
    try {
      await navigator.clipboard.writeText(exportText);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  }

  function flagSelected() {
    if (!selectedToken) return;
    if (flags.some((f) => f.str === selectedToken.str)) return;
    onFlagsChange([...flags, { str: selectedToken.str, tokenIndex: selectedToken.i, note: '' }]);
  }

  if (!session || !generated) {
    return (
      <div className="panel">
        <div className="panel-body empty">
          Create a session and generate a caption in the <b>Session</b> tab first — its tokens become clickable here.
        </div>
      </div>
    );
  }

  return (
    <div className="view-2col">
      <section className="panel">
        <header className="panel-header">
          <h2>caption tokens</h2>
          <span className="muted small">
            {generated.tokens.length} tokens · click one to read out the lens
          </span>
        </header>
        <div className="panel-body">
          <TokenChips tokens={generated.tokens} selected={selectedTokenIndex} flagged={flags.map((f) => f.str)} onSelect={onSelectToken} showIndices />
          <div className="row wrap">
            <button type="button" className="btn" onClick={readAll} disabled={!lensOk || generated.tokens.length === 0}>
              read out all tokens
            </button>
            <label className="field inline">
              <span>topk</span>
              <input
                type="number"
                className="mono"
                min={1}
                max={32}
                value={topk}
                onChange={(e) => setTopk(Math.max(1, Math.min(32, Number(e.target.value) || 8)))}
              />
            </label>
            <span className="mode-toggle" role="group" aria-label="cell shading">
              <span className="muted small">color</span>
              <button
                type="button"
                className={`btn ghost small${colorMode === 'prob' ? ' active' : ''}`}
                onClick={() => setColorModeChoice('prob')}
                title="shade cells by probability relative to the layer's top-1"
              >
                prob
              </button>
              <button
                type="button"
                className={`btn ghost small${colorMode === 'rank' ? ' active' : ''}`}
                onClick={() => setColorModeChoice('rank')}
                title="shade cells by 1-based rank within the row (default when comparing layers)"
              >
                rank
              </button>
            </span>
            {selectedToken ? (
              <span className="muted small">
                selected i={selectedToken.i} · <span className="mono">{selectedToken.str}</span> · logprob{' '}
                {selectedToken.logprob.toFixed(3)}
              </span>
            ) : (
              <span className="muted small">no token selected</span>
            )}
            {!lensOk ? <span className="badge warn">lens unavailable</span> : null}
          </div>
          <JobProgress job={job} label="lens readout" error={err} />
          {result && result.targets.length > 1 ? (
            <div className="row wrap">
              <label className="field inline">
                <span>target</span>
                <select value={targetIdx} onChange={(e) => setTargetIdx(Number(e.target.value))}>
                  {result.targets.map((t, idx) => (
                    <option key={`${t.kind}-${t.i}`} value={idx}>
                      {t.label || `${t.kind}[${t.i}]`}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          ) : null}
          {target ? (
            <>
              <div className="row wrap readout-head">
                <span className="badge">{target.label || `${target.kind}[${target.i}]`}</span>
                <span className="muted small">vocab {result?.vocab_size}</span>
                {points.length > 0 ? (
                  <span className="muted small">
                    shape {points.length} layers × top-{target.per_layer[0]?.topk.length ?? topk}
                  </span>
                ) : null}
                {ranks.length > 0 ? <span className="muted small">best rank #{Math.min(...ranks)}</span> : null}
              </div>
              <TopKTable
                target={target}
                tracked={tracked}
                nLayers={meta?.model.n_layers}
                colorMode={colorMode}
                onPickToken={(str) => {
                  const tok = generated.tokens.find((x) => x.str === str);
                  if (tok) onSelectToken(tok);
                }}
              />
              <LayerChart points={points} title={tracked[0] ? `tracked ${tracked[0]}` : undefined} />
            </>
          ) : (
            <div className="empty small">click a caption token to request its lens readout</div>
          )}
        </div>
      </section>
      <section className="panel">
        <header className="panel-header">
          <h2>flags</h2>
          <span className="muted small">hallucination candidates &amp; notes</span>
        </header>
        <div className="panel-body">
          <div className="row wrap">
            <button type="button" className="btn" onClick={flagSelected} disabled={!selectedToken}>
              flag selected token
            </button>
            <button
              type="button"
              className="btn ghost"
              onClick={() => onFlagsChange([])}
              disabled={flags.length === 0}
            >
              clear all
            </button>
          </div>
          {flags.length === 0 ? (
            <div className="empty small">no flagged tokens yet</div>
          ) : (
            <ul className="flag-list">
              {flags.map((f) => {
                const info = trackedInfo(f.str);
                return (
                  <li key={f.str} className="flag-item">
                    <div className="flag-head">
                      <span className="flag-token mono">{f.str}</span>
                      {f.tokenIndex !== null ? <span className="muted small">i={f.tokenIndex}</span> : null}
                      {info ? (
                        <span className="muted small mono">
                          rank #{info.rank} · p={info.prob.toFixed(4)} @ last layer
                        </span>
                      ) : (
                        <span className="muted small">no readout in view</span>
                      )}
                      <button type="button" className="btn ghost small" onClick={() => onFlagsChange(flags.filter((x) => x.str !== f.str))}>
                        remove
                      </button>
                    </div>
                    <textarea
                      rows={2}
                      className="flag-note"
                      placeholder="note (why is this token suspicious?)"
                      value={f.note}
                      onChange={(e) => onFlagsChange(flags.map((x) => (x.str === f.str ? { ...x, note: e.target.value } : x)))}
                    />
                  </li>
                );
              })}
            </ul>
          )}
          <div className="row">
            <span className="muted small">export</span>
            <button type="button" className="btn ghost small" onClick={copyExport} disabled={flags.length === 0}>
              {copied ? 'copied' : 'copy'}
            </button>
          </div>
          <textarea className="export-box mono" rows={Math.max(4, flags.length + 4)} readOnly value={exportText} />
        </div>
      </section>
      <section className="panel twin-panel">
        <header className="panel-header">
          <h2>no-image twin</h2>
          <span className="badge adapt">[ADAPTATION] prior-gap proxy</span>
        </header>
        <div className="panel-body">
          <div className="row wrap">
            <button
              type="button"
              className="btn"
              onClick={() => void runTwin()}
              disabled={!lensOk || !generated || session.variant === 'no_image'}
            >
              {twin ? 're-run no-image twin' : 'run no-image twin'}
            </button>
            <span className="muted small">
              same prompt, no image block — shows which tokens the language prior alone would carry
            </span>
            {session.variant === 'no_image' ? <span className="badge warn">this session is already a no_image twin</span> : null}
            {twin ? <span className="muted small mono">{twin.session.session_id}</span> : null}
          </div>
          <JobProgress job={twinJob} label="no-image twin" error={twinErr} />
          {twin ? (
            <>
              <div className="twin-grid">
                <div className="twin-col">
                  <div className="twin-head">
                    <span className="badge">image</span>
                    <span className="muted small mono">{session.session_id}</span>
                  </div>
                  <p className="quote">{generated?.caption}</p>
                </div>
                <div className="twin-col">
                  <div className="twin-head">
                    <span className="badge">no_image</span>
                    <span className="muted small mono">{twin.session.session_id}</span>
                  </div>
                  <p className="quote">{twin.generated.caption}</p>
                </div>
              </div>
              {compareTokens.length > 0 ? (
                <div className="table-wrap">
                  <table className="table twin-table">
                    <thead>
                      <tr>
                        <th>token</th>
                        <th className="mono">p image</th>
                        <th className="mono">p no_image</th>
                        <th className="mono">Δ</th>
                      </tr>
                    </thead>
                    <tbody>
                      {compareTokens.map((str) => {
                        const withImage = lastTrackedProb(twin.baselineLens, str);
                        const withoutImage = lastTrackedProb(twin.lens, str);
                        const delta =
                          withImage && withoutImage ? withoutImage.prob - withImage.prob : null;
                        return (
                          <tr key={str}>
                            <td className="mono">{displayToken(str)}</td>
                            <td className="mono">{withImage ? withImage.prob.toFixed(4) : '—'}</td>
                            <td className="mono">{withoutImage ? withoutImage.prob.toFixed(4) : '—'}</td>
                            <td className={`mono delta${delta === null ? '' : delta >= 0 ? ' pos' : ' neg'}`}>
                              {delta === null ? '—' : `${delta >= 0 ? '+' : ''}${delta.toFixed(4)}`}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                  <div className="muted small table-note">
                    [ADAPTATION] prior-gap proxy: Δ &gt; 0 means the token&apos;s last-layer lens mass is larger without the
                    image (prior-driven); Δ &lt; 0 means it collapses without the image (image-driven).
                  </div>
                </div>
              ) : (
                <div className="empty small">
                  flag a caption token to compare its lens probability with and without the image
                </div>
              )}
            </>
          ) : (
            <div className="empty small">
              no twin run yet — creates a second session with <span className="mono">variant=no_image</span>, generates the
              same prompt, and compares the lens readouts
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
