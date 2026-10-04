import { useEffect, useRef, useState } from 'react';
import { createSession, errorMessage, generate, getSamples, getSession, pollJob } from '../lib/api';
import type { GenerateResult, Job, Meta, Sample, SessionInfo, SessionResponse } from '../lib/types';
import { JobProgress } from '../components/JobProgress';
import { TokenChips } from '../components/TokenChips';

const DEFAULT_PROMPT = 'USER: <image>\nDescribe this image in detail.\nASSISTANT:';

export interface SessionViewProps {
  meta: Meta | null;
  session: SessionResponse | null;
  sessionInfo: SessionInfo | null;
  imageB64: string | null;
  generated: GenerateResult | null;
  onSession: (session: SessionResponse, info: SessionInfo | null, imageB64: string) => void;
  onGenerated: (result: GenerateResult) => void;
}

interface Upload {
  b64: string;
  name: string;
  preview: string;
}

/**
 * View 1 — pick a sample image or upload one, set the prompt, create a
 * session, then run greedy generation.
 */
export function SessionView({ meta, session, sessionInfo, imageB64, generated, onSession, onGenerated }: SessionViewProps) {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [samplesErr, setSamplesErr] = useState<string | null>(null);
  const [selectedSampleId, setSelectedSampleId] = useState<string | null>(null);
  const [upload, setUpload] = useState<Upload | null>(null);
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [creating, setCreating] = useState(false);
  const [createErr, setCreateErr] = useState<string | null>(null);
  const [genJob, setGenJob] = useState<Job<GenerateResult> | null>(null);
  const [genErr, setGenErr] = useState<string | null>(null);
  const [maxNew, setMaxNew] = useState(24);
  const fileRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    let live = true;
    getSamples()
      .then((res) => {
        if (!live) return;
        setSamples(res.samples);
        setSamplesErr(null);
        setSelectedSampleId((prev) => prev ?? res.samples[0]?.id ?? null);
      })
      .catch((e: unknown) => {
        if (live) setSamplesErr(errorMessage(e));
      });
    return () => {
      live = false;
    };
  }, []);

  const selectedSample = samples.find((s) => s.id === selectedSampleId) ?? null;
  const chosenB64 = upload?.b64 ?? selectedSample?.image_b64 ?? null;
  const previewSrc = upload?.preview ?? (selectedSample ? `data:image/png;base64,${selectedSample.image_b64}` : null);
  const maxTokens = meta?.capabilities.max_new_tokens ?? 64;

  function handleFile(file: File | undefined) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = typeof reader.result === 'string' ? reader.result : '';
      const b64 = dataUrl.includes(',') ? dataUrl.slice(dataUrl.indexOf(',') + 1) : dataUrl;
      if (!b64) return;
      setUpload({ b64, name: file.name, preview: dataUrl });
      setSelectedSampleId(null);
    };
    reader.readAsDataURL(file);
  }

  async function handleCreate() {
    if (!chosenB64) return;
    setCreating(true);
    setCreateErr(null);
    try {
      const body = {
        image_b64: chosenB64,
        ...(prompt.trim() ? { prompt: prompt.trim() } : {}),
      };
      const created = await createSession(body);
      let info: SessionInfo | null = null;
      try {
        info = await getSession(created.session_id);
      } catch {
        info = null;
      }
      onSession(created, info, chosenB64);
    } catch (e) {
      setCreateErr(errorMessage(e));
    } finally {
      setCreating(false);
    }
  }

  async function handleGenerate() {
    if (!session) return;
    setGenJob(null);
    setGenErr(null);
    try {
      const { job_id } = await generate({ session_id: session.session_id, max_new_tokens: maxNew });
      const done = await pollJob<GenerateResult>(job_id, setGenJob);
      if (done.result) onGenerated(done.result);
    } catch (e) {
      setGenErr(errorMessage(e));
    }
  }

  return (
    <div className="view-grid">
      <section className="panel">
        <header className="panel-header">
          <h2>image</h2>
          <span className="muted small">
            {meta ? `grid ${meta.model.grid[0]}×${meta.model.grid[1]} = ${meta.model.image_seq_length} patches` : 'grid 24×24'}
          </span>
        </header>
        <div className="panel-body">
          {samplesErr ? <div className="banner error small">samples: {samplesErr}</div> : null}
          <div className="thumb-grid">
            {samples.map((s) => (
              <button
                key={s.id}
                type="button"
                className={`thumb${selectedSampleId === s.id && !upload ? ' selected' : ''}`}
                onClick={() => {
                  setSelectedSampleId(s.id);
                  setUpload(null);
                }}
                title={s.label}
              >
                <img src={`data:image/png;base64,${s.image_b64}`} alt={s.label} />
                <span className="thumb-label small">{s.label}</span>
              </button>
            ))}
            {samples.length === 0 && !samplesErr ? <div className="empty small">loading samples…</div> : null}
          </div>
          <div className="row">
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              className="hidden-file"
              onChange={(e) => handleFile(e.target.files?.[0])}
            />
            <button type="button" className="btn" onClick={() => fileRef.current?.click()}>
              upload image…
            </button>
            {upload ? (
              <span className="muted small">
                {upload.name} · {Math.round(upload.b64.length / 1024)} KiB b64
              </span>
            ) : null}
          </div>
        </div>
      </section>

      <section className="panel">
        <header className="panel-header">
          <h2>prompt</h2>
          <span className="muted small">the &lt;image&gt; token marks the 576 image placeholders</span>
        </header>
        <div className="panel-body">
          <textarea
            className="prompt-input mono"
            rows={4}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            spellCheck={false}
          />
          <div className="row wrap">
            <button type="button" className="btn ghost small" onClick={() => setPrompt(DEFAULT_PROMPT)}>
              reset template
            </button>
            <span className="muted small">prompt is used verbatim; the backend inserts the image block</span>
          </div>
        </div>
      </section>

      <section className="panel">
        <header className="panel-header">
          <h2>session</h2>
          {session ? <span className="badge ok">created</span> : null}
        </header>
        <div className="panel-body">
          <div className="row wrap">
            <button type="button" className="btn primary" onClick={handleCreate} disabled={!chosenB64 || creating}>
              {session ? 'recreate session' : 'create session'}
            </button>
            {previewSrc ? <img className="mini-preview" src={previewSrc} alt="selected" /> : <span className="muted small">choose an image first</span>}
          </div>
          {createErr ? <div className="banner error small">{createErr}</div> : null}
          {session ? (
            <dl className="kv">
              <div>
                <dt>session</dt>
                <dd className="mono">{session.session_id}</dd>
              </div>
              <div>
                <dt>mode</dt>
                <dd className="mono">
                  {session.mode} · {session.model.name}
                </dd>
              </div>
              <div>
                <dt>prompt tokens</dt>
                <dd className="mono">{session.text_tokens.length}</dd>
              </div>
              <div>
                <dt>variant</dt>
                <dd className="mono">
                  {session.variant}
                  {session.image ? '' : ' · no image block'}
                </dd>
              </div>
              <div>
                <dt>image span</dt>
                <dd className="mono">
                  {session.image
                    ? `[${session.image.start}, ${session.image.end}) · ${session.image.count} patches`
                    : 'none (text-only twin)'}
                </dd>
              </div>
              <div>
                <dt>baseline</dt>
                <dd>{session.has_baseline ? 'available' : 'none'}</dd>
              </div>
              {sessionInfo?.baseline_caption ? (
                <div>
                  <dt>baseline caption</dt>
                  <dd className="quote">{sessionInfo.baseline_caption}</dd>
                </div>
              ) : null}
              <div>
                <dt>created</dt>
                <dd className="mono">{sessionInfo?.created_utc ?? '—'}</dd>
              </div>
            </dl>
          ) : (
            <div className="muted small">no session yet</div>
          )}
          {session && session.text_tokens.length > 0 ? (
            <details>
              <summary className="small muted">prompt tokens</summary>
              <TokenChips tokens={session.text_tokens} showIndices />
            </details>
          ) : null}
        </div>
      </section>

      <section className="panel">
        <header className="panel-header">
          <h2>generate</h2>
          {generated ? <span className="badge ok">caption ready</span> : null}
        </header>
        <div className="panel-body">
          <div className="row wrap">
            <label className="field inline">
              <span>max_new_tokens</span>
              <input
                type="number"
                className="mono"
                min={1}
                max={maxTokens}
                value={maxNew}
                onChange={(e) => setMaxNew(Math.max(1, Math.min(maxTokens, Number(e.target.value) || 1)))}
              />
            </label>
            <button type="button" className="btn primary" onClick={handleGenerate} disabled={!session}>
              generate caption
            </button>
            {session ? <span className="muted small">greedy decode · backend max {maxTokens}</span> : null}
          </div>
          <JobProgress job={genJob} label="generate" error={genErr} />
          {generated ? (
            <div className="generated-block">
              <div className="quote">{generated.caption}</div>
              <TokenChips tokens={generated.tokens} showIndices />
              <div className="muted small">
                {generated.tokens.length} tokens · prompt_len {generated.prompt_len} · click a token in the Caption tab to
                read out its lens distribution
              </div>
            </div>
          ) : null}
          {imageB64 && !generated ? <div className="muted small">run generation to get a caption for the readout views</div> : null}
        </div>
      </section>
    </div>
  );
}
