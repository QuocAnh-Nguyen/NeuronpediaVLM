import { useCallback, useEffect, useMemo, useState } from 'react';
import { errorMessage, getMeta } from './lib/api';
import type { Flag, GenToken, GenerateResult, Meta, SessionInfo, SessionResponse, TabKey } from './lib/types';
import { MetaBar } from './components/MetaBar';
import type { ChipToken } from './components/TokenChips';
import { SessionView } from './views/SessionView';
import { CaptionView } from './views/CaptionView';
import { PatchView } from './views/PatchView';
import { SteerView } from './views/SteerView';

const TABS: Array<{ key: TabKey; n: number; label: string }> = [
  { key: 'session', n: 1, label: 'Session' },
  { key: 'caption', n: 2, label: 'Caption & lens' },
  { key: 'patch', n: 3, label: 'Patch attribution' },
  { key: 'steer', n: 4, label: 'Knockout & steering' },
];

const META_REFRESH_MS = 15000;

/**
 * Demo shell: fetches /api/meta (with retry), owns the cross-view session
 * state (session, image, caption, flags, painted patches) and switches tabs.
 */
export default function App() {
  const [tab, setTab] = useState<TabKey>('session');
  const [meta, setMeta] = useState<Meta | null>(null);
  const [metaErr, setMetaErr] = useState<string | null>(null);
  const [metaLoading, setMetaLoading] = useState(true);
  const [session, setSession] = useState<SessionResponse | null>(null);
  const [sessionInfo, setSessionInfo] = useState<SessionInfo | null>(null);
  const [imageB64, setImageB64] = useState<string | null>(null);
  const [generated, setGenerated] = useState<GenerateResult | null>(null);
  const [selectedTokenIndex, setSelectedTokenIndex] = useState<number | null>(null);
  const [flags, setFlags] = useState<Flag[]>([]);
  const [selectedPatches, setSelectedPatches] = useState<Set<number>>(() => new Set());

  const loadMeta = useCallback(async () => {
    setMetaLoading(true);
    try {
      setMeta(await getMeta());
      setMetaErr(null);
    } catch (e) {
      setMetaErr(errorMessage(e));
    } finally {
      setMetaLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadMeta();
  }, [loadMeta]);

  // Quiet background refresh of the job / GPU counters.
  useEffect(() => {
    const id = window.setInterval(() => {
      getMeta()
        .then(setMeta)
        .catch(() => undefined);
    }, META_REFRESH_MS);
    return () => window.clearInterval(id);
  }, []);

  const selectedToken = useMemo<GenToken | null>(
    () => generated?.tokens.find((t) => t.i === selectedTokenIndex) ?? null,
    [generated, selectedTokenIndex],
  );
  const defaultToken = flags[0]?.str ?? selectedToken?.str ?? '';

  const handleSession = useCallback((next: SessionResponse, info: SessionInfo | null, b64: string) => {
    setSession(next);
    setSessionInfo(info);
    setImageB64(b64);
    setGenerated(null);
    setSelectedTokenIndex(null);
    setSelectedPatches(new Set());
  }, []);

  const handleGenerated = useCallback((result: GenerateResult) => {
    setGenerated(result);
    setSelectedTokenIndex(null);
    setTab('caption');
  }, []);

  const handleSelectToken = useCallback((token: ChipToken) => {
    setSelectedTokenIndex(token.i);
    setTab('caption');
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 20 20" width={22} height={22}>
              <circle cx="10" cy="10" r="8.25" fill="none" stroke="currentColor" strokeWidth="1.1" opacity="0.35" />
              <circle cx="10" cy="10" r="4.6" fill="none" stroke="currentColor" strokeWidth="1.1" opacity="0.7" />
              <circle cx="10" cy="10" r="1.5" style={{ fill: 'var(--primary)' }} />
            </svg>
          </span>
          <div className="brand-text">
            <h1>VLM J-Lens</h1>
            <span className="brand-sub">
              Jacobian-lens readouts, patch attribution, causal knockout &amp; steering — captioning-hallucination research
            </span>
          </div>
        </div>
        <nav className="tabs">
          {TABS.map((t) => (
            <button
              key={t.key}
              type="button"
              className={`tab${tab === t.key ? ' active' : ''}`}
              onClick={() => setTab(t.key)}
            >
              <span className="tab-num">{t.n}</span>
              {t.label}
              {t.key === 'caption' && flags.length > 0 ? <span className="tab-badge">{flags.length}</span> : null}
              {t.key === 'steer' && selectedPatches.size > 0 ? <span className="tab-badge">{selectedPatches.size}</span> : null}
            </button>
          ))}
        </nav>
      </header>

      <MetaBar meta={meta} error={metaErr} loading={metaLoading} onRetry={loadMeta} />

      <main className="app-main">
        {tab === 'session' ? (
          <SessionView
            meta={meta}
            session={session}
            sessionInfo={sessionInfo}
            imageB64={imageB64}
            generated={generated}
            onSession={handleSession}
            onGenerated={handleGenerated}
          />
        ) : null}
        {tab === 'caption' ? (
          <CaptionView
            meta={meta}
            session={session}
            generated={generated}
            selectedTokenIndex={selectedTokenIndex}
            onSelectToken={handleSelectToken}
            flags={flags}
            onFlagsChange={setFlags}
          />
        ) : null}
        {tab === 'patch' ? (
          <PatchView
            meta={meta}
            session={session}
            imageB64={imageB64}
            generated={generated}
            defaultToken={defaultToken}
            selectedPatches={selectedPatches}
            onSelectedPatchesChange={setSelectedPatches}
          />
        ) : null}
        {tab === 'steer' ? (
          <SteerView
            meta={meta}
            session={session}
            generated={generated}
            flags={flags}
            defaultToken={defaultToken}
            selectedPatches={selectedPatches}
            onSelectedPatchesChange={setSelectedPatches}
          />
        ) : null}
      </main>

      <footer className="app-footer">
        {session ? (
          <>
            <span className="foot-item">
              <span className="foot-k">session</span>
              <span className="foot-v">{session.session_id}</span>
            </span>
            <span className="foot-item">
              <span className="foot-k">caption</span>
              <span className="foot-v">{generated ? `${generated.tokens.length} tokens` : 'none'}</span>
            </span>
            <span className="foot-item">
              <span className="foot-k">flagged</span>
              <span className="foot-v">{flags.length}</span>
            </span>
            <span className="foot-item">
              <span className="foot-k">patches</span>
              <span className="foot-v">{selectedPatches.size} selected</span>
            </span>
          </>
        ) : (
          <span className="foot-item">
            <span className="foot-v">no session yet</span>
          </span>
        )}
      </footer>
    </div>
  );
}
