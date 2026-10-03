/**
 * Typed API client for the frozen HTTP contract.
 *
 * Base URL is `''` (same origin): the Vite dev server proxies `/api` to
 * http://127.0.0.1:8787, and in production the backend serves the built
 * `dist/` from the same origin.
 *
 * Fixtures mode (`?fixtures=1` or `VITE_FIXTURES=1`) swaps every call for the
 * deterministic canned backend in `./fixtures` — no network involved.
 */
import {
  fixtureMeta,
  fixtureSamples,
  fixtureSession,
  fixtureSessionInfo,
  pollFixtureJob,
  startFixtureJob,
} from './fixtures';
import type {
  AttributionRequest,
  GenerateRequest,
  Job,
  JobKind,
  JobRef,
  KnockoutRequest,
  LensRequest,
  Meta,
  SamplesResponse,
  SessionInfo,
  SessionRequest,
  SessionResponse,
  SteerRequest,
} from './types';

const API_BASE = '';

function detectFixturesMode(): boolean {
  // Vite always defines import.meta.env, but guard so the client also loads in
  // plain runtimes (tests/scripts) where it is absent.
  const env = (import.meta.env ?? {}) as { VITE_FIXTURES?: string };
  const envFlag = String(env.VITE_FIXTURES ?? '').toLowerCase();
  if (envFlag === '1' || envFlag === 'true' || envFlag === 'yes' || envFlag === 'on') return true;
  if (typeof window === 'undefined') return false;
  const urlFlag = new URLSearchParams(window.location.search).get('fixtures');
  if (urlFlag === null) return false;
  const v = urlFlag.toLowerCase();
  return v !== '0' && v !== 'false';
}

/** True when the app serves deterministic offline fixtures. */
export const FIXTURES_MODE: boolean = detectFixturesMode();

export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(status > 0 ? `HTTP ${status}: ${detail}` : detail);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

/** Human-readable message for any thrown value (used by `catch` blocks). */
export function errorMessage(e: unknown): string {
  if (e instanceof ApiError) return e.detail;
  if (e instanceof Error) return e.message;
  return String(e);
}

function normalizeDetail(status: number, body: unknown): string {
  if (typeof body === 'string' && body.trim()) return body.trim();
  if (body && typeof body === 'object') {
    const detail = (body as { detail?: unknown }).detail;
    if (typeof detail === 'string' && detail.trim()) return detail.trim();
    if (detail !== undefined) return JSON.stringify(detail);
    return JSON.stringify(body);
  }
  return `HTTP ${status}`;
}

function delay(ms: number): Promise<void> {
  const { promise, resolve } = Promise.withResolvers<void>();
  setTimeout(resolve, ms);
  return promise;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...init,
    });
  } catch (e) {
    throw new ApiError(0, `backend unreachable (${e instanceof Error ? e.message : String(e)})`);
  }
  const text = await res.text();
  let body: unknown = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = text;
    }
  }
  if (!res.ok) throw new ApiError(res.status, normalizeDetail(res.status, body));
  return body as T;
}

function post<T>(path: string, payload: unknown): Promise<T> {
  return request<T>(path, { method: 'POST', body: JSON.stringify(payload) });
}

async function startJob(path: string, payload: unknown, kind: JobKind): Promise<JobRef> {
  if (FIXTURES_MODE) {
    await delay(60);
    return startFixtureJob(kind, payload);
  }
  return post<JobRef>(path, payload);
}

/** The UI keeps raw base64 everywhere; the live API ships data URLs. */
function stripDataUrlPrefix(value: string): string {
  const comma = value.indexOf(',');
  return value.startsWith('data:') && comma !== -1 ? value.slice(comma + 1) : value;
}

/* ------------------------------- endpoints ------------------------------- */

export async function getMeta(): Promise<Meta> {
  if (FIXTURES_MODE) {
    await delay(80);
    return fixtureMeta();
  }
  return request<Meta>('/api/meta');
}

export async function getSamples(): Promise<SamplesResponse> {
  if (FIXTURES_MODE) {
    await delay(80);
    return fixtureSamples();
  }
  const res = await request<SamplesResponse>('/api/samples');
  return {
    ...res,
    samples: res.samples.map((s) => ({ ...s, image_b64: stripDataUrlPrefix(s.image_b64) })),
  };
}

export async function createSession(req: SessionRequest): Promise<SessionResponse> {
  if (FIXTURES_MODE) {
    await delay(120);
    return fixtureSession(req);
  }
  return post<SessionResponse>('/api/session', req);
}

export async function getSession(sessionId: string): Promise<SessionInfo> {
  if (FIXTURES_MODE) {
    await delay(60);
    return fixtureSessionInfo(sessionId);
  }
  return request<SessionInfo>(`/api/session/${encodeURIComponent(sessionId)}`);
}

export function generate(req: GenerateRequest): Promise<JobRef> {
  return startJob('/api/generate', req, 'generate');
}

export function lens(req: LensRequest): Promise<JobRef> {
  return startJob('/api/lens', req, 'lens');
}

export function attribution(req: AttributionRequest): Promise<JobRef> {
  return startJob('/api/attribution', req, 'attribution');
}

export function knockout(req: KnockoutRequest): Promise<JobRef> {
  return startJob('/api/knockout', req, 'knockout');
}

export function steer(req: SteerRequest): Promise<JobRef> {
  return startJob('/api/steer', req, 'steer');
}

export interface PollOptions {
  /** Poll interval in ms (contract: ~700). */
  intervalMs?: number;
  /** Stop polling and throw when aborted. */
  signal?: AbortSignal;
}

/**
 * Poll `GET /api/jobs/{id}` until the job is terminal.
 *
 * `onProgress` fires on every snapshot (including the terminal one). Resolves
 * with the `done` job; throws `ApiError` when the job reports `error` or when
 * polling is aborted.
 */
export async function pollJob<T>(
  jobId: string,
  onProgress?: (job: Job<T>) => void,
  opts: PollOptions = {},
): Promise<Job<T>> {
  if (FIXTURES_MODE) return pollFixtureJob<T>(jobId, onProgress);
  const interval = opts.intervalMs ?? 700;
  for (;;) {
    const job = await request<Job<T>>(`/api/jobs/${encodeURIComponent(jobId)}`);
    onProgress?.(job);
    if (job.status === 'done') return job;
    if (job.status === 'error') throw new ApiError(500, job.error ?? 'job failed');
    if (opts.signal?.aborted) throw new ApiError(0, 'polling cancelled');
    await delay(interval);
  }
}

