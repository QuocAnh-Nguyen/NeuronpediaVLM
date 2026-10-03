/**
 * Offline fixture backend.
 *
 * Enabled with `?fixtures=1` in the URL or `VITE_FIXTURES=1` at dev/build time
 * (see `lib/api.ts`). Every function here returns a deterministic stand-in for
 * the corresponding frozen-contract payload. Values come from
 * `../fixtures/*.json` and are adapted to the actual request — target index,
 * tracked tokens, layer, metric, selected patches, steer parameters — so every
 * control in the UI has a visible, reproducible effect without a backend.
 *
 * Sessions are registered at creation time, so a `variant: 'no_image'` twin
 * (CaptionView) deterministically gets a prior-driven caption and lens rows.
 *
 * Deterministic pseudo-randomness is a string hash (FNV-1a); no `Math.random`
 * and no wall-clock values are used for payload content.
 */
import attributionJson from '../fixtures/attribution.json';
import generateJson from '../fixtures/generate.json';
import knockoutJson from '../fixtures/knockout.json';
import lensJson from '../fixtures/lens.json';
import metaJson from '../fixtures/meta.json';
import samplesJson from '../fixtures/samples.json';
import sessionJson from '../fixtures/session.json';
import sessionInfoJson from '../fixtures/session_info.json';
import sessionNoImageJson from '../fixtures/session_no_image.json';
import steerJson from '../fixtures/steer.json';
import type {
  AttributionRequest,
  AttributionResult,
  GenerateRequest,
  GenerateResult,
  GenToken,
  Job,
  JobKind,
  JobRef,
  KnockoutOutcome,
  KnockoutOutcomeClass,
  KnockoutRequest,
  KnockoutResult,
  LensLayerReadout,
  LensRequest,
  LensResult,
  LensTopKEntry,
  LensTrackedEntry,
  Meta,
  Sample,
  SamplesResponse,
  SessionInfo,
  SessionRequest,
  SessionResponse,
  SessionVariant,
  SteerDiagnostic,
  SteerRequest,
  SteerResult,
  TargetKind,
} from './types';

/* ----------------------------- payload templates ------------------------- */

const META = metaJson as unknown as Meta;
const SESSION = sessionJson as unknown as SessionResponse;
const SESSION_NO_IMAGE = sessionNoImageJson as unknown as SessionResponse;
const SESSION_INFO = sessionInfoJson as unknown as SessionInfo;
const GENERATE = generateJson as unknown as GenerateResult;
const LENS_TARGET = (lensJson as unknown as LensResult).targets[0];
const LENS_TEMPLATE = LENS_TARGET.per_layer;
const LENS_MODEL_ROW = LENS_TARGET.model_row;
const BASE_GRID = attributionJson.grid as number[][];
const KNOCKOUT = knockoutJson as unknown as KnockoutResult;
const STEER = steerJson as unknown as SteerResult;
/** Caption a no-image twin produces from the language prior alone. */
const NO_IMAGE_CAPTION = 'A cat is sitting on a chair in a room.';

interface FixtureSessionSpec {
  variant: SessionVariant;
  prompt_len: number;
}

/** Sessions created via `fixtureSession`, so generate/lens can vary by variant. */
const fixtureSessions = new Map<string, FixtureSessionSpec>();

/* ------------------------- deterministic helpers ------------------------- */

/** FNV-1a, 32-bit. */
function hashString(s: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i += 1) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/** Deterministic [0, 1) from a hash. */
function unit(hash: number): number {
  return (hash % 10007) / 10007;
}

function round(v: number, digits: number): number {
  const f = 10 ** digits;
  return Math.round(v * f) / f;
}

/** Join SentencePiece-ish token strings back into display text. */
function captionFromTokens(tokens: Array<{ str: string }>): string {
  return tokens
    .map((t) => t.str)
    .join('')
    .replace(/▁/g, ' ')
    .replace(/<0x0A>/g, ' ')
    .replace(/\s+([.,!?;:])/g, '$1')
    .trim();
}

/** Strip SentencePiece markers so a token can be used as a plain word. */
function wordOf(token: string): string {
  return token.replace(/▁/g, ' ').replace(/<0x0A>/g, ' ').trim();
}

/** Build a plausible token list for a caption (deterministic ids/logprobs). */
function tokensForCaption(caption: string, startIndex: number): GenToken[] {
  return caption
    .replace(/\.$/, '')
    .split(/\s+/)
    .filter(Boolean)
    .map((w, k) => ({
      i: startIndex + k,
      id: 500 + ((k * 7919 + hashString(w) % 30000) % 30000),
      str: k === 0 ? w : `▁${w}`,
      logprob: round(-0.22 - 0.1 * k, 3),
    }));
}

/**
 * Logit line for a tracked token: rises smoothly with depth, so the tracked
 * curve reliably improves and its rank falls. Self-consistent: `rank` is
 * derived from `prob` against a fixed top-1 mass. A `no_image` twin scales the
 * curve by `priorFactor`, so the same token can gain or lose mass.
 */
function trackedCurve(token: string, layer: number, nLayers: number, variant: SessionVariant = 'image'): LensTrackedEntry {
  const t = nLayers > 1 ? layer / (nLayers - 1) : 1;
  const seed = unit(hashString(token));
  const topProb = 0.42 + 0.22 * seed;
  const start = -7 + 1.4 * seed;
  const end = Math.log(topProb / (1 - topProb));
  let prob = 1 / (1 + Math.exp(-(start + (end - start) * Math.pow(t, 1.35))));
  if (variant === 'no_image') prob *= priorFactor(token);
  prob = Math.min(0.9999, Math.max(1e-6, prob));
  const rank = Math.max(1, Math.min(META.model.vocab_size - 1, Math.round((topProb / prob) ** 1.6)));
  return { prob: round(prob, 4), rank, logit: round(Math.log(prob / (1 - prob)), 2) };
}

/**
 * Deterministic no-image scaling of a token's lens mass: prior-driven tokens
 * (u < 0.3) keep or gain mass without the image, image-driven ones lose it.
 */
function priorFactor(token: string): number {
  const u = unit(hashString(`prior:${token}`));
  return u < 0.3 ? 1.15 + 0.5 * u : 0.3 + 0.55 * u;
}

/* ----------------------------- payload builders -------------------------- */

export function fixtureMeta(): Meta {
  return META;
}

export function fixtureSamples(): SamplesResponse {
  return { samples: samplesJson.samples as Sample[] };
}

export function fixtureSession(req: SessionRequest): SessionResponse {
  const imageKey = req.image_b64 ?? req.image_path ?? 'default';
  const variant: SessionVariant = req.variant === 'no_image' ? 'no_image' : 'image';
  const template = variant === 'no_image' ? SESSION_NO_IMAGE : SESSION;
  const rawPrompt = req.prompt && req.prompt.trim() ? req.prompt : template.prompt_text;
  // No-image twins drop the `<image>` placeholder; their text tokens come from
  // the dedicated fixture, whose tokens are already re-indexed.
  const prompt = variant === 'no_image' ? rawPrompt.replace(/<image>/g, '').replace(/[ \t]{2,}/g, ' ').trim() : rawPrompt;
  const prompt_len = variant === 'no_image' ? template.text_tokens.length : (SESSION.image?.start ?? template.text_tokens.length);
  const session_id = `fixture-session-${variant === 'no_image' ? 'noimg' : 'img'}-${hashString(imageKey).toString(36)}`;
  fixtureSessions.set(session_id, { variant, prompt_len });
  return {
    ...template,
    session_id,
    prompt_text: prompt,
    image: variant === 'no_image' ? null : SESSION.image,
    variant,
  };
}

export function fixtureSessionInfo(id: string): SessionInfo {
  return { ...SESSION_INFO, session_id: id };
}

export function fixtureGenerate(req: GenerateRequest): GenerateResult {
  const spec = fixtureSessions.get(req.session_id);
  if (spec?.variant === 'no_image') {
    const all = tokensForCaption(NO_IMAGE_CAPTION, spec.prompt_len);
    const max = Math.max(1, Math.min(req.max_new_tokens || all.length, all.length));
    const tokens = all.slice(0, max);
    return { caption: captionFromTokens(tokens), tokens, prompt_len: spec.prompt_len };
  }
  const max = Math.max(1, Math.min(req.max_new_tokens || GENERATE.tokens.length, GENERATE.tokens.length));
  const tokens = GENERATE.tokens.slice(0, max);
  return { caption: captionFromTokens(tokens), tokens, prompt_len: GENERATE.prompt_len };
}

/** Fresh copy of a top-k row; `no_image` rescales it by prior mass and re-ranks. */
function variantTopkRow(base: LensTopKEntry[], variant: SessionVariant): LensTopKEntry[] {
  if (variant === 'image') return base.map((e) => ({ ...e }));
  const scaled = base
    .map((e) => {
      const prob = Math.min(0.9999, Math.max(1e-6, e.prob * priorFactor(e.str)));
      return { str: e.str, prob: round(prob, 4), logit: round(Math.log(prob / (1 - prob)), 2), rank: 0 };
    })
    .sort((a, b) => b.prob - a.prob);
  return scaled.map((e, idx) => ({ ...e, rank: idx + 1 }));
}

/** Scale one template row to a depth factor (lens rows sharpen toward the output). */
function layerRow(base: LensTopKEntry[], layerFactor: number): LensTopKEntry[] {
  return base.map((e) => {
    const prob = round(e.prob * layerFactor, 4);
    return { str: e.str, prob, logit: round(Math.log(prob) + 12.6, 2), rank: e.rank };
  });
}

/**
 * Merge tracked tokens into a top-k row (inserting them when they rank inside
 * the row) and return both the row and the tracked map with consistent ranks.
 */
function mergeTracked(
  base: LensTopKEntry[],
  track: string[],
  k: number,
  curveFor: (token: string) => LensTrackedEntry,
): { topk: LensTopKEntry[]; tracked: Record<string, LensTrackedEntry> } {
  const entries = base.filter((e) => !track.includes(e.str)).map((e) => ({ ...e }));
  const tracked: Record<string, LensTrackedEntry> = {};
  for (const token of track) {
    const curve = curveFor(token);
    const rankInList = entries.filter((e) => e.prob > curve.prob).length + 1;
    if (rankInList <= k) {
      entries.splice(rankInList - 1, 0, { str: token, prob: curve.prob, logit: curve.logit, rank: rankInList });
      tracked[token] = { prob: curve.prob, rank: rankInList, logit: curve.logit };
    } else {
      tracked[token] = curve;
    }
  }
  const topk = entries.slice(0, k).map((e, idx) => ({ ...e, rank: idx + 1 }));
  for (const token of Object.keys(tracked)) {
    const at = topk.findIndex((e) => e.str === token);
    if (at >= 0) tracked[token] = { ...tracked[token], rank: at + 1 };
  }
  return { topk, tracked };
}

export function fixtureLens(req: LensRequest): LensResult {
  const nLayers = META.model.n_layers;
  const k = Math.max(1, req.topk || 8);
  const track = (req.track ?? []).filter((t) => t.length > 0);
  const variant: SessionVariant = fixtureSessions.get(req.session_id)?.variant ?? 'image';
  const targets = req.targets.map((spec) => {
    const per_layer: LensLayerReadout[] = [];
    for (let L = 0; L < nLayers; L += 1) {
      const t = nLayers > 1 ? L / (nLayers - 1) : 1;
      const base = layerRow(variantTopkRow(LENS_TEMPLATE[Math.min(L, LENS_TEMPLATE.length - 1)].topk, variant), 0.3 + 0.95 * t);
      const { topk, tracked } = mergeTracked(base, track, k, (token) => trackedCurve(token, L, nLayers, variant));
      per_layer.push({ layer: L, topk, tracked });
    }
    const modelRow = mergeTracked(variantTopkRow(LENS_MODEL_ROW.topk, variant), track, k, (token) =>
      trackedCurve(token, nLayers - 1, nLayers, variant),
    );
    return {
      kind: spec.kind,
      i: spec.i,
      label: labelForTarget(spec.kind, spec.i),
      per_layer,
      model_row: { topk: modelRow.topk, tracked: modelRow.tracked },
    };
  });
  return { targets, vocab_size: META.model.vocab_size };
}

function labelForTarget(kind: TargetKind, i: number): string {
  if (kind === 'patch') return `patch[${i}] (${quarterName(i)})`;
  if (kind === 'prompt') {
    const tok = SESSION.text_tokens.find((t) => t.i === i);
    return tok ? `prompt[${i}] '${tok.str}'` : `prompt[${i}]`;
  }
  const tok = GENERATE.tokens.find((t) => t.i === i);
  return tok ? `gen[${i}] '${tok.str}'` : `gen[${i}]`;
}

export function quarterName(idx: number): string {
  const col = idx % 24;
  const row = Math.floor(idx / 24);
  const q = (row < 12 ? 0 : 2) + (col < 12 ? 0 : 1);
  return `row ${row}, col ${col}, Q${q}`;
}

export function fixtureAttribution(req: AttributionRequest): AttributionResult {
  const seed = unit(hashString(wordOf(req.token) || req.token || 'image'));
  const layerFactor = 0.62 + 0.75 * unit(hashString(`L${req.layer}`));
  const metricGamma = req.metric === 'lens_logit' ? 0.62 : req.metric === 'attn_rollout' ? 1.7 : 1;
  const phase = seed * 6 + req.layer * 0.7;
  const raw = BASE_GRID.map((row, r) =>
    row.map((v, c) => {
      const shaped = v ** metricGamma * layerFactor * (1 + 0.18 * Math.sin((r + c) * 0.3 + phase));
      return shaped;
    }),
  );
  const grid = req.rollout ? blur3(raw) : raw;
  const values = grid.flat();
  const vmin = round(Math.min(...values), 4);
  const vmax = round(Math.max(...values), 4);
  const sums = [0, 0, 0, 0];
  const counts = [0, 0, 0, 0];
  grid.forEach((row, r) =>
    row.forEach((v, c) => {
      const q = (r < 12 ? 0 : 2) + (c < 12 ? 0 : 1);
      sums[q] += v;
      counts[q] += 1;
    }),
  );
  return {
    layer: req.layer,
    metric: req.metric,
    rollout: req.rollout,
    grid: grid.map((row) => row.map((v) => round(v, 4))),
    grid_rank: req.metric === 'attn_rollout' ? null : rankGrid(grid),
    quarters: sums.map((s, q) => round(s / Math.max(1, counts[q]), 4)),
    vmin,
    vmax,
  };
}

/** 1-based rank of every patch in the grid (rank 1 = highest score). */
function rankGrid(grid: number[][]): number[][] {
  const rank = grid.map((row) => row.map(() => 0));
  grid
    .flatMap((row, r) => row.map((v, c) => ({ v, r, c })))
    .sort((a, b) => b.v - a.v || a.r - b.r || a.c - b.c)
    .forEach((cell, pos) => {
      rank[cell.r][cell.c] = pos + 1;
    });
  return rank;
}

function blur3(grid: number[][]): number[][] {
  const rows = grid.length;
  const cols = grid[0]?.length ?? 0;
  return grid.map((row, r) =>
    row.map((_, c) => {
      let sum = 0;
      let n = 0;
      for (let dr = -1; dr <= 1; dr += 1) {
        for (let dc = -1; dc <= 1; dc += 1) {
          const rr = r + dr;
          const cc = c + dc;
          if (rr >= 0 && rr < rows && cc >= 0 && cc < cols) {
            sum += grid[rr][cc];
            n += 1;
          }
        }
      }
      return sum / Math.max(1, n);
    }),
  );
}

export function fixtureKnockout(req: KnockoutRequest): KnockoutResult {
  const n = req.patches.length;
  if (n === 0) {
    return {
      patches: [],
      mode: req.mode,
      baseline_caption: GENERATE.caption,
      caption_after: GENERATE.caption,
      tokens_after: GENERATE.tokens.map((t) => ({ ...t })),
      deltas: KNOCKOUT.deltas.map((d) => ({
        token: d.token,
        logprob_before: d.logprob_before,
        logprob_after: d.logprob_before,
        delta: 0,
      })),
      outcomes: KNOCKOUT.outcomes.map((o): KnockoutOutcome => ({
        token: o.token,
        class: 'persisted',
        delta: 0,
        in_caption_after: true,
      })),
      classification_note: KNOCKOUT.classification_note,
    };
  }
  const strength = Math.min(2.2, 0.55 + n / 14);
  const captionAfter = req.mode === 'mean' ? 'A gray cat sitting on a bench in a garden.' : KNOCKOUT.caption_after;
  const tokensAfter = tokensForCaption(captionAfter, GENERATE.prompt_len);
  const captionWords = captionWordsOf(captionAfter);
  return {
    patches: [...req.patches],
    mode: req.mode,
    baseline_caption: GENERATE.caption,
    caption_after: captionAfter,
    tokens_after: tokensAfter,
    deltas: KNOCKOUT.deltas.map((d) => {
      const delta = round(d.delta * strength, 3);
      return { token: d.token, logprob_before: d.logprob_before, logprob_after: round(d.logprob_before + delta, 3), delta };
    }),
    outcomes: KNOCKOUT.outcomes.map((o): KnockoutOutcome => {
      const delta = round(o.delta * strength, 3);
      const in_caption_after = captionWords.has(wordOf(o.token).toLowerCase());
      return { token: o.token, class: classifyOutcome(delta, in_caption_after), delta, in_caption_after };
    }),
    classification_note: KNOCKOUT.classification_note,
  };
}

/** Lower-cased word set of a caption, for outcome classification. */
function captionWordsOf(caption: string): Set<string> {
  return new Set(
    caption
      .toLowerCase()
      .replace(/[.,!?;:]/g, ' ')
      .split(/\s+/)
      .filter(Boolean),
  );
}

/** removed = absent from the caption after; changed = still there but moved by |Δ| ≥ 0.25. */
function classifyOutcome(delta: number, inCaptionAfter: boolean): KnockoutOutcomeClass {
  if (!inCaptionAfter) return 'removed';
  return Math.abs(delta) >= 0.25 ? 'changed' : 'persisted';
}

export function fixtureSteer(req: SteerRequest): SteerResult {
  const before = STEER.caption_before;
  const tokenWord = wordOf(req.token ?? '');
  const sourceWord = wordOf(req.source_token ?? '');
  const baseWords = before.replace(/\.$/, '').split(/\s+/).filter(Boolean);
  let words = [...baseWords];
  if (req.alpha > 0 && tokenWord) {
    if (req.mode === 'add') {
      const reps = Math.max(1, Math.min(4, Math.round(1 + req.alpha / 2)));
      for (let r = 0; r < reps; r += 1) words.splice(2, 0, tokenWord);
    } else if (req.mode === 'ablate') {
      words = words.filter((w) => w.toLowerCase() !== tokenWord.toLowerCase());
    } else {
      let swapped = false;
      words = words.map((w) => {
        if (!swapped && w.toLowerCase() === sourceWord.toLowerCase()) {
          swapped = true;
          return tokenWord;
        }
        return w;
      });
      if (!swapped && words.length > 1) words[1] = tokenWord;
    }
  }
  const after = words.length > 0 ? `${words.join(' ')}.` : before;
  const tokensAfter = tokensForCaption(after, GENERATE.prompt_len);
  const template = STEER.diagnostics[0];
  const baseH = template?.h_norm ?? 12;
  const diagnostics: SteerDiagnostic[] = [
    {
      layer: req.layer,
      mode: req.mode,
      v_norm: round((template?.v_norm ?? 3.4) * (0.75 + 0.5 * unit(hashString(`v:${req.token ?? ''}:${req.mode}`))), 3),
      h_norm: req.positions === 'all' ? null : round(baseH * (0.85 + 0.4 * unit(hashString(`h:${req.layer}`))), 3),
      // Swaps deliberately land in the nearly-collinear regime, so the UI's
      // cond > 1e3 warning is exercised deterministically in fixtures mode.
      cond: req.mode === 'swap' ? round(1050 + 1500 * unit(hashString(`cond:${req.token ?? ''}:${req.source_token ?? ''}`)), 1) : null,
    },
  ];
  return {
    edits: [
      {
        layer: req.layer,
        mode: req.mode,
        token: req.token ?? null,
        source_token: req.source_token ?? null,
        alpha: req.alpha,
        positions: req.positions,
      },
    ],
    diagnostics,
    caption_before: before,
    caption_after: after,
    tokens_before: STEER.tokens_before.map((t) => ({ ...t })),
    tokens_after: tokensAfter,
    n_edit_forwards: req.positions === 'all' ? tokensAfter.length + 1 : tokensAfter.length,
  };
}

/* ------------------------------ job simulator ---------------------------- */

const JOB_STAGES: Record<JobKind, string[]> = {
  generate: ['queue', 'prefill', 'decode', 'serialize'],
  lens: ['load lens', 'readout', 'top-k', 'serialize'],
  attribution: ['forward pass', 'gradient hooks', 'reduce to grid', 'serialize'],
  knockout: ['baseline pass', 'knocked-out pass', 'compute deltas', 'serialize'],
  steer: ['resolve lens vectors', 'edit + prefill', 'decode', 'serialize'],
};

const jobs = new Map<string, { kind: JobKind; request: unknown }>();
let jobCounter = 0;

export function startFixtureJob(kind: JobKind, request: unknown): JobRef {
  jobCounter += 1;
  const job_id = `fx-${kind}-${jobCounter}`;
  jobs.set(job_id, { kind, request });
  return { job_id };
}

function fixtureJobResult(kind: JobKind, request: unknown): unknown {
  switch (kind) {
    case 'generate':
      return fixtureGenerate(request as GenerateRequest);
    case 'lens':
      return fixtureLens(request as LensRequest);
    case 'attribution':
      return fixtureAttribution(request as AttributionRequest);
    case 'knockout':
      return fixtureKnockout(request as KnockoutRequest);
    case 'steer':
      return fixtureSteer(request as SteerRequest);
    default:
      return null;
  }
}

function sleep(ms: number): Promise<void> {
  const { promise, resolve } = Promise.withResolvers<void>();
  setTimeout(resolve, ms);
  return promise;
}

/** Simulated job: 3 progress ticks (~280 ms each) then a `done` snapshot. */
export async function pollFixtureJob<T>(
  jobId: string,
  onProgress?: (job: Job<T>) => void,
): Promise<Job<T>> {
  const spec = jobs.get(jobId);
  if (!spec) throw new Error(`unknown fixture job: ${jobId}`);
  const stages = JOB_STAGES[spec.kind];
  const tSubmit = Date.now() / 1000;
  const steps: Array<{ status: 'running' | 'done'; stage: string; progress: number }> = [
    { status: 'running', stage: stages[0], progress: 0.08 },
    { status: 'running', stage: stages[1], progress: 0.45 },
    { status: 'running', stage: stages[2], progress: 0.82 },
    { status: 'done', stage: stages[3], progress: 1 },
  ];
  for (const step of steps) {
    await sleep(280);
    const job: Job<T> = {
      job_id: jobId,
      kind: spec.kind,
      status: step.status,
      stage: step.stage,
      progress: step.progress,
      t_submit: tSubmit,
      t_start: tSubmit,
      t_end: step.status === 'done' ? Date.now() / 1000 : null,
      error: null,
      result: step.status === 'done' ? (fixtureJobResult(spec.kind, spec.request) as T) : null,
    };
    onProgress?.(job);
    if (step.status === 'done') {
      jobs.delete(jobId);
      return job;
    }
  }
  throw new Error('unreachable');
}

