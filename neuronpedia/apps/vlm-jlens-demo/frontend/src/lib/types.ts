/**
 * Types mirroring the frozen HTTP contract of the vlm-jlens-demo backend.
 *
 * Every field name below is copied verbatim from the contract (see README,
 * "HTTP contract"). Do not rename fields here without changing the backend;
 * the UI reads these shapes directly. All ranks in the contract are 1-based.
 */

export type Mode = 'tiny' | 'hf-llava';

/** [columns, rows] of the image-token grid. The contract fixes [24, 24]. */
export type GridShape = [number, number];

/** Whether a session carries the image block (`image`) or is a text-only twin (`no_image`). */
export type SessionVariant = 'image' | 'no_image';

export type TargetKind = 'gen' | 'prompt' | 'patch';
export type AttributionMetric = 'lens_prob' | 'lens_logit' | 'attn_rollout';
export type KnockoutMode = 'zero' | 'mean';
export type SteerMode = 'add' | 'ablate' | 'swap';
export type SteerPositions = 'last' | 'all';
export type JobKind = 'generate' | 'lens' | 'attribution' | 'knockout' | 'steer';
export type JobStatus = 'queued' | 'running' | 'done' | 'error';

export interface MetaModel {
  name: string;
  n_layers: number;
  d_model: number;
  image_seq_length: number;
  grid: GridShape;
  vocab_size: number;
}

export interface LensInfo {
  available: boolean;
  dir: string;
  source_layers: number[];
  masks: string[];
  n_prompts: Record<string, number>;
  mock: boolean;
  notes: string | null;
  error: string | null;
}

export interface Capabilities {
  lens_readout: true;
  attribution_metrics: AttributionMetric[];
  knockout: true;
  steer_modes: SteerMode[];
  max_new_tokens: number;
}

export interface JobsInfo {
  active: number;
  queued: number;
}

export interface GpuInfo {
  free_mb: number;
  total_mb: number;
}

export interface Meta {
  mode: Mode;
  device: string;
  dtype: string;
  model: MetaModel;
  lens: LensInfo;
  capabilities: Capabilities;
  jobs: JobsInfo;
  gpu: GpuInfo | null;
}

export interface Sample {
  id: string;
  label: string;
  image_b64: string;
}

export interface SamplesResponse {
  samples: Sample[];
}

/** One text token of the prompt (prompt position `i`, tokenizer id, string). */
export interface TextToken {
  i: number;
  id: number;
  str: string;
}

export interface ImageSpan {
  start: number;
  end: number;
  count: number;
  grid: GridShape;
  /** 576 entries in 0..3: which 2x2 image quarter each patch belongs to. */
  quarter_of_patch: number[];
}

export interface SessionRequest {
  image_b64?: string;
  image_path?: string;
  prompt?: string;
  /** `no_image` creates a text-only twin (no image block); defaults to `image`. */
  variant?: SessionVariant;
}

export interface SessionResponse {
  session_id: string;
  mode: Mode;
  model: MetaModel;
  lens: LensInfo;
  prompt_text: string;
  text_tokens: TextToken[];
  /** The image block; null for `variant: 'no_image'` sessions. */
  image: ImageSpan | null;
  variant: SessionVariant;
  has_baseline: boolean;
}

/** Response of GET /api/session/{id}. */
export interface SessionInfo {
  session_id: string;
  has_baseline: boolean;
  baseline_caption: string | null;
  lens: LensInfo;
  created_utc: string;
}

export interface GenToken {
  i: number;
  id: number;
  str: string;
  logprob: number;
}

export interface GenerateResult {
  caption: string;
  tokens: GenToken[];
  prompt_len: number;
}

export interface GenerateRequest {
  session_id: string;
  max_new_tokens: number;
}

export interface LensTargetSpec {
  kind: TargetKind;
  i: number;
}

export interface LensRequest {
  session_id: string;
  layers?: number[];
  targets: LensTargetSpec[];
  topk: number;
  track?: string[];
}

export interface LensTopKEntry {
  str: string;
  prob: number;
  logit: number;
  rank: number;
}

export interface LensTrackedEntry {
  prob: number;
  rank: number;
  logit: number;
}

export interface LensLayerReadout {
  layer: number;
  topk: LensTopKEntry[];
  /** Keyed by the tracked token string sent in the request. */
  tracked: Record<string, LensTrackedEntry>;
}

/** The model's own next-token distribution at a target position (not lens evidence). */
export interface LensModelRow {
  topk: LensTopKEntry[];
  tracked: Record<string, LensTrackedEntry>;
}

export interface LensTargetResult {
  kind: TargetKind;
  i: number;
  label: string;
  per_layer: LensLayerReadout[];
  /** The model's own logits for this position; the final lens layer matches it. */
  model_row: LensModelRow;
}

export interface LensResult {
  targets: LensTargetResult[];
  vocab_size: number;
}

export interface AttributionRequest {
  session_id: string;
  layer: number;
  token: string;
  metric: AttributionMetric;
  rollout: boolean;
}

export interface AttributionResult {
  layer: number;
  metric: AttributionMetric;
  rollout: boolean;
  /** Row-major 24x24 patch scores (grid[row][col], patch = row*24 + col). */
  grid: number[][];
  /** Row-major 24x24 1-based patch ranks (rank 1 = highest score); null for attn_rollout. */
  grid_rank: number[][] | null;
  /** Mean score of each of the four 2x2 image quarters (Q0..Q3). */
  quarters: number[];
  vmin: number;
  vmax: number;
}

export interface KnockoutRequest {
  session_id: string;
  patches: number[];
  mode: KnockoutMode;
  target_tokens?: number[];
  max_new_tokens?: number;
}

export interface KnockoutDelta {
  token: string;
  logprob_before: number;
  logprob_after: number;
  delta: number;
}

/** How a knockout changed one candidate token. */
export type KnockoutOutcomeClass = 'removed' | 'persisted' | 'changed';

/** One classified candidate token after the knockout (`KnockoutResult.outcomes`). */
export interface KnockoutOutcome {
  token: string;
  class: KnockoutOutcomeClass;
  delta: number;
  /** Whether the token still appears in `caption_after`. */
  in_caption_after: boolean;
}

export interface KnockoutResult {
  patches: number[];
  mode: KnockoutMode;
  baseline_caption: string;
  caption_after: string;
  tokens_after: GenToken[];
  deltas: KnockoutDelta[];
  /** Classified candidate tokens (removed / persisted / changed). */
  outcomes: KnockoutOutcome[];
  /** Human-readable definition of the outcome classes. */
  classification_note: string;
}

/** Shape of one applied steer edit as echoed by the backend. */
export interface SteerEdit {
  layer: number;
  mode: SteerMode;
  token: string | null;
  source_token: string | null;
  alpha: number;
  /** Echo of the request's `positions` (the backend may resolve it to ints). */
  positions: number[] | SteerPositions;
}

export interface SteerRequest {
  session_id: string;
  layer: number;
  mode: SteerMode;
  token?: string;
  source_token?: string;
  alpha: number;
  positions: SteerPositions;
  max_new_tokens?: number;
}

/** Per-edit steering diagnostics echoed by the backend. */
export interface SteerDiagnostic {
  layer: number;
  mode: SteerMode;
  /** Norm of the applied lens vector. */
  v_norm: number;
  /** Norm of the residual activation at the edit site; null when not measured. */
  h_norm: number | null;
  /** Condition number of the edit basis; null for single-direction edits. */
  cond: number | null;
}

export interface SteerResult {
  edits: SteerEdit[];
  caption_before: string;
  caption_after: string;
  tokens_before: GenToken[];
  tokens_after: GenToken[];
  n_edit_forwards: number;
  /** One entry per applied edit (norms and conditioning). */
  diagnostics: SteerDiagnostic[];
}

export interface JobRef {
  job_id: string;
}

export interface Job<T = unknown> {
  job_id: string;
  kind: JobKind;
  status: JobStatus;
  stage: string;
  progress: number;
  t_submit: number;
  t_start: number | null;
  t_end: number | null;
  error: string | null;
  result: T | null;
}

/* ------------------------------------------------------------------ */
/* UI-only types (not part of the HTTP contract).                      */
/* ------------------------------------------------------------------ */

/** A researcher-flagged caption token, with a free-form note. */
export interface Flag {
  /** Raw token string as returned by the model (may carry a SentencePiece `▁`). */
  str: string;
  /** Global position `i` in the generated sequence, when known. */
  tokenIndex: number | null;
  note: string;
}

export type TabKey = 'session' | 'caption' | 'patch' | 'steer';

/** Hover state for the patch heatmap tooltip. */
export interface PatchHover {
  idx: number;
  x: number;
  y: number;
}
