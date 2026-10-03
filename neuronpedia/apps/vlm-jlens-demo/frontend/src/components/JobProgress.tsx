import type { Job } from '../lib/types';

export interface JobProgressProps {
  /** Latest job snapshot from `pollJob` (null when no job has started). */
  job: Job<unknown> | null;
  /** Optional human label, e.g. "attribution" or "generate". */
  label?: string;
  /** Error message from a failed request/job; shown next to the progress row. */
  error?: string | null;
}

/**
 * One-line job status: status pill, backend stage, progress bar, elapsed time.
 * Rendered under every action that starts a backend job.
 */
export function JobProgress({ job, label, error }: JobProgressProps) {
  if (!job && !error) return null;
  const status = job ? job.status : 'error';
  const pct = job ? Math.round(Math.max(0, Math.min(1, job.progress)) * 100) : 0;
  const started = job?.t_start ?? null;
  const ended = job?.t_end ?? null;
  const elapsed = started === null ? null : Math.max(0, (ended ?? Date.now() / 1000) - started);
  return (
    <div className="job">
      {label ? <span className="job-label">{label}</span> : null}
      <span className={`status-pill status-${status}`}>{status}</span>
      <span className="job-stage" title="backend stage">
        {job?.stage ?? ''}
      </span>
      <div className="progress" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <div className="progress-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="mono job-pct">{pct}%</span>
      {elapsed !== null ? <span className="mono muted">{elapsed.toFixed(1)}s</span> : null}
      {error ? <span className="job-error">{error}</span> : null}
    </div>
  );
}
