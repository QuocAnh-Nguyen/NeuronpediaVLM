import { useMemo } from 'react';

/**
 * Word-level diff of two captions (LCS), used for steer / knockout
 * before-after comparisons. Removed words are highlighted on the `before`
 * row, inserted words on the `after` row.
 */

export interface DiffOp {
  type: 'same' | 'del' | 'ins';
  word: string;
}

export function diffWords(before: string, after: string): DiffOp[] {
  const a = before.split(/\s+/).filter(Boolean);
  const b = after.split(/\s+/).filter(Boolean);
  const n = a.length;
  const m = b.length;
  // dp[i][j] = LCS length of a[i:], b[j:]
  const dp: number[][] = Array.from({ length: n + 1 }, () => new Array<number>(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i -= 1) {
    for (let j = m - 1; j >= 0; j -= 1) {
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  const ops: DiffOp[] = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) {
      ops.push({ type: 'same', word: a[i] });
      i += 1;
      j += 1;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      ops.push({ type: 'del', word: a[i] });
      i += 1;
    } else {
      ops.push({ type: 'ins', word: b[j] });
      j += 1;
    }
  }
  while (i < n) {
    ops.push({ type: 'del', word: a[i] });
    i += 1;
  }
  while (j < m) {
    ops.push({ type: 'ins', word: b[j] });
    j += 1;
  }
  return ops;
}

export interface CaptionDiffProps {
  before: string;
  after: string;
  labels?: [string, string];
}

function Row({ label, ops, side }: { label: string; ops: DiffOp[]; side: 'before' | 'after' }) {
  const visible = ops.filter((op) => (side === 'before' ? op.type !== 'ins' : op.type !== 'del'));
  return (
    <div className={`diff-row diff-${side}`}>
      <span className="diff-label">{label}</span>
      <p className="diff-text">
        {visible.map((op, idx) => (
          <span key={`${idx}-${op.word}`} className={`diff-word ${op.type}`}>
            {op.word}{' '}
          </span>
        ))}
      </p>
    </div>
  );
}

export function CaptionDiff({ before, after, labels }: CaptionDiffProps) {
  const ops = useMemo(() => diffWords(before, after), [before, after]);
  const changed = ops.filter((op) => op.type !== 'same').length;
  return (
    <div className="diff">
      <div className="diff-meta muted small">
        {changed === 0 ? 'identical' : `${changed} divergent word${changed === 1 ? '' : 's'}`}
      </div>
      <Row label={labels?.[0] ?? 'before'} ops={ops} side="before" />
      <Row label={labels?.[1] ?? 'after'} ops={ops} side="after" />
    </div>
  );
}
