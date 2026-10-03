/**
 * Clickable token chips for generated caption tokens or prompt text tokens.
 * The SentencePiece `▁` marker is rendered as `␣` so word boundaries stay
 * visible; `<0x0A>` becomes `⏎`.
 */

export interface ChipToken {
  i: number;
  str: string;
  id?: number;
  logprob?: number;
}

export interface TokenChipsProps {
  tokens: ChipToken[];
  /** Selected token position `i` (only one chip is highlighted). */
  selected?: number | null;
  /** Raw token strings that are flagged (rendered with a marker). */
  flagged?: string[];
  onSelect?: (token: ChipToken) => void;
  /** Show the global token index inside the chip. */
  showIndices?: boolean;
  emptyText?: string;
}

/** Display form of a raw token string (`▁` → `␣`). */
export function displayToken(str: string): string {
  return str.replace(/▁/g, '␣').replace(/<0x0A>/g, '⏎');
}

/** Plain-text form of a raw token string (markers stripped). */
export function plainToken(str: string): string {
  return str.replace(/▁/g, ' ').replace(/<0x0A>/g, ' ').trim();
}

function logprobClass(logprob: number | undefined): string {
  if (logprob === undefined) return '';
  if (logprob > -0.5) return 'lp-good';
  if (logprob > -1.5) return 'lp-mid';
  return 'lp-bad';
}

export function TokenChips({ tokens, selected = null, flagged = [], onSelect, showIndices = false, emptyText }: TokenChipsProps) {
  if (tokens.length === 0) {
    return <div className="empty small">{emptyText ?? 'no tokens'}</div>;
  }
  return (
    <div className="chips">
      {tokens.map((t) => {
        const isFlagged = flagged.includes(t.str);
        const classes = [
          'chip',
          selected === t.i ? 'selected' : '',
          isFlagged ? 'flagged' : '',
          logprobClass(t.logprob),
        ]
          .filter(Boolean)
          .join(' ');
        const title =
          `i=${t.i}` +
          (t.id !== undefined ? ` id=${t.id}` : '') +
          (t.logprob !== undefined ? ` logprob=${t.logprob.toFixed(3)}` : '') +
          (isFlagged ? ' · flagged' : '');
        return (
          <button
            key={`${t.i}-${t.str}`}
            type="button"
            className={classes}
            title={title}
            onClick={onSelect ? () => onSelect(t) : undefined}
            disabled={!onSelect}
          >
            <span className="chip-str">{displayToken(t.str)}</span>
            {showIndices ? <span className="chip-idx mono">{t.i}</span> : null}
            {isFlagged ? <span className="chip-flag" aria-label="flagged">●</span> : null}
          </button>
        );
      })}
    </div>
  );
}
