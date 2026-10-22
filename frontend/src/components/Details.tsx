import type { ReactNode } from 'react';
import { ChevronIcon } from './Icons';

/** Expandable panel for optional technical detail. */
export function Details({ summary, children, defaultOpen = false }: { summary: ReactNode; children: ReactNode; defaultOpen?: boolean }) {
  return (
    <details className="details" open={defaultOpen}>
      <summary>
        <span>{summary}</span>
        <ChevronIcon className="chevron" />
      </summary>
      <div className="details-body">{children}</div>
    </details>
  );
}
