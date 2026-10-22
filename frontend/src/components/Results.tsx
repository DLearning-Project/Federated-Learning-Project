import type { ReactNode } from 'react';
import { Details } from './Details';

export interface Stat {
  label: ReactNode;
  value: ReactNode;
  note?: ReactNode;
}

export function StatGrid({ stats }: { stats: Stat[] }) {
  return (
    <div className="stat-grid">
      {stats.map((s, i) => (
        <div className="stat" key={i}>
          <span className="stat-label">{s.label}</span>
          <span className="stat-value">{s.value}</span>
          {s.note && <span className="stat-note">{s.note}</span>}
        </div>
      ))}
    </div>
  );
}

/** Assumptions returned by the backend, always visible next to results. */
export function AssumptionsList({ assumptions, title = 'Assumptions' }: { assumptions: string[]; title?: string }) {
  return (
    <Details summary={title} defaultOpen>
      <ul className="assumptions">
        {assumptions.map((a) => (
          <li key={a}>{a}</li>
        ))}
      </ul>
    </Details>
  );
}

/** Shows the mode badge the backend attaches to every numerical response. */
export function ModeBadge({ mode }: { mode: string }) {
  return <span className="badge">Mode: {mode}</span>;
}

/** A labelled horizontal meter: value vs. a limit, with text status (never color alone). */
export function Meter({
  label,
  segments,
  limit,
  max,
  valueText,
  statusText,
}: {
  label: string;
  segments: { value: number; color: string; label: string }[];
  limit?: number;
  max: number;
  valueText: string;
  statusText?: ReactNode;
}) {
  const scale = max > 0 ? max : 1;
  return (
    <div className="meter">
      <div className="meter-head">
        <strong>{label}</strong>
        <span>{valueText}</span>
      </div>
      <div className="meter-track" role="img" aria-label={`${label}: ${valueText}`}>
        {segments.map((s) => (
          <span
            key={s.label}
            className="meter-fill"
            title={s.label}
            style={{ width: `${Math.min(100, (s.value / scale) * 100)}%`, background: s.color }}
          />
        ))}
        {limit !== undefined && <span className="meter-limit" style={{ left: `calc(${Math.min(100, (limit / scale) * 100)}% - 1px)` }} />}
      </div>
      {statusText && <div className="small">{statusText}</div>}
    </div>
  );
}
