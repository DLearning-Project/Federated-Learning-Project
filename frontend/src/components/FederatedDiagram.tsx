import { useId, useState, type ReactNode } from 'react';

export type DiagramPart = 'data' | 'train' | 'upload' | 'aggregate' | 'download';

interface DiagramLabels {
  server: string;
  client: string;
  data: string;
  serverDetail?: string;
}

const DEFAULT_LABELS: DiagramLabels = {
  server: 'Server',
  client: 'Client',
  data: 'Local data',
  serverDetail: 'Aggregates updates',
};

const CLIENT_X = [40, 240, 440];
const SERVER_X = 200;

/**
 * Client–server diagram: raw data stays on each client; only model updates go up
 * and the aggregated model comes back down. Parts named in `active` are highlighted.
 */
export function FederatedDiagram({ active = 'all', labels, title }: { active?: DiagramPart[] | 'all'; labels?: Partial<DiagramLabels>; title: string }) {
  const id = useId().replace(/:/g, '');
  const l = { ...DEFAULT_LABELS, ...labels };
  const on = (part: DiagramPart) => active === 'all' || active.includes(part);

  const ink = 'var(--color-ink)';
  const faded = 'var(--color-chart-light)';
  const accent = 'var(--color-orange)';

  return (
    <figure style={{ margin: 0, display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
      <svg className="chart" viewBox="0 0 640 330" role="img" aria-labelledby={`${id}-t`}>
        <title id={`${id}-t`}>{title}</title>
        <defs>
          {[
            ['on', accent],
            ['off', faded],
            ['ink', ink],
          ].map(([k, c]) => (
            <marker key={k} id={`${id}-arrow-${k}`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0 0 10 5 0 10z" fill={c} />
            </marker>
          ))}
        </defs>

        {/* Server */}
        <rect x={SERVER_X} y={16} width={240} height={78} rx={14} fill="var(--color-surface)" stroke={on('aggregate') ? ink : faded} strokeWidth={on('aggregate') ? 2 : 1.5} />
        {on('aggregate') && active !== 'all' && <rect x={SERVER_X + 20} y={90} width={200} height={4} rx={2} fill="var(--color-yellow)" />}
        <text x={320} y={50} textAnchor="middle" fontSize={26} fontWeight={600} opacity={on('aggregate') ? 1 : 0.5}>
          {l.server}
        </text>
        <text x={320} y={80} textAnchor="middle" fontSize={21} className="label-muted" opacity={on('aggregate') ? 1 : 0.5}>
          {l.serverDetail}
        </text>

        {/* Links */}
        {CLIENT_X.map((x, i) => {
          const cx = x + 80;
          const sx = 280 + i * 40;
          const up = on('upload');
          const down = on('download');
          return (
            <g key={x}>
              <line x1={cx - 12} y1={214} x2={sx - 8} y2={100} stroke={up ? accent : faded} strokeWidth={up ? 2.5 : 1.5} markerEnd={`url(#${id}-arrow-${up ? 'on' : 'off'})`} />
              <line
                x1={sx + 8}
                y1={100}
                x2={cx + 12}
                y2={214}
                stroke={down ? ink : faded}
                strokeWidth={down ? 2.5 : 1.5}
                strokeDasharray={down ? undefined : '4 4'}
                markerEnd={`url(#${id}-arrow-${down ? 'ink' : 'off'})`}
              />
            </g>
          );
        })}

        {/* Clients */}
        {CLIENT_X.map((x, i) => (
          <g key={x}>
            <rect x={x} y={222} width={160} height={96} rx={14} fill="var(--color-surface)" stroke={on('train') ? ink : faded} strokeWidth={on('train') ? 2 : 1.5} />
            <text x={x + 80} y={256} textAnchor="middle" fontSize={24} fontWeight={600} opacity={on('train') ? 1 : 0.55}>
              {l.client} {i + 1}
            </text>
            {/* Data cylinder */}
            <g opacity={on('data') ? 1 : 0.4}>
              <title>{l.data}</title>
              <ellipse cx={x + 80} cy={274} rx={14} ry={5} fill="none" stroke={ink} strokeWidth={1.5} />
              <path d={`M${x + 66} 274v24a14 5 0 0 0 28 0v-24`} fill="none" stroke={ink} strokeWidth={1.5} />
              {on('data') && active !== 'all' && <circle cx={x + 80} cy={287} r={4} fill="var(--color-orange)" />}
            </g>
          </g>
        ))}
      </svg>
      <figcaption>
        <ul className="legend" aria-label="Diagram legend">
          <li>
            <span className="swatch" style={{ background: accent, height: 3 }} /> Model update sent up (upload)
          </li>
          <li>
            <span className="swatch" style={{ background: ink, height: 3 }} /> Aggregated model sent down (download)
          </li>
          <li>
            <span className="swatch" style={{ border: `1.5px solid ${ink}`, borderRadius: '50% / 25%' }} /> {l.data}: raw data stays on the client
          </li>
        </ul>
      </figcaption>
    </figure>
  );
}

export interface DiagramStep {
  title: string;
  body: ReactNode;
  active: DiagramPart[];
}

/** Step-through walkthrough of the federated round using the shared diagram. */
export function StepThrough({ steps, labels, title }: { steps: DiagramStep[]; labels?: Partial<DiagramLabels>; title: string }) {
  const [index, setIndex] = useState(0);
  const step = steps[index];
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
      <ol className="steps" aria-label="Workflow steps">
        {steps.map((s, i) => (
          <li key={s.title}>
            <button
              type="button"
              className={`step-btn ${i < index ? 'done' : ''}`}
              aria-current={i === index ? 'step' : undefined}
              onClick={() => setIndex(i)}
            >
              <span className="step-num">{i + 1}</span>
              {s.title}
            </button>
          </li>
        ))}
      </ol>
      <div className="progress" aria-hidden>
        <span style={{ width: `${((index + 1) / steps.length) * 100}%` }} />
      </div>
      <FederatedDiagram active={step.active} labels={labels} title={`${title} — step ${index + 1}: ${step.title}`} />
      <div className="callout" aria-live="polite">
        <strong>
          Step {index + 1} of {steps.length}: {step.title}
        </strong>
        <div>{step.body}</div>
      </div>
      <div className="btn-row">
        <button type="button" className="btn btn-secondary" disabled={index === 0} onClick={() => setIndex(index - 1)}>
          Previous
        </button>
        <button type="button" className="btn btn-primary" disabled={index === steps.length - 1} onClick={() => setIndex(index + 1)}>
          Next step
        </button>
      </div>
    </div>
  );
}
