import { formatInt } from '../../components/format';
import spectrogram from './data/synthetic_spectrogram.json';

/** Neutral sequential ramp: quiet = off-white, loud = charcoal. */
function shade(db: number, min: number, max: number) {
  const t = Math.min(1, Math.max(0, (db - min) / (max - min)));
  const from = [255, 253, 247];
  const to = [36, 36, 36];
  const c = from.map((f, i) => Math.round(f + (to[i] - f) * Math.pow(t, 0.9)));
  return `rgb(${c.join(',')})`;
}

export function Spectrogram() {
  const { db, time_frames: T, frequency_bins: F, duration_s, max_frequency_hz, min_db, max_db } = spectrogram;
  const left = 78;
  const top = 10;
  const cellW = 10;
  const cellH = 7;
  const plotW = T * cellW;
  const plotH = F * cellH;
  const timeTicks = [0, 1, 2, 3, 4, 5, 6];
  const freqTicks = [0, 250, 500, 750, 1000];

  return (
    <figure style={{ margin: 0, display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
      <svg className="chart" viewBox={`0 0 ${left + plotW + 20} ${top + plotH + 64}`} role="img" aria-label="Synthetic spectrogram: a tone near 94 hertz with three harmonics across the whole clip, and a short broadband burst around 3 seconds between 500 and 900 hertz.">
        <g shapeRendering="crispEdges">
          {db.map((row, f) =>
            row.map((v, t) => <rect key={`${f}-${t}`} x={left + t * cellW} y={top + plotH - (f + 1) * cellH} width={cellW} height={cellH} fill={shade(v, min_db, max_db)} />),
          )}
        </g>
        <rect x={left} y={top} width={plotW} height={plotH} fill="none" stroke="var(--color-border-strong)" />
        {timeTicks.map((s) => (
          <text key={s} x={left + (s / duration_s) * plotW} y={top + plotH + 26} textAnchor="middle" fontSize={20} className="label-muted">
            {s}
          </text>
        ))}
        <text x={left + plotW / 2} y={top + plotH + 56} textAnchor="middle" fontSize={20}>
          Time (s)
        </text>
        {freqTicks.map((hz) => (
          <text key={hz} x={left - 8} y={top + plotH - (hz / max_frequency_hz) * plotH + 5} textAnchor="end" fontSize={20} className="label-muted">
            {hz}
          </text>
        ))}
        <text transform={`translate(18 ${top + plotH / 2}) rotate(-90)`} textAnchor="middle" fontSize={20}>
          Frequency (Hz)
        </text>
      </svg>
      <figcaption className="small" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', flexWrap: 'wrap' }}>
          <span className="muted">Quiet</span>
          <span aria-hidden style={{ display: 'inline-flex', border: '1px solid var(--color-border)', borderRadius: 3, overflow: 'hidden' }}>
            {[0, 0.25, 0.5, 0.75, 1].map((t) => (
              <span key={t} style={{ width: '1.5rem', height: '0.75rem', background: shade(min_db + t * (max_db - min_db), min_db, max_db) }} />
            ))}
          </span>
          <span className="muted">Loud (relative dB)</span>
        </div>
        <span className="muted">{spectrogram.description}</span>
      </figcaption>
    </figure>
  );
}

export const CLASS_COLORS = ['var(--color-ink)', 'var(--color-chart-gray)', 'var(--color-yellow)', 'var(--color-orange)'];
const CLASS_TEXT = ['var(--color-surface)', 'var(--color-ink)', 'var(--color-ink)', 'var(--color-ink)'];

/** Horizontal stacked bars: one bar per client, segments per vessel class. */
export function StackedBars({ clients, classes, counts }: { clients: string[]; classes: string[]; counts: number[][] }) {
  const max = Math.max(...counts.map((row) => row.reduce((a, b) => a + b, 0)));
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
      <ul className="legend" aria-label="Vessel classes">
        {classes.map((c, i) => (
          <li key={c}>
            <span className="swatch" style={{ background: CLASS_COLORS[i], border: '1px solid var(--color-ink)' }} />
            {c}
          </li>
        ))}
      </ul>
      {clients.map((name, ci) => {
        const row = counts[ci];
        const total = row.reduce((a, b) => a + b, 0);
        return (
          <div key={name} className="meter">
            <div className="meter-head">
              <strong>{name}</strong>
              <span>{formatInt(total)} samples</span>
            </div>
            <div style={{ display: 'flex', width: `${(total / max) * 100}%`, minWidth: '2rem', gap: 2 }} role="img" aria-label={`${name}: ${classes.map((c, i) => `${c} ${row[i]}`).join(', ')}`}>
              {row.map((v, i) =>
                v > 0 ? (
                  <span
                    key={classes[i]}
                    title={`${classes[i]}: ${v}`}
                    style={{
                      flex: v,
                      minHeight: '1.75rem',
                      background: CLASS_COLORS[i],
                      color: CLASS_TEXT[i],
                      border: '1px solid var(--color-ink)',
                      borderRadius: 4,
                      fontSize: 'var(--text-small)',
                      lineHeight: '1.6rem',
                      textAlign: 'center',
                      overflow: 'hidden',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {v / total >= 0.12 ? v : ''}
                  </span>
                ) : null,
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
