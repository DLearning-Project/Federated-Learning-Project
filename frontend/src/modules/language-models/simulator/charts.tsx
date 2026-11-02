import { formatBytes } from '../../../components/format';
import type { MethodId, MethodPayload } from '../types';

/** Unicode subscript digits for native SVG tooltips, which cannot render KaTeX (σ₁, σ₁₂). */
const subscript = (n: number) => String(n).replace(/\d/g, (d) => '₀₁₂₃₄₅₆₇₈₉'[Number(d)]);

export const METHOD_LABEL: Record<MethodId, string> = {
  florist: 'FLoRIST',
  fedit: 'FedIT',
  exact_stack: 'Exact stacking',
};

/**
 * Singular values of the aggregate, ordered largest first. Kept components
 * (the first p) are orange; discarded ones are light gray. The ink line is the
 * cumulative energy, and the dashed line is the threshold τ.
 */
export function SpectrumChart({
  values,
  energy,
  p,
  tau,
  xLabel = 'component (largest first)',
  plain = false,
}: {
  values: number[];
  energy: number[];
  p: number;
  tau: number;
  xLabel?: string;
  /** Beginner wording: no σ, τ or p symbols on the chart. */
  plain?: boolean;
}) {
  const W = 640;
  const H = 260;
  const pad = { l: 44, r: 44, t: 16, b: 36 };
  const iw = W - pad.l - pad.r;
  const ih = H - pad.t - pad.b;
  const n = Math.max(values.length, 1);
  const max = Math.max(...values, 1e-12);
  const slot = iw / n;
  const bw = Math.max(2, Math.min(28, slot * 0.7));
  const x = (i: number) => pad.l + slot * i + slot / 2;
  const y = (v: number) => pad.t + ih - (v / max) * ih;
  const ye = (e: number) => pad.t + ih - e * ih;
  const line = energy.map((e, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${ye(e).toFixed(1)}`).join(' ');
  const cutX = pad.l + slot * p;
  const desc = `Singular values of the aggregated update: ${values.length} components, FLoRIST keeps the first ${p} to reach ${Math.round(
    tau * 100,
  )}% of the energy.`;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart lm-chart" role="img" aria-label={desc}>
      <line x1={pad.l} x2={W - pad.r} y1={pad.t + ih} y2={pad.t + ih} stroke="var(--color-border-strong)" />
      {values.map((v, i) => (
        <rect
          key={i}
          x={x(i) - bw / 2}
          y={y(v)}
          width={bw}
          height={Math.max(0, pad.t + ih - y(v))}
          fill={i < p ? 'var(--color-orange)' : 'var(--color-chart-light)'}
          stroke={i < p ? 'none' : 'var(--color-chart-gray)'}
        >
          <title>{`σ${subscript(i + 1)} = ${v.toPrecision(3)} (${i < p ? 'kept' : 'discarded'}), cumulative energy ${(energy[i] * 100).toFixed(1)}%`}</title>
        </rect>
      ))}
      <path d={line} fill="none" stroke="var(--color-ink)" strokeWidth={2} />
      {energy.map((e, i) => (
        <circle key={i} cx={x(i)} cy={ye(e)} r={2.5} fill="var(--color-ink)" />
      ))}
      <line x1={pad.l} x2={W - pad.r} y1={ye(tau)} y2={ye(tau)} stroke="var(--color-orange-text)" strokeDasharray="5 4" />
      <text x={W - pad.r + 4} y={ye(tau) + 4} className="label-muted">
        {plain ? 'target' : <tspan className="lm-svg-math">τ</tspan>}
      </text>
      {p > 0 && p < values.length && (
        <>
          <line x1={cutX} x2={cutX} y1={pad.t} y2={pad.t + ih} stroke="var(--color-ink)" strokeDasharray="2 3" />
          <text x={cutX + 4} y={pad.t + ih * 0.55}>
            {plain ? (
              `keep ${p}`
            ) : (
              <>
                <tspan className="lm-svg-math">p</tspan> = {p}
              </>
            )}
          </text>
        </>
      )}
      <text x={pad.l - 8} y={pad.t + 8} textAnchor="end" className="label-muted">
        {plain ? '' : <tspan className="lm-svg-math">σ</tspan>}
      </text>
      <text x={W - pad.r + 4} y={pad.t + 8} className="label-muted">
        100%
      </text>
      <text x={W - pad.r + 4} y={pad.t + ih} className="label-muted">
        0%
      </text>
      <text x={pad.l + iw / 2} y={H - 8} textAnchor="middle" className="label-muted">
        {xLabel.length === 1 ? <tspan className="lm-svg-math">{xLabel}</tspan> : xLabel}
      </text>
    </svg>
  );
}

const METHOD_STYLE: Record<MethodId, { fill: string; className: string }> = {
  florist: { fill: 'var(--color-ink)', className: 'lm-bar-solid' },
  fedit: { fill: 'var(--color-orange)', className: 'lm-bar-solid' },
  exact_stack: {
    fill: 'var(--color-chart-light)',
    className: 'lm-bar-outline',
  },
};

/** Per-client download size per round, one bar per method. */
export function DownloadBars({
  payload,
  labels = METHOD_LABEL,
  showRank = true,
}: {
  payload: MethodPayload[];
  labels?: Record<MethodId, string>;
  showRank?: boolean;
}) {
  const order: MethodId[] = ['exact_stack', 'fedit', 'florist'];
  const rows = order.map((m) => payload.find((p) => p.method === m)).filter((p): p is MethodPayload => !!p);
  const max = Math.max(...rows.map((r) => r.download_per_client_bytes), 1);
  return (
    <ul className="lm-bars" aria-label="Download size per client per round">
      {rows.map((r) => (
        <li key={r.method}>
          <span className="lm-bars-label">{labels[r.method]}</span>
          <span className="lm-bars-track">
            <span
              className={`lm-bars-fill ${METHOD_STYLE[r.method].className}`}
              style={{
                width: `${(r.download_per_client_bytes / max) * 100}%`,
                background: METHOD_STYLE[r.method].fill,
              }}
            />
          </span>
          <span className="lm-bars-value">
            {formatBytes(r.download_per_client_bytes)} {showRank && <span className="muted small">rank {r.download_rank}</span>}
          </span>
        </li>
      ))}
    </ul>
  );
}

/**
 * Class mix of every client: one row per client, one cell per class, darker =
 * more samples of that class. Sampled clients this round are marked.
 */
export function ClassMixGrid({ counts, sampled, ranks }: { counts: number[][]; sampled: number[]; ranks: number[] }) {
  const max = Math.max(...counts.flat(), 1);
  const picked = new Set(sampled);
  return (
    <div
      className="lm-mix"
      role="img"
      aria-label={`Class mix of ${counts.length} clients over ${counts[0]?.length ?? 0} classes; darker cells hold more samples.`}
    >
      {counts.map((row, c) => {
        const total = row.reduce((a, b) => a + b, 0);
        return (
          <div key={c} className={`lm-mix-row ${picked.has(c) ? 'picked' : ''}`}>
            <span className="lm-mix-id">
              {picked.has(c) ? '●' : ''} #{c + 1}
            </span>
            <span className="lm-mix-cells">
              {row.map((v, k) => (
                <span
                  key={k}
                  className="lm-mix-cell"
                  style={{ opacity: v === 0 ? 0.06 : 0.15 + 0.85 * (v / max) }}
                  title={`Client ${c + 1}, class ${k + 1}: ${v} samples`}
                />
              ))}
            </span>
            <span className="lm-mix-meta small muted">
              {total} · r={ranks[c]}
            </span>
          </div>
        );
      })}
    </div>
  );
}
