import { formatInt, formatPercent } from '../../components/format';
import type { LoraCalcResult } from './types';

const MAX_SIDE = 170;
const MIN_SIDE = 4;

/**
 * Draws the shapes of W (frozen), B and A (trainable) to scale.
 * Only geometry is derived here; parameter counts come from the backend.
 */
export function MatrixShapes({ dIn, dOut, rank }: { dIn: number; dOut: number; rank: number }) {
  const scale = MAX_SIDE / Math.max(dIn, dOut);
  const side = (n: number) => Math.max(MIN_SIDE, n * scale);
  const wW = side(dIn);
  const wH = side(dOut);
  const r = side(rank);

  // Each matrix sits centred in a column wide enough for its labels.
  const top = 36;
  const gap = 40;
  const colW = Math.max(wW, 120);
  const colB = Math.max(r, 90);
  const colA = Math.max(wW, 90);
  const wX = (colW - wW) / 2;
  const bX = colW + gap + (colB - r) / 2;
  const aX = colW + gap + colB + gap + (colA - wW) / 2;
  const width = Math.max(colW + gap + colB + gap + colA, 420);
  const height = top + Math.max(wH, r) + 60;

  return (
    <svg className="chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Frozen W is ${dOut} by ${dIn}; B is ${dOut} by ${rank}; A is ${rank} by ${dIn}.`} style={{ maxHeight: '22rem' }}>
      <rect x={wX} y={top} width={wW} height={wH} fill="var(--color-border)" stroke="var(--color-chart-gray)" rx={3} />
      <text x={wX + wW / 2} y={top + wH + 24} textAnchor="middle" fontSize={16} fontWeight={600}>
        W (frozen)
      </text>
      <text x={wX + wW / 2} y={top + wH + 44} textAnchor="middle" fontSize={16} className="label-muted">
        {dOut} × {dIn}
      </text>

      <text x={colW + gap / 2} y={top + Math.min(wH, 60) / 2 + 6} textAnchor="middle" fontSize={22} fontWeight={600}>
        +
      </text>

      <rect x={bX} y={top} width={r} height={wH} fill="var(--color-yellow)" stroke="var(--color-ink)" rx={2} />
      <text x={bX + r / 2} y={top + wH + 24} textAnchor="middle" fontSize={16} fontWeight={600}>
        B
      </text>
      <text x={bX + r / 2} y={top + wH + 44} textAnchor="middle" fontSize={16} className="label-muted">
        {dOut}×{rank}
      </text>

      <text x={colW + gap + colB + gap / 2} y={top + Math.min(wH, 60) / 2 + 6} textAnchor="middle" fontSize={20} fontWeight={600}>
        ×
      </text>

      <rect x={aX} y={top} width={wW} height={r} fill="var(--color-yellow)" stroke="var(--color-ink)" rx={2} />
      <text x={aX + wW / 2} y={top + r + 24} textAnchor="middle" fontSize={16} fontWeight={600}>
        A
      </text>
      <text x={aX + wW / 2} y={top + r + 44} textAnchor="middle" fontSize={16} className="label-muted">
        {rank}×{dIn}
      </text>
      <text x={0} y={16} fontSize={16} className="label-muted">
        Drawn to scale (very thin strips get a minimum width)
      </text>
    </svg>
  );
}

/** Compares full-matrix and adapter parameter counts returned by the API. */
export function SavingsBars({ results }: { results: LoraCalcResult }) {
  const rows = [
    { label: 'Full fine-tuning of W', value: results.full_params, fraction: 1, color: 'var(--color-chart-light)' },
    { label: 'LoRA adapters (A + B)', value: results.adapter_params, fraction: results.adapter_fraction, color: 'var(--color-orange)' },
  ];
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
      {rows.map((row) => (
        <div className="meter" key={row.label}>
          <div className="meter-head">
            <strong>{row.label}</strong>
            <span>
              {formatInt(row.value)} parameters ({formatPercent(row.fraction, row.fraction < 0.01 ? 2 : 1)})
            </span>
          </div>
          <div className="meter-track" aria-hidden>
            <span className="meter-fill" style={{ width: `max(3px, ${row.fraction * 100}%)`, background: row.color }} />
          </div>
        </div>
      ))}
    </div>
  );
}
