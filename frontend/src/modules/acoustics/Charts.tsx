import { useEffect, useMemo, useRef } from 'react';
import { formatInt, formatNumber } from '../../components/format';
import type { SpectrogramResult } from './types';

/** Neutral sequential ramp: quiet = off-white, loud = charcoal. `t` is 0..1. */
function shadeRgb(t: number): [number, number, number] {
  const from = [255, 253, 247];
  const to = [36, 36, 36];
  const k = Math.pow(Math.min(1, Math.max(0, t)), 0.9);
  return from.map((f, i) => Math.round(f + (to[i] - f) * k)) as [number, number, number];
}

const RAMP = Array.from({ length: 256 }, (_, i) => shadeRgb(i / 255));
const FREQ_TICKS_HZ = [50, 100, 200, 500, 1000, 2000, 4000, 8000, 14000];
const MIN_TICK_GAP = 0.09; // fraction of the axis between labels, so they never overlap
const QUIET_PERCENTILE = 0.02; // colors span this percentile up to the loudest cell

/**
 * dB range the colors span: the clip's 2nd percentile up to its loudest cell.
 * Real recordings use only part of the backend's 80 dB display range, so a fixed
 * range would render them mostly mid-gray. Display only; the legend shows the values.
 */
function colorRange(db: number[][], dbMax: number): [number, number] {
  const values = db.flat().sort((a, b) => a - b);
  const low = values[Math.floor(QUIET_PERCENTILE * (values.length - 1))] ?? dbMax - 1;
  return [low, Math.max(dbMax, low + 1)];
}

function hzLabel(hz: number) {
  return hz >= 1000 ? `${formatNumber(hz / 1000, 1)}k` : String(hz);
}

/** Where `hz` sits on the Mel axis (0 = bottom, 1 = top), interpolating between band centers. */
function melAxisPosition(hz: number, centers: number[]): number | null {
  const n = centers.length;
  if (n < 2 || hz < centers[0] || hz > centers[n - 1]) return null;
  let i = 0;
  while (i < n - 2 && centers[i + 1] < hz) i++;
  const frac = (hz - centers[i]) / (centers[i + 1] - centers[i]);
  return (i + frac + 0.5) / n;
}

/**
 * Log-Mel spectrogram drawn on a canvas, one pixel per (frame, Mel band) cell and
 * scaled up without smoothing. Axes are positioned HTML so text stays crisp.
 */
export function SpectrogramChart({ result, dimmed = false }: { result: SpectrogramResult; dimmed?: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { spectrogram_db: db, frames, mel_bins: bins, db_max, mel_center_hz, frame_seconds } = result;
  const [quietDb, loudDb] = useMemo(() => colorRange(db, db_max), [db, db_max]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    canvas.width = frames;
    canvas.height = bins;
    const image = ctx.createImageData(frames, bins);
    const range = loudDb - quietDb;
    for (let m = 0; m < bins; m++) {
      const row = db[m];
      const y = bins - 1 - m; // low frequencies at the bottom
      for (let t = 0; t < frames; t++) {
        const level = Math.min(255, Math.max(0, Math.round(((row[t] - quietDb) / range) * 255)));
        const [r, g, b] = RAMP[level];
        const o = (y * frames + t) * 4;
        image.data[o] = r;
        image.data[o + 1] = g;
        image.data[o + 2] = b;
        image.data[o + 3] = 255;
      }
    }
    ctx.putImageData(image, 0, 0);
  }, [db, frames, bins, quietDb, loudDb]);

  const duration = frames * frame_seconds;
  const timeTicks = Array.from({ length: Math.floor(duration) + 1 }, (_, s) => s);
  const freqTicks: { hz: number; pos: number }[] = [];
  for (const hz of FREQ_TICKS_HZ) {
    const pos = melAxisPosition(hz, mel_center_hz);
    if (pos !== null && (!freqTicks.length || pos - freqTicks[freqTicks.length - 1].pos >= MIN_TICK_GAP)) freqTicks.push({ hz, pos });
  }
  const lowHz = Math.round(mel_center_hz[0]);
  const highHz = Math.round(mel_center_hz[mel_center_hz.length - 1]);

  return (
    <figure style={{ margin: 0, display: 'flex', flexDirection: 'column', gap: 'var(--space-3)', opacity: dimmed ? 0.55 : 1, transition: 'opacity 120ms' }}>
      <div style={{ display: 'grid', gridTemplateColumns: '3rem minmax(0, 1fr)', gridTemplateRows: 'auto 15rem auto auto', columnGap: 'var(--space-2)' }}>
        <span />
        <span className="small muted">Frequency (Hz, Mel-spaced)</span>
        <div style={{ position: 'relative' }} aria-hidden>
          {freqTicks.map((t) => (
            <span key={t.hz} className="small muted" style={{ position: 'absolute', right: 0, bottom: `${t.pos * 100}%`, transform: 'translateY(50%)', lineHeight: 1 }}>
              {hzLabel(t.hz)}
            </span>
          ))}
        </div>
        <canvas
          ref={canvasRef}
          role="img"
          aria-label={`Log-Mel spectrogram: ${formatInt(frames)} time frames by ${bins} Mel bands, ${formatNumber(duration, 1)} seconds from ${lowHz} to ${highHz} hertz, shaded from ${formatNumber(quietDb, 0)} to ${formatNumber(loudDb, 0)} dB.`}
          style={{ width: '100%', height: '100%', imageRendering: 'pixelated', border: '1px solid var(--color-border-strong)', borderRadius: 4, background: 'var(--color-surface)' }}
        />
        <span />
        <div style={{ position: 'relative', height: '1.75rem' }} aria-hidden>
          {timeTicks.map((s) => (
            <span key={s} className="small muted" style={{ position: 'absolute', left: `${(s / duration) * 100}%`, top: 'var(--space-1)', transform: s === 0 ? 'none' : 'translateX(-50%)' }}>
              {s}
            </span>
          ))}
        </div>
        <span />
        <span className="small muted" style={{ textAlign: 'center' }}>
          Time (s)
        </span>
      </div>
      <figcaption className="small" style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', flexWrap: 'wrap' }}>
        <span className="muted">Quiet · {formatNumber(quietDb, 0)} dB or less</span>
        <span aria-hidden style={{ display: 'inline-flex', border: '1px solid var(--color-border)', borderRadius: 3, overflow: 'hidden' }}>
          {[0, 0.25, 0.5, 0.75, 1].map((t) => (
            <span key={t} style={{ width: '1.5rem', height: '0.75rem', background: `rgb(${shadeRgb(t).join(',')})` }} />
          ))}
        </span>
        <span className="muted">Loud · {formatNumber(loudDb, 0)} dB</span>
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
