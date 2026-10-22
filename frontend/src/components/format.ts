/** Display formatting only — no domain formulas live here. */

const nf = new Intl.NumberFormat('en-US');

export function formatInt(n: number): string {
  return nf.format(Math.round(n));
}

export function formatNumber(n: number, digits = 2): string {
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: digits }).format(n);
}

/** Decimal (SI) byte units: 1 kB = 1,000 bytes, 1 MB = 1,000,000 bytes. */
export function formatBytes(bytes: number): string {
  const units = ['B', 'kB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let i = 0;
  while (Math.abs(value) >= 1000 && i < units.length - 1) {
    value /= 1000;
    i++;
  }
  return `${formatNumber(value, value < 10 && i > 0 ? 2 : 1)} ${units[i]}`;
}

export function formatPercent(fraction: number, digits = 1): string {
  return `${formatNumber(fraction * 100, digits)}%`;
}

export function formatSeconds(s: number): string {
  if (s < 60) return `${formatNumber(s, 1)} s`;
  const m = s / 60;
  if (m < 60) return `${formatNumber(m, 1)} min`;
  return `${formatNumber(m / 60, 2)} h`;
}
