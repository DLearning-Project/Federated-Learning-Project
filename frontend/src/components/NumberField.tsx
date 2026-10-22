import { useId, type ReactNode } from 'react';

interface NumberFieldProps {
  label: ReactNode;
  value: string;
  onChange: (value: string) => void;
  unit?: string;
  hint?: ReactNode;
  error?: string | null;
  min?: number;
  max?: number;
  step?: number | 'any';
}

/** Numeric input kept as a string so users can type freely; parse on submit. */
export function NumberField({ label, value, onChange, unit, hint, error, min, max, step = 1 }: NumberFieldProps) {
  const id = useId();
  const hintId = `${id}-hint`;
  const errorId = `${id}-error`;
  const describedBy = [hint ? hintId : null, error ? errorId : null].filter(Boolean).join(' ') || undefined;
  return (
    <div className="field">
      <label className="field-label" htmlFor={id}>
        <span>{label}</span>
        {unit && <span className="field-unit">{unit}</span>}
      </label>
      <input
        id={id}
        className="input"
        type="number"
        inputMode="decimal"
        value={value}
        min={min}
        max={max}
        step={step}
        aria-invalid={error ? true : undefined}
        aria-describedby={describedBy}
        onChange={(e) => onChange(e.target.value)}
      />
      {hint && !error && (
        <span id={hintId} className="field-hint">
          {hint}
        </span>
      )}
      {error && (
        <span id={errorId} className="field-error">
          {error}
        </span>
      )}
    </div>
  );
}

/** Parse a numeric field and enforce bounds; returns [value, errorMessage]. */
export function parseNumber(
  raw: string,
  { min, max, integer = false, label = 'Value' }: { min?: number; max?: number; integer?: boolean; label?: string },
): [number, string | null] {
  if (raw.trim() === '') return [NaN, `${label} is required.`];
  const n = Number(raw);
  if (!Number.isFinite(n)) return [NaN, `${label} must be a number.`];
  if (integer && !Number.isInteger(n)) return [n, `${label} must be a whole number.`];
  if (min !== undefined && n < min) return [n, `${label} must be at least ${min}.`];
  if (max !== undefined && n > max) return [n, `${label} must be at most ${max}.`];
  return [n, null];
}
