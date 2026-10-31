import { useEffect, useId, useMemo, useRef, type ReactNode } from 'react';
import { useModuleState } from '../../app/AppStateContext';
import { Card } from '../../components/Card';
import { Details } from '../../components/Details';
import { formatInt, formatNumber } from '../../components/format';
import { AssumptionsList, ModeBadge, StatGrid } from '../../components/Results';
import { ApiErrorState, LoadingState, StatusMessage } from '../../components/StatusMessage';
import { useApiAction } from '../../hooks/useApiAction';
import { getSpectrogram, listClips } from './api';
import { SpectrogramChart } from './Charts';
import { FFT_SIZES, LIMITS, PANN_DEFAULT, PRESETS, SAMPLE_RATES, clampSettings, matchPreset, sameSettings } from './presets';
import type { ClipCategory, ClipInfo, SpectrogramInput, SpectrogramSettings } from './types';

const SYNTHETIC = 'synthetic';
const SLIDER_DEBOUNCE_MS = 250;

const CATEGORY_LABELS: Record<ClipCategory, string> = {
  speedboat_target: 'Speedboat present',
  speedboat_other: 'Other vessels, no speedboat',
  noise: 'Background noise, no vessel',
};

function sameInput(a: SpectrogramInput, b: SpectrogramInput) {
  return a.source === b.source && a.clip_id === b.clip_id && sameSettings(a, b);
}

function RangeField({ label, value, unit, min, max, step, onChange }: { label: string; value: number; unit: string; min: number; max: number; step: number; onChange: (v: number) => void }) {
  const id = useId();
  return (
    <div className="field">
      <label className="field-label" htmlFor={id}>
        <span>{label}</span>
        <span className="field-unit">
          {formatInt(value)} {unit}
        </span>
      </label>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} style={{ accentColor: 'var(--color-ink)', width: '100%' }} />
    </div>
  );
}

function SelectField({ label, value, options, onChange, children }: { label: string; value: number; options: number[]; onChange: (v: number) => void; children?: ReactNode }) {
  const id = useId();
  return (
    <div className="field">
      <label className="field-label" htmlFor={id}>
        {label}
      </label>
      <select id={id} className="select" value={value} onChange={(e) => onChange(Number(e.target.value))}>
        {options.map((o) => (
          <option key={o} value={o}>
            {formatInt(o)}
          </option>
        ))}
      </select>
      {children}
    </div>
  );
}

/**
 * Pick a real QiandaoEar22 clip (or the synthetic signal) and a preset; the backend
 * computes the log-Mel spectrogram live for those exact settings.
 */
export function SpectrogramExplorer() {
  const [selection, setSelection] = useModuleState<string>('ac.spec.selection', ''); // '' = not chosen yet
  const [settings, setSettings] = useModuleState<SpectrogramSettings>('ac.spec.settings', PANN_DEFAULT);
  const clips = useApiAction('ac.spec.clips', listClips);
  const spec = useApiAction('ac.spec.result', getSpectrogram);
  const immediateRef = useRef(true);

  const runClips = clips.run;
  useEffect(() => {
    if (clips.state.status === 'idle') void runClips(undefined);
  }, [clips.state.status, runClips]);

  // Default: the first speedboat recording when the dataset is on the server, else the synthetic signal.
  const clipList = clips.data;
  useEffect(() => {
    if (selection || !clipList) return;
    const first = clipList.clips.find((c) => c.category === 'speedboat_target') ?? clipList.clips[0];
    setSelection(first ? first.id : SYNTHETIC);
  }, [clipList, selection, setSelection]);

  const input = useMemo<SpectrogramInput | null>(() => {
    if (!selection) return null;
    const recording = selection !== SYNTHETIC;
    return { source: recording ? 'recording' : 'synthetic', clip_id: recording ? selection : null, ...settings };
  }, [selection, settings]);

  // Fetch whenever the inputs change: right away for clip and preset choices,
  // after a short pause for sliders so dragging does not send a request per pixel.
  const runSpec = spec.run;
  const current = spec.data;
  useEffect(() => {
    if (!input || (current && sameInput(current.inputs, input))) return;
    const delay = immediateRef.current ? 0 : SLIDER_DEBOUNCE_MS;
    immediateRef.current = false;
    const timer = window.setTimeout(() => void runSpec(input), delay);
    return () => window.clearTimeout(timer);
    // `current` is deliberately left out: a new result must not trigger another request.
  }, [input, runSpec]);

  const choose = (next: SpectrogramSettings) => {
    immediateRef.current = true;
    setSettings(next);
  };
  const adjust = (patch: Partial<SpectrogramSettings>) => setSettings((s) => clampSettings({ ...s, ...patch }));

  const preset = matchPreset(settings);
  const grouped = useMemo(() => {
    const groups = new Map<ClipCategory, ClipInfo[]>();
    for (const c of clipList?.clips ?? []) groups.set(c.category, [...(groups.get(c.category) ?? []), c]);
    return groups;
  }, [clipList]);

  const result = current?.results;
  const loading = spec.state.status === 'loading';
  const hopMax = Math.min(LIMITS.hop.max, settings.n_fft);
  const fmaxMax = Math.min(LIMITS.fmax.max, settings.sample_rate_hz / 2);

  return (
    <Card
      title="What the model hears"
      subtitle={current ? (current.mode === 'measured' ? 'Computed live from a real recording' : 'Synthetic signal: generated, not recorded') : 'Log-Mel spectrogram, computed live by the backend'}
      actions={current && <ModeBadge mode={current.mode} />}
    >
      <div className="workbench">
        <div className="form">
          <div className="field">
            <label className="field-label" htmlFor="ac-clip">
              Audio
            </label>
            <select
              id="ac-clip"
              className="select"
              value={selection}
              disabled={!clipList}
              onChange={(e) => {
                immediateRef.current = true;
                setSelection(e.target.value);
              }}
            >
              {!selection && <option value="">{clips.state.status === 'error' ? 'Clips unavailable' : 'Loading clips…'}</option>}
              <option value={SYNTHETIC}>Synthetic engine and propeller signal (not a recording)</option>
              {[...grouped].map(([category, list]) => (
                <optgroup key={category} label={CATEGORY_LABELS[category]}>
                  {list.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.category === 'noise' ? c.label : `${c.label} · recording ${c.recording_id}`}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
            {clipList && !clipList.dataset_found && (
              <span className="field-hint">
                Only the synthetic signal is available. Real QiandaoEar22 recordings appear here when the dataset is downloaded on the server, in <code>{clipList.folder_hint}</code>.
              </span>
            )}
          </div>

          <div className="field">
            <span className="field-label">Preset</span>
            <div className="segmented" role="group" aria-label="Spectrogram preset">
              {PRESETS.map((p) => (
                <button key={p.id} type="button" aria-pressed={preset?.id === p.id} onClick={() => choose(p.settings)}>
                  {p.label}
                </button>
              ))}
            </div>
            <span className="field-hint">{preset ? preset.teaches : 'Custom settings from the Advanced controls below.'}</span>
          </div>

          <Details summary="Advanced: set each value yourself">
            <div className="form">
              <SelectField label="Sample rate (Hz)" value={settings.sample_rate_hz} options={SAMPLE_RATES} onChange={(v) => adjust({ sample_rate_hz: v as SpectrogramSettings['sample_rate_hz'] })}>
                <span className="field-hint">The pretrained PANN model expects 32 kHz.</span>
              </SelectField>
              <SelectField label="FFT window (samples)" value={settings.n_fft} options={FFT_SIZES} onChange={(v) => adjust({ n_fft: v as SpectrogramSettings['n_fft'] })}>
                <span className="field-hint">{formatNumber((settings.n_fft / settings.sample_rate_hz) * 1000, 0)} ms per window: longer means sharper frequency, blurrier time.</span>
              </SelectField>
              <RangeField label="Hop between frames" unit="samples" value={settings.hop_length} min={LIMITS.hop.min} max={hopMax} step={LIMITS.hop.step} onChange={(v) => adjust({ hop_length: v })} />
              <RangeField label="Mel bands" unit="bands" value={settings.n_mels} min={LIMITS.mels.min} max={LIMITS.mels.max} step={LIMITS.mels.step} onChange={(v) => adjust({ n_mels: v })} />
              <RangeField label="Lowest frequency" unit="Hz" value={settings.fmin_hz} min={LIMITS.fmin.min} max={Math.min(LIMITS.fmin.max, settings.fmax_hz - 100)} step={LIMITS.fmin.step} onChange={(v) => adjust({ fmin_hz: v })} />
              <RangeField label="Highest frequency" unit="Hz" value={settings.fmax_hz} min={LIMITS.fmax.min} max={fmaxMax} step={LIMITS.fmax.step} onChange={(v) => adjust({ fmax_hz: v })} />
              <div>
                <button type="button" className="btn btn-secondary" onClick={() => choose(PANN_DEFAULT)} disabled={sameSettings(settings, PANN_DEFAULT)}>
                  Back to PANN default
                </button>
              </div>
            </div>
          </Details>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)', minWidth: 0 }}>
          {clips.state.status === 'error' && <ApiErrorState error={clips.state.error} onRetry={() => void runClips(undefined)} />}
          {spec.state.status === 'error' && <ApiErrorState error={spec.state.error} onRetry={input ? () => void runSpec(input) : undefined} />}
          {!result && (loading || clips.state.status === 'loading') && <LoadingState label="Computing spectrogram…" />}

          {result && (
            <>
              <p className="small" style={{ fontWeight: 600 }}>
                {result.source_label}
                {result.recording_id && <span className="muted" style={{ fontWeight: 400 }}> · recording {result.recording_id}</span>}
              </p>
              <SpectrogramChart result={result} dimmed={loading} />
              {result.empty_mel_filters > 0 && (
                <StatusMessage tone="warning" title={`${result.empty_mel_filters} Mel bands are empty`}>
                  The FFT is too coarse for this many narrow low-frequency bands, so some rows stay blank. Try a larger FFT size or fewer Mel bands.
                </StatusMessage>
              )}
              <StatGrid
                stats={[
                  { label: 'Original sample rate', value: `${formatNumber(result.original_sample_rate_hz / 1000, 1)} kHz`, note: `resampled to ${formatNumber(current.inputs.sample_rate_hz / 1000, 0)} kHz` },
                  { label: 'Model input size', value: `${result.mel_bins} × ${formatInt(result.frames)}`, note: 'Mel bands × time frames' },
                  { label: 'Loudness (RMS)', value: `${formatNumber(result.rms_dbfs, 1)} dBFS`, note: 'relative to full scale' },
                  { label: 'Server time', value: `${formatNumber(result.timings.total_ms, 0)} ms`, note: 'read, resample, STFT, Mel' },
                ]}
              />
              <AssumptionsList assumptions={current.assumptions} />
            </>
          )}
        </div>
      </div>
    </Card>
  );
}
