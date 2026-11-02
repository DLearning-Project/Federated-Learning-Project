import type { FftSize, SampleRate, SpectrogramSettings } from './types';

export interface Preset {
  id: string;
  label: string;
  /** One sentence: what changes in the picture and why. */
  teaches: string;
  settings: SpectrogramSettings;
}

/** The PANN MobileNetV2 (AudioSet, 32 kHz) front end: what the pretrained model receives. */
export const PANN_DEFAULT: SpectrogramSettings = {
  sample_rate_hz: 32000,
  n_fft: 1024,
  hop_length: 320,
  n_mels: 64,
  fmin_hz: 50,
  fmax_hz: 14000,
};

export const PRESETS: Preset[] = [
  {
    id: 'pann',
    label: 'PANN default',
    teaches: 'Exactly what the pretrained model receives: 32 ms windows every 10 ms, 64 Mel bands from 50 Hz to 14 kHz.',
    settings: PANN_DEFAULT,
  },
  {
    id: 'time',
    label: 'Sharper in time',
    teaches: 'A short 8 ms window catches quick changes, but each frequency band gets blurry and some low bands end up empty.',
    settings: { ...PANN_DEFAULT, n_fft: 256, hop_length: 128 },
  },
  {
    id: 'frequency',
    label: 'Sharper in frequency',
    teaches: 'A long 128 ms window separates close tones, such as engine harmonics, but smears quick events over time.',
    settings: { ...PANN_DEFAULT, n_fft: 4096, hop_length: 1024 },
  },
  {
    id: 'fewer-mels',
    label: 'Fewer Mel bands',
    teaches: 'Half as many bands: a smaller input for the model, with less frequency detail.',
    settings: { ...PANN_DEFAULT, n_mels: 32 },
  },
  {
    id: '16k',
    label: '16 kHz audio',
    teaches: 'Half the sample rate: nothing above 8 kHz survives, so high-frequency propeller noise disappears.',
    settings: { ...PANN_DEFAULT, sample_rate_hz: 16000, n_fft: 512, hop_length: 160, fmax_hz: 8000 },
  },
  {
    id: 'engine',
    label: 'Engine tones',
    teaches:
      'Zoom in on 0–1000 Hz with a long window, where engine harmonics would appear as steady lines. Caution: the QiandaoEar22 hydrophone is only specified from 1 kHz, so levels here are not calibrated.',
    settings: { ...PANN_DEFAULT, n_fft: 4096, n_mels: 32, fmin_hz: 0, fmax_hz: 1000 },
  },
];

export const SAMPLE_RATES: SampleRate[] = [16000, 32000];
export const FFT_SIZES: FftSize[] = [256, 512, 1024, 2048, 4096];

/** Bounds from backend/app/modules/acoustics/schemas.py. The server validates again. */
export const LIMITS = {
  hop: { min: 128, max: 2048, step: 16 },
  mels: { min: 16, max: 128, step: 8 },
  fmin: { min: 0, max: 8000, step: 10 },
  fmax: { min: 500, max: 16000, step: 100 },
};

const MIN_BAND_HZ = 100; // keep fmin a little below fmax so the band never collapses

/**
 * Pull settings back inside the backend's rules after one control changes, e.g.
 * lowering the sample rate also lowers fmax to the new Nyquist frequency.
 */
export function clampSettings(s: SpectrogramSettings): SpectrogramSettings {
  const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
  const hop_length = clamp(s.hop_length, LIMITS.hop.min, Math.min(LIMITS.hop.max, s.n_fft));
  const fmax_hz = clamp(s.fmax_hz, LIMITS.fmax.min, Math.min(LIMITS.fmax.max, s.sample_rate_hz / 2));
  const fmin_hz = clamp(s.fmin_hz, LIMITS.fmin.min, Math.min(LIMITS.fmin.max, fmax_hz - MIN_BAND_HZ));
  return { ...s, hop_length, fmin_hz, fmax_hz };
}

export function sameSettings(a: SpectrogramSettings, b: SpectrogramSettings): boolean {
  return (Object.keys(PANN_DEFAULT) as (keyof SpectrogramSettings)[]).every((k) => a[k] === b[k]);
}

/** The preset these settings match exactly, or undefined for custom settings. */
export function matchPreset(s: SpectrogramSettings): Preset | undefined {
  return PRESETS.find((p) => sameSettings(p.settings, s));
}
