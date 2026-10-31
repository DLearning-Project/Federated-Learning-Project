import type { CalcEnvelope } from '../../services/api';

/** Mirrors backend/app/modules/acoustics/schemas.py — keep field names in sync. */
export interface ClientUpdate {
  name: string;
  sample_count: number;
  vector: number[];
}

export interface FedAvgInput {
  clients: ClientUpdate[];
}

export interface ClientWeight {
  name: string;
  sample_count: number;
  weight: number;
}

export interface FedAvgResult {
  total_samples: number;
  weights: ClientWeight[];
  global_vector: number[];
  unweighted_mean: number[];
}

export type FedAvgResponse = CalcEnvelope<FedAvgInput, FedAvgResult>;

export type SampleRate = 16000 | 32000;
export type FftSize = 256 | 512 | 1024 | 2048 | 4096;

/** The front-end settings the presets and Advanced controls change. */
export interface SpectrogramSettings {
  sample_rate_hz: SampleRate;
  n_fft: FftSize;
  hop_length: number;
  n_mels: number;
  fmin_hz: number;
  fmax_hz: number;
}

export interface SpectrogramInput extends SpectrogramSettings {
  source: 'synthetic' | 'recording';
  clip_id: string | null;
}

export interface StageTimings {
  read_ms: number;
  resample_ms: number;
  stft_ms: number;
  mel_ms: number;
  total_ms: number;
}

export interface SpectrogramResult {
  source_label: string;
  recording_id: string | null;
  original_sample_rate_hz: number;
  clip_seconds: number;
  rms_dbfs: number;
  frames: number;
  mel_bins: number;
  frame_seconds: number;
  empty_mel_filters: number;
  db_max: number;
  db_min_shown: number;
  mel_center_hz: number[];
  spectrogram_db: number[][]; // [mel bin, low to high frequency][frame]
  timings: StageTimings;
}

export type SpectrogramResponse = CalcEnvelope<SpectrogramInput, SpectrogramResult>;

export type ClipCategory = 'speedboat_target' | 'speedboat_other' | 'noise';

export interface ClipInfo {
  id: string;
  label: string;
  category: ClipCategory;
  recording_id: string;
}

export interface ClipList {
  dataset_found: boolean;
  folder_hint: string;
  clips: ClipInfo[];
}

export interface ClientRow {
  name: string;
  samples: string;
  vector: string[];
}

export type Scenario = 'balanced' | 'skewed';
