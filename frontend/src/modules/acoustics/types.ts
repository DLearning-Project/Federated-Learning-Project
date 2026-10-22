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

export interface ClientRow {
  name: string;
  samples: string;
  vector: string[];
}

export type Scenario = 'balanced' | 'skewed';
