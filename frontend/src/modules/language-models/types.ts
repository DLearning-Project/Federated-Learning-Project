import type { CalcEnvelope } from '../../services/api';

/** Mirrors backend/app/modules/language_models/schemas.py — keep field names in sync. */
export interface LoraCalcInput {
  d_in: number;
  d_out: number;
  rank: number;
  clients: number;
  rounds: number;
  bytes_per_param: 1 | 2 | 4;
}

export interface LoraCalcResult {
  full_params: number;
  adapter_params: number;
  a_params: number;
  b_params: number;
  adapter_fraction: number;
  savings_ratio: number;
  per_client_update_bytes: number;
  per_round_upload_bytes: number;
  total_upload_bytes: number;
  total_download_bytes: number;
  total_bytes: number;
  full_update_bytes: number;
}

export type LoraCalcResponse = CalcEnvelope<LoraCalcInput, LoraCalcResult>;

/** Raw form values, kept as strings while the user edits. */
export type LoraForm = Record<keyof LoraCalcInput, string>;

/* ---------- Simulator page (mirrors SimulatorRoundInput / SimulatorRoundResponse) ---------- */

export interface SimulatorRoundInput {
  total_clients: number;
  clients_per_round: number;
  round: number;
  alpha: number;
  heterogeneous: boolean;
  tau: number;
  seed: number;
  bytes_per_param: 1 | 2 | 4;
}

export interface ErrorValue {
  absolute: number;
  relative: number | null;
}

export type MethodId = 'fedit' | 'exact_stack' | 'florist';

export interface MethodPayload {
  method: MethodId;
  download_rank: number;
  download_per_client_bytes: number;
  download_total_bytes: number;
  upload_total_bytes: number;
}

export interface ServerCost {
  fedit_ops: number | null;
  florist_ops: number;
  direct_svd_ops: number;
  stacked_rank: number;
  stacked_rank_below_min_dim: boolean;
  assumptions: string[];
}

export interface SimulatorRoundResult {
  provenance: 'computed_live_synthetic';
  m: number;
  n: number;
  sampled_clients: number[];
  client_ranks: number[];
  all_client_ranks: number[];
  sample_counts: number[];
  weights: number[];
  class_counts: number[][];
  p: number;
  singular_values: number[];
  cumulative_energy: number[];
  reconstruction_error: ErrorValue;
  eq5_bound: number;
  aggregation_error_fedit: ErrorValue | null;
  aggregation_error_exact_stack: ErrorValue;
  payload: MethodPayload[];
  payload_assumptions: string[];
  server_cost: ServerCost;
  notices: string[];
}

export interface SimulatorRoundResponse {
  inputs: SimulatorRoundInput;
  results: SimulatorRoundResult;
  assumptions: string[];
}

/* ---------- Glossary and paper facts (backend/app/modules/language_models/data/paper_facts.json) ---------- */

export type Level = 'beginner' | 'student' | 'researcher';

export interface GlossaryTerm {
  id: string;
  label: string;
  short: string;
  long: string;
  equation: string;
  equation_plain: string;
  reference: string;
  analogy: string | null;
  levels: Record<Level, string>;
}

export interface Table3Row {
  model: string;
  dataset: string;
  client_config: string;
  method: string;
  upload_mb: number;
  download_mb: number;
}

export interface Table4Row {
  model: string;
  dataset: string;
  client_config: string;
  method: string;
  server_flops: number;
}

export interface PaperResponse {
  provenance: 'paper_reported';
  paper: { title: string; authors: string[]; venue: string; code_url: string };
  reported_results: {
    table3: {
      reference: string;
      caption: string;
      setup: string;
      rows: Table3Row[];
    };
    table4: {
      reference: string;
      caption: string;
      setup: string;
      rows: Table4Row[];
    };
    [table: string]: unknown;
  };
}
