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
