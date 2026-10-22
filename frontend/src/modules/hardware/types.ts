import type { CalcEnvelope } from '../../services/api';

/** Mirrors backend/app/modules/hardware/schemas.py — keep field names in sync. */
export interface DeviceProfile {
  id: string;
  name: string;
  description: string;
  seconds_per_epoch: number;
  train_power_w: number;
  radio_power_w: number;
  typical_bandwidth_mbps: number;
}

export interface DeviceProfilesResponse {
  mode: string;
  note: string;
  profiles: DeviceProfile[];
}

export interface HardwareEstimateInput {
  profile_id: string;
  update_mb: number;
  bandwidth_mbps: number;
  local_epochs: number;
  energy_budget_wh: number;
  time_budget_s: number | null;
}

export interface HardwareEstimateResult {
  profile_name: string;
  upload_time_s: number;
  download_time_s: number;
  communication_time_s: number;
  training_time_s: number;
  total_time_s: number;
  training_energy_wh: number;
  communication_energy_wh: number;
  total_energy_wh: number;
  energy_budget_used: number;
  meets_energy_budget: boolean;
  time_budget_used: number | null;
  meets_time_budget: boolean | null;
}

export type HardwareEstimateResponse = CalcEnvelope<HardwareEstimateInput, HardwareEstimateResult>;

export interface HardwareForm {
  profile_id: string;
  update_mb: string;
  bandwidth_mbps: string;
  local_epochs: string;
  energy_budget_wh: string;
  time_budget_s: string;
}
