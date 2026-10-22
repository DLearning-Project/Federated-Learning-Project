import { apiGet, apiPost } from '../../services/api';
import type { DeviceProfilesResponse, HardwareEstimateInput, HardwareEstimateResponse } from './types';

export function getDeviceProfiles(_: void, signal?: AbortSignal) {
  return apiGet<DeviceProfilesResponse>('/hardware/device-profiles', signal);
}

export function estimateResources(input: HardwareEstimateInput, signal?: AbortSignal) {
  return apiPost<HardwareEstimateResponse>('/hardware/estimate', input, signal);
}
