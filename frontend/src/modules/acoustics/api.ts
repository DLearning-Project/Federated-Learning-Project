import { apiPost } from '../../services/api';
import type { FedAvgInput, FedAvgResponse } from './types';

export function aggregateFedAvg(input: FedAvgInput, signal?: AbortSignal) {
  return apiPost<FedAvgResponse>('/acoustics/aggregate', input, signal);
}
