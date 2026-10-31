import { apiGet, apiPost } from '../../services/api';
import type { ClipList, FedAvgInput, FedAvgResponse, SpectrogramInput, SpectrogramResponse } from './types';

export function aggregateFedAvg(input: FedAvgInput, signal?: AbortSignal) {
  return apiPost<FedAvgResponse>('/acoustics/aggregate', input, signal);
}

export function listClips(_input: void, signal?: AbortSignal) {
  return apiGet<ClipList>('/acoustics/clips', signal);
}

export function getSpectrogram(input: SpectrogramInput, signal?: AbortSignal) {
  return apiPost<SpectrogramResponse>('/acoustics/spectrogram', input, signal);
}
