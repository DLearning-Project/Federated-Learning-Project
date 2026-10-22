import { apiPost } from '../../services/api';
import type { LoraCalcInput, LoraCalcResponse } from './types';

export function calculateLora(input: LoraCalcInput, signal?: AbortSignal) {
  return apiPost<LoraCalcResponse>('/language-models/calculate', input, signal);
}
