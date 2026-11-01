import { apiGet, apiPost } from '../../services/api';
import type { GlossaryTerm, LoraCalcInput, LoraCalcResponse, PaperResponse, SimulatorRoundInput, SimulatorRoundResponse } from './types';

export function calculateLora(input: LoraCalcInput, signal?: AbortSignal) {
  return apiPost<LoraCalcResponse>('/language-models/calculate', input, signal);
}

export function simulateRound(input: SimulatorRoundInput, signal?: AbortSignal) {
  return apiPost<SimulatorRoundResponse>('/language-models/simulator/round', input, signal);
}

export function fetchGlossary(signal?: AbortSignal) {
  return apiGet<GlossaryTerm[]>('/language-models/glossary', signal);
}

export function fetchPaper(signal?: AbortSignal) {
  return apiGet<PaperResponse>('/language-models/paper', signal);
}
