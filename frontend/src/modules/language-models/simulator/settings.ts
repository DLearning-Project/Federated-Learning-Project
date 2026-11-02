import type { SimulatorRoundInput } from '../types';

export interface Settings {
  total_clients: number;
  clients_per_round: number;
  rounds: number;
  alpha: number;
  heterogeneous: boolean;
  tau: number;
  seed: number;
  bytes_per_param: 1 | 2 | 4;
  show_fedit: boolean;
  show_florist: boolean;
  /** Beginners start without the FedIT comparison; an experiment or checkbox turns it on. */
  beginner_fedit: boolean;
}

export const DEFAULTS: Settings = {
  total_clients: 10,
  clients_per_round: 3,
  rounds: 10,
  alpha: 0.5,
  heterogeneous: false,
  tau: 0.9,
  seed: 42,
  bytes_per_param: 4,
  show_fedit: true,
  show_florist: true,
  beginner_fedit: false,
};

export type Tab = 'accuracy' | 'spectrum' | 'download' | 'data';

export function toInput(s: Settings, round: number): SimulatorRoundInput {
  return {
    total_clients: s.total_clients,
    clients_per_round: Math.min(s.clients_per_round, s.total_clients),
    round,
    alpha: s.alpha,
    heterogeneous: s.heterogeneous,
    tau: s.tau,
    seed: s.seed,
    bytes_per_param: s.bytes_per_param,
  };
}

export function sameInput(a: SimulatorRoundInput, b: SimulatorRoundInput): boolean {
  return (Object.keys(b) as (keyof SimulatorRoundInput)[]).every((k) => a[k] === b[k]);
}
