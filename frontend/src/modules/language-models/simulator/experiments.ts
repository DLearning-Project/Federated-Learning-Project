import { formatBytes, formatPercent } from '../../../components/format';
import { byLevel } from '../lab/levels';
import type { Level, MethodPayload, SimulatorRoundResponse } from '../types';
import type { Settings, Tab } from './settings';

/* Helpers shared by the Simulator page and its experiments. */

export const pct = (x: number | null | undefined) => (x == null ? 'n/a' : x < 1e-9 ? '0%' : x < 0.001 ? '< 0.1%' : formatPercent(x));
export const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);
export const payloadOf = (r: SimulatorRoundResponse | undefined, m: MethodPayload['method']) => r?.results.payload.find((p) => p.method === m);
export const classesHeld = (row: number[]) => row.filter((v) => v > 0).length;

interface Wording {
  title: string;
  question: string;
  options: Record<string, string>;
}

/**
 * Predict, then run. Each experiment changes the settings, waits for the live
 * result, and checks the learner's prediction against it (never a stored answer).
 */
export interface Experiment {
  id: string;
  wording: { beginner: Wording; student: Wording; researcher?: Wording };
  optionIds: string[];
  tab: Tab;
  apply: (s: Settings) => Settings;
  verdict: (before: SimulatorRoundResponse, after: SimulatorRoundResponse, level: Level) => { correct: string; explanation: string };
}

export const EXPERIMENTS: Experiment[] = [
  {
    id: 'tau',
    optionIds: ['down', 'up'],
    wording: {
      beginner: {
        title: 'Send less',
        question: 'Move the slider so FLoRIST keeps 70% of the detail instead of 90%. What happens to each client’s download?',
        options: {
          down: 'It gets smaller or stays the same',
          up: 'It gets bigger',
        },
      },
      student: {
        title: 'Lower the energy threshold',
        question: 'If $\\tau$ drops from 0.90 to 0.70, what happens to the global rank $p$ that FLoRIST sends?',
        options: { down: '$p$ shrinks or stays the same', up: '$p$ grows' },
      },
      researcher: {
        title: '$\\tau$ sweep: $0.90 \\to 0.70$',
        question: 'Given $${p = \\min\\Bigl\\{ j : \\frac{\\sum_{i \\le j} \\sigma_i^2}{\\sum_i \\sigma_i^2} \\ge \\tau \\Bigr\\}},$$ lowering $\\tau$ to 0.70 makes $p$…',
        options: { down: 'non-increasing', up: 'strictly larger' },
      },
    },
    tab: 'spectrum',
    apply: (s) => ({ ...s, tau: 0.7 }),
    verdict: (b, a, level) => {
      const correct = a.results.p <= b.results.p ? 'down' : 'up';
      const same = a.results.p === b.results.p;
      const fb = payloadOf(b, 'florist')!.download_per_client_bytes;
      const fa = payloadOf(a, 'florist')!.download_per_client_bytes;
      return {
        correct,
        explanation: byLevel(level, {
          beginner: `FLoRIST kept ${b.results.p} piece(s) before and ${a.results.p} now, so the download went from ${formatBytes(fb)} to ${formatBytes(fa)}.${
            same ? ' The same pieces already held 70% of the detail.' : ''
          }`,
          student: String.raw`$p$ went from ${b.results.p} to ${a.results.p}. A lower $\tau$ needs less of the energy, so FLoRIST can stop at fewer singular values${
            same ? '; here the same components already reach 70%.' : '.'
          }`,
          researcher: String.raw`$p: ${b.results.p} \to ${a.results.p}$. The cumulative energy is non-decreasing in $j$, so the smallest $j$ meeting a lower $\tau$ cannot be larger.`,
        }),
      };
    },
  },
  {
    id: 'download',
    optionIds: ['stack', 'florist', 'same'],
    wording: {
      beginner: {
        title: 'Who sends more?',
        question: 'Which is bigger to download: every client’s add-on stapled together, or FLoRIST’s compressed version?',
        options: {
          stack: 'Stapled together',
          florist: 'FLoRIST’s version',
          same: 'The same size',
        },
      },
      student: {
        title: 'Who downloads more?',
        question: 'Which aggregate is larger for each client to download: exact stacking or FLoRIST?',
        options: {
          stack: 'Exact stacking',
          florist: 'FLoRIST',
          same: 'They are the same size',
        },
      },
      researcher: {
        title: 'Download payload',
        question: 'Compare per-client download: exact stacking (rank $\\sum_k r_k$) vs FLoRIST (rank $p$).',
        options: {
          stack: '$\\sum_k r_k\\,(m+n) > p\\,(m+n)$',
          florist: '$p\\,(m+n) > \\sum_k r_k\\,(m+n)$',
          same: 'Equal',
        },
      },
    },
    tab: 'download',
    apply: (s) => s,
    verdict: (_b, a, level) => {
      const st = payloadOf(a, 'exact_stack')!;
      const fl = payloadOf(a, 'florist')!;
      const x = st.download_per_client_bytes;
      const y = fl.download_per_client_bytes;
      return {
        correct: x > y ? 'stack' : x < y ? 'florist' : 'same',
        explanation: byLevel(level, {
          beginner: `Stapled together: ${formatBytes(x)}. FLoRIST: ${formatBytes(y)}.`,
          student: String.raw`Exact stacking sends rank $\sum_k r_k = ${st.download_rank}$ (${formatBytes(x)}); FLoRIST sends rank $p = ${fl.download_rank}$ (${formatBytes(y)}).`,
          researcher: String.raw`$\sum_k r_k = ${st.download_rank}$, $p = ${fl.download_rank}$, $m + n = ${a.results.m + a.results.n}$: ${formatBytes(x)} vs ${formatBytes(y)}. Always $p \le \min(m, n, \sum_k r_k)$.`,
        }),
      };
    },
  },
  {
    id: 'fedit',
    optionIds: ['yes', 'no'],
    wording: {
      beginner: {
        title: 'Is simple averaging enough?',
        question:
          'Each add-on has two parts. If the server averages each part on its own and then multiplies them, does it get the clients’ true combined update?',
        options: { yes: 'Yes, exactly', no: 'No, something gets mixed up' },
      },
      student: {
        title: 'Is FedIT exact?',
        question: 'Every client has the same rank. Does FedIT (average B, average A, then multiply) give exactly the average of the clients’ updates?',
        options: { yes: 'Yes, exactly', no: 'No, there is an error' },
      },
      researcher: {
        title: 'FedIT cross terms',
        question: 'Homogeneous ranks. For these clients, is $$\\Bigl(\\sum_k w_k B_k\\Bigr)\\Bigl(\\sum_k w_k A_k\\Bigr) = \\sum_k w_k B_k A_k\\,?$$',
        options: { yes: 'Equal', no: 'Not equal' },
      },
    },
    tab: 'spectrum',
    apply: (s) => ({
      ...s,
      heterogeneous: false,
      show_fedit: true,
      beginner_fedit: true,
    }),
    verdict: (_b, a, level) => {
      const e = a.results.aggregation_error_fedit;
      const correct = e && e.absolute > 1e-9 ? 'no' : 'yes';
      if (correct === 'yes')
        return {
          correct,
          explanation: 'This round the clients happen to agree, so FedIT is exact.',
        };
      return {
        correct,
        explanation: byLevel(level, {
          beginner: `It is off by ${pct(e!.relative)}. Multiplying the averages mixes one client’s first part with another client’s second part.`,
          student: `FedIT is off by ${pct(e!.relative)} of the true update this round. Multiplying the averages adds cross terms $B_i A_j$ between different clients.`,
          researcher: String.raw`$\lVert \Delta W - \bar{B}\bar{A} \rVert_F = ${e!.absolute.toPrecision(3)}$ (relative ${pct(e!.relative)}). The product adds $\sum_{i \neq j} w_i w_j B_i A_j$ and scales the diagonal by $w_k^2$ instead of $w_k$.`,
        }),
      };
    },
  },
  {
    id: 'alpha',
    optionIds: ['most', 'few'],
    wording: {
      beginner: {
        title: 'Very uneven data',
        question: 'Make the data very uneven. How many of the 16 topics does a typical hospital have?',
        options: { most: 'Most of the topics', few: 'Only a few topics' },
      },
      student: {
        title: 'Very non-IID data',
        question: 'Set $\\alpha$ to 0.05. How many of the 16 classes does a typical sampled client hold?',
        options: { most: 'Most of the classes', few: 'Only a few classes' },
      },
      researcher: {
        title: 'Dirichlet $\\alpha = 0.05$',
        question: 'Per class, $(q_1, \\dots, q_K) \\sim \\mathrm{Dir}(0.05)$. Median number of classes held by a sampled client:',
        options: { most: '$> 8$ of 16', few: '$\\le 8$ of 16' },
      },
    },
    tab: 'data',
    apply: (s) => ({ ...s, alpha: 0.05 }),
    verdict: (_b, a, level) => {
      const held = a.results.sampled_clients.map((c) => classesHeld(a.results.class_counts[c])).sort((x, y) => x - y);
      const median = held[Math.floor(held.length / 2)];
      return {
        correct: median <= 8 ? 'few' : 'most',
        explanation: byLevel(level, {
          beginner: `This round’s hospitals have ${held.join(', ')} of the 16 topics. When data is very uneven, each hospital sees only a few topics.`,
          student: String.raw`The sampled clients hold ${held.join(', ')} of 16 classes. A small $\alpha$ makes each client's data concentrate on a few classes.`,
          researcher: String.raw`Classes held by sampled clients: [${held.join(', ')}], median ${median}. Small $\alpha$ puts most of each class’s mass on few clients.`,
        }),
      };
    },
  },
];
