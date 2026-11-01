import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { useModuleState } from '../../../app/AppStateContext';
import { formatBytes, formatInt, formatNumber, formatPercent } from '../../../components/format';
import { AssumptionsList } from '../../../components/Results';
import { ApiErrorState, EmptyState, LoadingState, StatusMessage } from '../../../components/StatusMessage';
import { useApiAction, type ActionState } from '../../../hooks/useApiAction';
import { fetchPaper, simulateRound } from '../api';
import { useLab } from '../lab/LabContext';
import { Term } from '../lab/Term';
import { ClassMixGrid, DownloadBars, SpectrumChart } from '../simulator/charts';
import type { Level, MethodPayload, PaperResponse, SimulatorRoundInput, SimulatorRoundResponse } from '../types';

type PaperAction = { state: ActionState<PaperResponse>; data?: PaperResponse; run: (input: null) => Promise<void> };

/* ---------- Settings ---------- */

interface Settings {
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
}

const DEFAULTS: Settings = {
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
};

type Tab = 'accuracy' | 'spectrum' | 'download' | 'data';
const TABS: { id: Tab; label: string }[] = [
  { id: 'accuracy', label: 'Accuracy' },
  { id: 'spectrum', label: 'Singular values (τ)' },
  { id: 'download', label: 'Download cost' },
  { id: 'data', label: 'Client data' },
];

const STEP_MS = 700;

function toInput(s: Settings, round: number): SimulatorRoundInput {
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

function sameInput(a: SimulatorRoundInput, b: SimulatorRoundInput): boolean {
  return (Object.keys(b) as (keyof SimulatorRoundInput)[]).every((k) => a[k] === b[k]);
}

const pct = (x: number | null | undefined) => (x == null ? 'n/a' : x < 1e-9 ? '0%' : x < 0.001 ? '< 0.1%' : formatPercent(x));
const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);
const payloadOf = (r: SimulatorRoundResponse | undefined, m: MethodPayload['method']) => r?.results.payload.find((p) => p.method === m);
const classesHeld = (row: number[]) => row.filter((v) => v > 0).length;

/* ---------- Experiments: predict, then run (all checked against live results) ---------- */

interface Experiment {
  id: string;
  title: string;
  question: string;
  options: { id: string; label: string }[];
  tab: Tab;
  apply: (s: Settings) => Settings;
  verdict: (before: SimulatorRoundResponse, after: SimulatorRoundResponse) => { correct: string; explanation: string };
}

const EXPERIMENTS: Experiment[] = [
  {
    id: 'tau',
    title: 'Lower the energy threshold',
    question: 'If τ drops from 0.90 to 0.70, what happens to the global rank p that FLoRIST sends?',
    options: [
      { id: 'down', label: 'p shrinks or stays the same' },
      { id: 'up', label: 'p grows' },
    ],
    tab: 'spectrum',
    apply: (s) => ({ ...s, tau: 0.7 }),
    verdict: (b, a) => ({
      correct: a.results.p <= b.results.p ? 'down' : 'up',
      explanation:
        `p went from ${b.results.p} to ${a.results.p}. A lower τ needs less of the energy, so FLoRIST can stop at fewer singular values` +
        (a.results.p === b.results.p ? '; here the same components already reach 70%.' : '.'),
    }),
  },
  {
    id: 'download',
    title: 'Who downloads more?',
    question: 'Which aggregate is larger for each client to download: exact stacking or FLoRIST?',
    options: [
      { id: 'stack', label: 'Exact stacking' },
      { id: 'florist', label: 'FLoRIST' },
      { id: 'same', label: 'They are the same size' },
    ],
    tab: 'download',
    apply: (s) => s,
    verdict: (_b, a) => {
      const st = payloadOf(a, 'exact_stack')!;
      const fl = payloadOf(a, 'florist')!;
      const correct = st.download_per_client_bytes > fl.download_per_client_bytes ? 'stack' : st.download_per_client_bytes < fl.download_per_client_bytes ? 'florist' : 'same';
      return {
        correct,
        explanation: `Exact stacking sends rank Σr_k = ${st.download_rank} (${formatBytes(st.download_per_client_bytes)}); FLoRIST sends rank p = ${fl.download_rank} (${formatBytes(
          fl.download_per_client_bytes,
        )}).`,
      };
    },
  },
  {
    id: 'fedit',
    title: 'Is FedIT exact?',
    question: 'Every client has the same rank. Does FedIT (average B, average A, then multiply) give exactly the average of the clients’ updates?',
    options: [
      { id: 'yes', label: 'Yes, exactly' },
      { id: 'no', label: 'No, there is an error' },
    ],
    tab: 'spectrum',
    apply: (s) => ({ ...s, heterogeneous: false, show_fedit: true }),
    verdict: (_b, a) => {
      const e = a.results.aggregation_error_fedit;
      const correct = e && e.absolute > 1e-9 ? 'no' : 'yes';
      return {
        correct,
        explanation:
          correct === 'no'
            ? `FedIT is off by ${pct(e!.relative)} of the true update this round. Multiplying the averages adds cross terms B_i A_j between different clients.`
            : 'This round the clients happen to agree, so FedIT is exact.',
      };
    },
  },
  {
    id: 'alpha',
    title: 'Very non-IID data',
    question: 'Set α to 0.05. How many of the 16 classes does a typical sampled client hold?',
    options: [
      { id: 'most', label: 'Most of the classes' },
      { id: 'few', label: 'Only a few classes' },
    ],
    tab: 'data',
    apply: (s) => ({ ...s, alpha: 0.05 }),
    verdict: (_b, a) => {
      const held = a.results.sampled_clients.map((c) => classesHeld(a.results.class_counts[c])).sort((x, y) => x - y);
      const median = held[Math.floor(held.length / 2)];
      return {
        correct: median <= 8 ? 'few' : 'most',
        explanation: `The sampled clients hold ${held.join(', ')} of 16 classes. A small α makes each client's data concentrate on a few classes.`,
      };
    },
  },
];

interface Prediction {
  experiment: string;
  choice: string;
  before: SimulatorRoundResponse;
  target: SimulatorRoundInput;
}

/* ---------- Page ---------- */

export default function SimulatorPage() {
  const { level } = useLab();
  const lvl: Level = level ?? 'student';
  const [settings, setSettings] = useModuleState<Settings>('lm.sim.settings', DEFAULTS);
  const [round, setRound] = useModuleState('lm.sim.round', 1);
  const [tab, setTab] = useModuleState<Tab>('lm.sim.tab', 'spectrum');
  const [showMath, setShowMath] = useModuleState('lm.sim.math', false);
  const [prediction, setPrediction] = useModuleState<Prediction | null>('lm.sim.prediction', null);
  const [activeExp, setActiveExp] = useModuleState('lm.sim.exp', EXPERIMENTS[0].id);
  const [playing, setPlaying] = useState(false);

  const sim = useApiAction('lm.sim', simulateRound);
  const paper = useApiAction('lm.paper', (_: null, signal: AbortSignal) => fetchPaper(signal));

  const input = useMemo(() => toInput(settings, round), [settings, round]);
  const inputKey = JSON.stringify(input);
  const data = sim.data;
  const current = data && sameInput(data.inputs, input) ? data : undefined;

  // Fetch the current round whenever settings change (debounced while dragging sliders).
  useEffect(() => {
    if (current && sim.state.status === 'success') return;
    const t = setTimeout(() => sim.run(input), data ? 200 : 0);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inputKey]);

  // Playback: advance one round after each response arrives.
  useEffect(() => {
    if (!playing) return;
    if (sim.state.status === 'error') return setPlaying(false);
    if (sim.state.status !== 'success' || !current) return;
    const t = setTimeout(() => {
      if (round >= settings.rounds) setPlaying(false);
      else setRound(round + 1);
    }, STEP_MS);
    return () => clearTimeout(t);
  }, [playing, sim.state, current, round, settings.rounds, setRound]);

  // Load paper-reported Table 3 the first time the download tab opens.
  useEffect(() => {
    if (tab === 'download' && paper.state.status === 'idle') paper.run(null);
  }, [tab, paper]);

  const update = (patch: Partial<Settings>) => {
    setPlaying(false);
    setSettings((s) => {
      const next = { ...s, ...patch };
      if (next.clients_per_round > next.total_clients) next.clients_per_round = next.total_clients;
      return next;
    });
    if (patch.rounds !== undefined && round > patch.rounds) setRound(patch.rounds);
  };

  const runPlayback = () => {
    if (round >= settings.rounds) setRound(1);
    setPlaying(true);
  };
  const step = () => {
    setPlaying(false);
    setRound(round >= settings.rounds ? 1 : round + 1);
  };
  const reset = () => {
    setPlaying(false);
    setRound(1);
    setSettings(DEFAULTS);
    setPrediction(null);
  };

  const predict = (exp: Experiment, choice: string) => {
    if (!current) return;
    const next = exp.apply(settings);
    setPlaying(false);
    setSettings(next);
    setTab(exp.tab);
    setPrediction({ experiment: exp.id, choice, before: current, target: toInput(next, round) });
  };

  const mathVisible = lvl === 'researcher' || (lvl === 'student' && showMath);
  const r = current?.results;

  return (
    <div className="page lm-sim">
      <header className="lm-sim-head">
        <div>
          <span className="eyebrow">Page 4 of 4 · Language models</span>
          <h1>FLoRIST Simulator</h1>
          <p className="muted">
            Sample clients, give them LoRA adapters, and aggregate them three ways. Every number here is computed live by the backend on small synthetic
            adapters.
          </p>
        </div>
        <span className="pill">
          <span className="pill-dot ok" /> Computed live · synthetic adapters
        </span>
      </header>

      {lvl === 'beginner' && (
        <div className="card inset lm-scenario">
          <strong>The scenario</strong>
          <p>
            Imagine {settings.total_clients} hospitals that each want a better medical chatbot but cannot share patient notes. Each hospital trains a small
            add-on (a LoRA adapter) on its own notes. A server combines the add-ons and sends one back to everybody. This page shows how big that download is,
            and how much of the hospitals’ work survives the combining.
          </p>
        </div>
      )}

      <div className="lm-sim-grid">
        <Controls settings={settings} update={update} level={lvl} />

        <div className="lm-sim-main">
          <Kpis result={current} settings={settings} />

          <section className="card lm-results" aria-label="Results">
            <div className="lm-results-head">
              <div className="segmented" role="tablist" aria-label="Result views">
                {TABS.map((t) => (
                  <button key={t.id} type="button" role="tab" aria-selected={tab === t.id} onClick={() => setTab(t.id)}>
                    {t.label}
                  </button>
                ))}
              </div>
              {sim.state.status === 'loading' && <span className="small muted">Updating…</span>}
            </div>

            {sim.state.status === 'error' && <ApiErrorState error={sim.state.error} onRetry={() => sim.run(input)} />}
            {!data && sim.state.status === 'loading' && <LoadingState label="Simulating round…" />}

            {tab === 'accuracy' && (
              <EmptyState>
                <strong>Accuracy is available after the toy experiment (build step 5).</strong> The adapters on this page are synthetic, so there is no model
                to score yet. We do not draw made-up curves. For accuracy measured in the paper, see Table 2 of the FLoRIST paper.
              </EmptyState>
            )}

            {r && tab !== 'accuracy' && (
              <div className={sim.state.status === 'loading' ? 'lm-stale' : undefined}>
                {tab === 'spectrum' && (
                  <>
                    <SpectrumChart values={r.singular_values} energy={r.cumulative_energy} p={r.p} tau={current!.inputs.tau} />
                    <ul className="legend">
                      <li>
                        <span className="swatch" style={{ background: 'var(--color-orange)' }} /> kept by FLoRIST
                      </li>
                      <li>
                        <span className="swatch" style={{ background: 'var(--color-chart-light)', border: '1px solid var(--color-chart-gray)' }} /> discarded
                      </li>
                      <li>
                        <span className="swatch" style={{ background: 'var(--color-ink)', height: 3 }} /> cumulative energy
                      </li>
                      <li>
                        <span className="swatch lm-swatch-dash" /> threshold τ
                      </li>
                    </ul>
                  </>
                )}
                {tab === 'download' && (
                  <DownloadTab data={current!} settings={settings} level={lvl} paper={paper} />
                )}
                {tab === 'data' && <ClassMixGrid counts={r.class_counts} sampled={r.sampled_clients} ranks={r.all_client_ranks} />}

                <p className="lm-reading">
                  <strong>Reading this:</strong> {reading(tab, current!, lvl)}
                </p>

                {lvl === 'student' && (
                  <button type="button" className="btn btn-ghost lm-link-btn" aria-expanded={showMath} onClick={() => setShowMath(!showMath)}>
                    {showMath ? 'Hide the math' : 'Show the math'}
                  </button>
                )}
                {mathVisible && <MathBox tab={tab} data={current!} />}
              </div>
            )}
          </section>

          <Playback round={round} rounds={settings.rounds} playing={playing} onRun={runPlayback} onPause={() => setPlaying(false)} onStep={step} onReset={reset} />

          {r && r.notices.length > 0 && (
            <StatusMessage tone="warning" title="Notes for this round">
              <ul className="lm-list">
                {r.notices.map((n) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            </StatusMessage>
          )}
          {current && <AssumptionsList title="Assumptions behind these numbers" assumptions={[...current.assumptions, ...current.results.payload_assumptions]} />}
        </div>

        <aside className="lm-sim-side" aria-label="Experiments">
          <Experiments
            data={data}
            current={current}
            prediction={prediction}
            activeExp={activeExp}
            setActiveExp={(id) => {
              setActiveExp(id);
              setPrediction(null);
            }}
            onPredict={predict}
          />
          {lvl === 'beginner' && (
            <div className="card inset lm-analogy">
              <div className="lm-kicker">Think of it like this</div>
              <p>
                Each hospital sends a summary of what it learned. Stacking staples all the summaries together: nothing is lost, but the packet keeps growing.
                FLoRIST rewrites them into one short summary that keeps the points most hospitals agree on (the tall bars).
              </p>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}

/* ---------- Controls ---------- */

function Group({ title, children, defaultOpen = true }: { title: string; children: ReactNode; defaultOpen?: boolean }) {
  return (
    <details className="lm-group" open={defaultOpen}>
      <summary>{title}</summary>
      <div className="lm-group-body">{children}</div>
    </details>
  );
}

function Slider({
  id,
  label,
  value,
  display,
  min,
  max,
  step,
  onChange,
  hint,
}: {
  id: string;
  label: ReactNode;
  value: number;
  display: string;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  hint?: ReactNode;
}) {
  return (
    <div className="lm-field">
      <div className="lm-field-head">
        {label}
        <output htmlFor={id} className="lm-value">
          {display}
        </output>
      </div>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
      {hint && <span className="small muted">{hint}</span>}
    </div>
  );
}

function Controls({ settings: s, update, level }: { settings: Settings; update: (p: Partial<Settings>) => void; level: Level }) {
  const logAlpha = Math.log10(s.alpha);
  return (
    <aside className="lm-sim-controls" aria-label="Simulation settings">
      <Group title="Federation">
        {level !== 'beginner' && (
          <Slider
            id="lm-total"
            label={
              <Term htmlFor="lm-total" definition="Every device or organisation that holds its own data and could take part in training.">
                Total clients
              </Term>
            }
            value={s.total_clients}
            display={String(s.total_clients)}
            min={2}
            max={50}
            step={1}
            onChange={(v) => update({ total_clients: v })}
          />
        )}
        <Slider
          id="lm-per-round"
          label={
            <Term htmlFor="lm-per-round" definition="How many clients the server picks at random to train in each round.">
              Clients per round
            </Term>
          }
          value={Math.min(s.clients_per_round, s.total_clients)}
          display={String(Math.min(s.clients_per_round, s.total_clients))}
          min={1}
          max={Math.min(10, s.total_clients)}
          step={1}
          onChange={(v) => update({ clients_per_round: v })}
        />
        <Slider
          id="lm-rounds"
          label={
            <Term htmlFor="lm-rounds" definition="One round: the server sends the model, clients train locally, and the server combines their updates.">
              Rounds
            </Term>
          }
          value={s.rounds}
          display={String(s.rounds)}
          min={5}
          max={30}
          step={1}
          onChange={(v) => update({ rounds: v })}
        />
      </Group>

      <Group title="Client data">
        <Slider
          id="lm-alpha"
          label={
            <Term htmlFor="lm-alpha" definition="Dirichlet α controls how uneven the clients' data is: small α gives each client only a few classes; large α gives everyone a similar mix.">
              Dirichlet α
            </Term>
          }
          value={logAlpha}
          display={formatNumber(s.alpha, s.alpha < 1 ? 2 : 1)}
          min={-2}
          max={3}
          step={0.05}
          onChange={(v) => update({ alpha: Number(Math.pow(10, v).toPrecision(2)) })}
          hint={
            <>
              More uneven ← → more similar · paper uses 0.5{' '}
              {s.alpha !== 0.5 && (
                <button type="button" className="btn btn-ghost lm-inline-btn" onClick={() => update({ alpha: 0.5 })}>
                  reset
                </button>
              )}
            </>
          }
        />
      </Group>

      <Group title="LoRA ranks">
        <div className="lm-field">
          <Term termId="heterogeneous_ranks">Client ranks</Term>
          <div className="segmented lm-full" role="group" aria-label="Client ranks">
            <button type="button" aria-pressed={!s.heterogeneous} onClick={() => update({ heterogeneous: false })}>
              Homogeneous
            </button>
            <button type="button" aria-pressed={s.heterogeneous} onClick={() => update({ heterogeneous: true })}>
              Heterogeneous
            </button>
          </div>
          <span className="small muted">{s.heterogeneous ? 'Mix of ranks 1, 2 and 4 (most clients small, like the paper).' : 'Every client uses rank 4.'}</span>
        </div>
      </Group>

      <Group title="Aggregation">
        <div className="lm-field" role="group" aria-label="Methods to compare">
          <span>Methods</span>
          <label className="lm-check">
            <input type="checkbox" checked={s.show_florist} onChange={(e) => update({ show_florist: e.target.checked })} />
            <Term termId="selected_rank">FLoRIST</Term>
          </label>
          <label className="lm-check">
            <input type="checkbox" checked={s.show_fedit} onChange={(e) => update({ show_fedit: e.target.checked })} />
            <Term termId="fedit">FedIT</Term>
          </label>
          <label className="lm-check">
            <input type="checkbox" checked disabled />
            <Term termId="weighted_stacking">Exact stacking</Term> <span className="small muted">(reference)</span>
          </label>
        </div>
        <Slider
          id="lm-tau"
          label={
            <Term termId="energy_threshold" htmlFor="lm-tau">
              Energy threshold τ
            </Term>
          }
          value={s.tau}
          display={s.tau.toFixed(2)}
          min={0.5}
          max={1}
          step={0.01}
          onChange={(v) => update({ tau: v })}
          hint="FLoRIST only · paper default 0.90"
        />
      </Group>

      {level !== 'beginner' && (
        <Group title="Run settings" defaultOpen={level === 'researcher'}>
          <div className="lm-field">
            <label htmlFor="lm-seed">Random seed</label>
            <input
              id="lm-seed"
              className="input compact"
              type="number"
              min={0}
              max={1000000}
              value={s.seed}
              onChange={(e) => {
                const v = Math.round(Number(e.target.value));
                if (Number.isFinite(v) && v >= 0 && v <= 1_000_000) update({ seed: v });
              }}
            />
          </div>
          <div className="lm-field">
            <span>Bytes per parameter</span>
            <div className="segmented lm-full" role="group" aria-label="Bytes per parameter">
              {([4, 2, 1] as const).map((b) => (
                <button key={b} type="button" aria-pressed={s.bytes_per_param === b} onClick={() => update({ bytes_per_param: b })}>
                  {b} B{b === 4 ? ' · FP32' : b === 2 ? ' · FP16' : ' · INT8'}
                </button>
              ))}
            </div>
          </div>
        </Group>
      )}
    </aside>
  );
}

/* ---------- KPIs ---------- */

function Kpi({ label, value, sub }: { label: ReactNode; value: ReactNode; sub: ReactNode }) {
  return (
    <div className="lm-kpi">
      <span className="small muted">{label}</span>
      <strong className="lm-kpi-value">{value}</strong>
      <span className="small muted">{sub}</span>
    </div>
  );
}

function Kpis({ result, settings }: { result: SimulatorRoundResponse | undefined; settings: Settings }) {
  const r = result?.results;
  const fl = payloadOf(result, 'florist');
  const st = payloadOf(result, 'exact_stack');
  const dash = '—';
  return (
    <div className="lm-kpis">
      <Kpi
        label={<Term termId="selected_rank">Global rank p</Term>}
        value={r ? r.p : dash}
        sub={r ? `max r_k = ${Math.max(...r.client_ranks)} · Σr_k = ${sum(r.client_ranks)}` : 'waiting for results'}
      />
      <Kpi
        label={<Term termId="communication_cost">Download per client</Term>}
        value={fl ? formatBytes(fl.download_per_client_bytes) : dash}
        sub={
          fl && st
            ? `FLoRIST vs ${formatBytes(st.download_per_client_bytes)} for exact stacking (${formatNumber(st.download_per_client_bytes / Math.max(fl.download_per_client_bytes, 1), 1)}× less)`
            : 'FLoRIST, per round'
        }
      />
      <Kpi
        label={<Term termId="eckart_young_bound">FLoRIST error</Term>}
        value={r ? pct(r.reconstruction_error.relative) : dash}
        sub={r ? `of the exact update · equals the Eq. 5 bound (${formatNumber(r.eq5_bound, 3)})` : 'relative to exact stacking'}
      />
      <Kpi
        label={<Term termId="cross_term">FedIT error</Term>}
        value={!settings.show_fedit ? 'hidden' : r ? (r.aggregation_error_fedit ? pct(r.aggregation_error_fedit.relative) : 'n/a') : dash}
        sub={!settings.show_fedit ? 'turn FedIT on to compare' : r && !r.aggregation_error_fedit ? 'mixed ranks: FedIT cannot average them' : 'of the exact update (cross terms)'}
      />
    </div>
  );
}

/* ---------- Download tab ---------- */

function DownloadTab({
  data,
  settings,
  level,
  paper,
}: {
  data: SimulatorRoundResponse;
  settings: Settings;
  level: Level;
  paper: PaperAction;
}) {
  const r = data.results;
  const shown = r.payload.filter((p) => (p.method === 'fedit' ? settings.show_fedit : p.method === 'florist' ? settings.show_florist : true));
  const table3 = paper.data?.reported_results.table3;
  return (
    <div className="lm-stack">
      <DownloadBars payload={shown} />
      {level !== 'beginner' && (
        <p className="small muted">
          Server work this round (estimated operations): FLoRIST {formatInt(r.server_cost.florist_ops)} vs SVD of the full ΔW {formatInt(r.server_cost.direct_svd_ops)}
          {r.server_cost.fedit_ops != null ? ` vs FedIT ${formatInt(r.server_cost.fedit_ops)}` : ''}.
        </p>
      )}
      <div className="lm-paper">
        <div className="lm-paper-head">
          <span className="badge">Paper-reported</span>
          <strong>{table3?.reference ?? 'Table 3'}</strong>
          <span className="small muted">Not computed here: these are the paper’s numbers, for scale.</span>
        </div>
        {paper.state.status === 'loading' && <LoadingState label="Loading paper values…" />}
        {paper.state.status === 'error' && <ApiErrorState error={paper.state.error} onRetry={() => paper.run(null)} />}
        {table3 && (
          <>
            <p className="small">{table3.caption}</p>
            <div className="lm-table-wrap">
              <table className="lm-table">
                <thead>
                  <tr>
                    <th scope="col">Method</th>
                    <th scope="col">Upload (MB)</th>
                    <th scope="col">Download (MB)</th>
                  </tr>
                </thead>
                <tbody>
                  {table3.rows.map((row) => (
                    <tr key={row.method}>
                      <th scope="row">{row.method}</th>
                      <td>{formatNumber(row.upload_mb, 2)}</td>
                      <td>{formatNumber(row.download_mb, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {level !== 'beginner' && <p className="small muted">{table3.setup}</p>}
          </>
        )}
      </div>
    </div>
  );
}

/* ---------- Reading captions and math ---------- */

function reading(tab: Tab, d: SimulatorRoundResponse, level: Level): string {
  const r = d.results;
  const sigma = sum(r.client_ranks);
  if (tab === 'spectrum') {
    const kept = r.p > 0 ? r.cumulative_energy[r.p - 1] : 0;
    if (level === 'beginner')
      return `Each bar is one direction in which the clients changed the model; tall bars matter most. FLoRIST keeps the ${r.p} orange bar${r.p === 1 ? '' : 's'}, which hold ${formatPercent(
        kept,
      )} of the total, and drops the rest.`;
    return `The ${r.singular_values.length} bars are the singular values of the weighted average update ΔW. The first ${r.p} hold ${formatPercent(
      kept,
    )} of the energy, which meets τ = ${d.inputs.tau.toFixed(2)}, so FLoRIST sends rank ${r.p} instead of Σr_k = ${sigma}.`;
  }
  if (tab === 'download') {
    const fl = payloadOf(d, 'florist')!;
    const st = payloadOf(d, 'exact_stack')!;
    const fe = payloadOf(d, 'fedit');
    return `Each client downloads ${formatBytes(fl.download_per_client_bytes)} with FLoRIST (rank ${fl.download_rank}) and ${formatBytes(
      st.download_per_client_bytes,
    )} with exact stacking (rank ${st.download_rank}).${fe ? ` FedIT sends ${formatBytes(fe.download_per_client_bytes)}, but its average is not exact.` : ''}`;
  }
  const held = r.sampled_clients.map((c) => classesHeld(r.class_counts[c]));
  const avg = held.length ? sum(held) / held.length : 0;
  return `Rows are clients and columns are the 16 classes; darker cells hold more examples. With α = ${formatNumber(d.inputs.alpha, 2)}, this round's ${
    r.sampled_clients.length
  } sampled clients (●) hold ${formatNumber(avg, 1)} classes on average, and they are weighted by how many examples they have.`;
}

function MathBox({ tab, data }: { tab: Tab; data: SimulatorRoundResponse }) {
  const r = data.results;
  const { m, n } = r;
  const sigma = sum(r.client_ranks);
  const w = r.weights.map((x) => x.toFixed(3)).join(', ');
  const lines: string[] =
    tab === 'spectrum'
      ? [
          `ΔW = Σ_k w_k B_k A_k = B_stack · A_stack     (m × n = ${m} × ${n}; w = [${w}])`,
          'B_stack = U_B S_B V_Bᵀ,  A_stack = U_A S_A V_Aᵀ,  P = S_B (V_Bᵀ U_A) S_A = U_P S_P V_Pᵀ',
          `p = smallest j with Σ_{i≤j} σ_i² / Σ_i σ_i² ≥ τ = ${data.inputs.tau.toFixed(2)}   →   p = ${r.p}`,
          `‖ΔW − B_g A_g‖_F = √(Σ_{i>p} σ_i²) = ${formatNumber(r.eq5_bound, 4)}   (Eckart–Young, Eq. 5)`,
          r.aggregation_error_fedit
            ? `FedIT: ‖ΔW − (Σ w_k B_k)(Σ w_k A_k)‖_F = ${formatNumber(r.aggregation_error_fedit.absolute, 4)}`
            : 'FedIT: not defined when client ranks differ',
        ]
      : tab === 'download'
        ? [
            `Download per client = rank × (m + n) × bytes = rank × ${m + n} × ${data.inputs.bytes_per_param} B`,
            `Exact stacking: rank Σr_k = ${sigma};  FLoRIST: rank p = ${r.p};  FedIT: rank r (equal ranks only)`,
            `Upload per client k = r_k × (m + n) × bytes`,
          ]
        : [
            'For each class c: (q_1, …, q_K) ~ Dirichlet(α); client k gets about q_k × 60 examples of class c',
            `w_k = n_k / Σ_j n_j   →   sample counts [${r.sample_counts.join(', ')}], weights [${w}]`,
          ];
  return (
    <div className="lm-equation lm-math" aria-label="The math">
      {lines.map((l) => (
        <div key={l}>{l}</div>
      ))}
    </div>
  );
}

/* ---------- Playback ---------- */

function Playback({
  round,
  rounds,
  playing,
  onRun,
  onPause,
  onStep,
  onReset,
}: {
  round: number;
  rounds: number;
  playing: boolean;
  onRun: () => void;
  onPause: () => void;
  onStep: () => void;
  onReset: () => void;
}) {
  return (
    <div className="card lm-playback">
      <div className="lm-row">
        {playing ? (
          <button type="button" className="btn btn-primary" onClick={onPause}>
            Pause
          </button>
        ) : (
          <button type="button" className="btn btn-primary" onClick={onRun}>
            Run simulation
          </button>
        )}
        <button type="button" className="btn btn-secondary" onClick={onStep}>
          Step
        </button>
        <button type="button" className="btn btn-ghost" onClick={onReset}>
          Reset
        </button>
      </div>
      <div className="lm-progress">
        <span className="small" aria-live="polite">
          Round {round} / {rounds}
        </span>
        <span className="lm-progress-track" aria-hidden>
          <span className="lm-progress-fill" style={{ width: `${(round / rounds) * 100}%` }} />
        </span>
      </div>
      <p className="small muted lm-full-row">
        Each round samples a new set of clients and aggregates their adapters. The adapters are synthetic, so nothing is trained between rounds yet.
      </p>
    </div>
  );
}

/* ---------- Experiments ---------- */

function Experiments({
  data,
  current,
  prediction,
  activeExp,
  setActiveExp,
  onPredict,
}: {
  data: SimulatorRoundResponse | undefined;
  current: SimulatorRoundResponse | undefined;
  prediction: Prediction | null;
  activeExp: string;
  setActiveExp: (id: string) => void;
  onPredict: (exp: Experiment, choice: string) => void;
}) {
  const exp = EXPERIMENTS.find((e) => e.id === activeExp) ?? EXPERIMENTS[0];
  const mine = prediction?.experiment === exp.id ? prediction : null;
  const after = mine && data && sameInput(data.inputs, mine.target) ? data : undefined;
  const verdict = mine && after ? exp.verdict(mine.before, after) : null;

  return (
    <>
      <div className="card lm-predict">
        <div className="lm-kicker">Predict, then run</div>
        <strong>{exp.title}</strong>
        <p>{exp.question}</p>
        <div className="lm-options" role="radiogroup" aria-label="Your prediction">
          {exp.options.map((o) => (
            <button
              key={o.id}
              type="button"
              role="radio"
              className="choice"
              aria-checked={mine?.choice === o.id}
              disabled={!current || !!mine}
              onClick={() => onPredict(exp, o.id)}
            >
              {o.label}
            </button>
          ))}
        </div>
        {mine && !verdict && <LoadingState label="Running…" />}
        {verdict && (
          <StatusMessage tone={verdict.correct === mine!.choice ? 'success' : 'warning'} title={verdict.correct === mine!.choice ? 'You called it' : 'Not this time'}>
            {verdict.explanation}
          </StatusMessage>
        )}
        {mine && (
          <button type="button" className="btn btn-ghost lm-link-btn" onClick={() => setActiveExp(exp.id)}>
            Try again
          </button>
        )}
      </div>
      <div className="card lm-experiments">
        <div className="lm-kicker">Experiments</div>
        <ol>
          {EXPERIMENTS.map((e) => (
            <li key={e.id}>
              <button type="button" aria-current={e.id === exp.id ? 'true' : undefined} onClick={() => setActiveExp(e.id)}>
                {e.title}
              </button>
            </li>
          ))}
        </ol>
        <p className="small muted">Every answer is checked against the live result, not a stored answer.</p>
      </div>
    </>
  );
}

