import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { useModuleState } from '../../../app/AppStateContext';
import { formatBytes, formatNumber, formatPercent } from '../../../components/format';
import { AssumptionsList } from '../../../components/Results';
import { ApiErrorState, EmptyState, LoadingState, StatusMessage } from '../../../components/StatusMessage';
import { useApiAction, type ActionState } from '../../../hooks/useApiAction';
import { fetchPaper, simulateRound } from '../api';
import { useLab } from '../lab/LabContext';
import { byLevel } from '../lab/levels';
import { Term } from '../lab/Term';
import { ClassMixGrid, DownloadBars, METHOD_LABEL, SpectrumChart } from '../simulator/charts';
import { classesHeld, EXPERIMENTS, payloadOf, pct, sum, type Experiment } from '../simulator/experiments';
import { DEFAULTS, sameInput, toInput, type Settings, type Tab } from '../simulator/settings';
import { ClientsTable, ServerCostTable, SingularValuesTable } from '../simulator/tables';
import type { Level, MethodId, PaperResponse, SimulatorRoundInput, SimulatorRoundResponse } from '../types';

type PaperAction = {
  state: ActionState<PaperResponse>;
  data?: PaperResponse;
  run: (input: null) => Promise<void>;
};

const STEP_MS = 700;

const TAB_LABELS: Record<Tab, { beginner: string; student: string; researcher: string }> = {
  accuracy: {
    beginner: 'Accuracy',
    student: 'Accuracy',
    researcher: 'Accuracy',
  },
  spectrum: {
    beginner: 'What gets kept',
    student: 'Singular values (τ)',
    researcher: 'Spectrum σ(P), τ',
  },
  download: {
    beginner: 'Download size',
    student: 'Download cost',
    researcher: 'Payload & server cost',
  },
  data: {
    beginner: 'Who has what data',
    student: 'Client data',
    researcher: 'Partition & weights',
  },
};
const TAB_ORDER: Tab[] = ['accuracy', 'spectrum', 'download', 'data'];

const BEGINNER_METHOD_LABEL: Record<MethodId, string> = {
  exact_stack: 'Send everything',
  fedit: 'Simple averaging',
  florist: 'FLoRIST',
};

/** Plain-language version of the backend assumptions, for beginners. */
const BEGINNER_ASSUMPTIONS = [
  'The add-ons here are small made-up examples, not trained on real notes, so the numbers show how combining works rather than how good a chatbot gets.',
  'The “hospitals” have made-up data in 16 topics; hospitals with more examples count for more when the server combines.',
  'Download sizes count only the add-on, not the big model each hospital already has.',
];

/** Paper references shown next to each result for researchers (term ids in paper_facts.json). */
const TAB_REFS: Record<Tab, string[]> = {
  accuracy: [],
  spectrum: ['weighted_stacking', 'energy_threshold', 'eckart_young_bound'],
  download: ['communication_cost'],
  data: ['aggregation_weight'],
};

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

  // Paper-reported tables load the first time the download tab opens.
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
    setPrediction({
      experiment: exp.id,
      choice,
      before: current,
      target: toInput(next, round),
    });
  };

  const fedVisible = lvl === 'beginner' ? settings.beginner_fedit : settings.show_fedit;
  const mathVisible = lvl === 'researcher' || (lvl === 'student' && showMath);
  const r = current?.results;

  return (
    <div className="page lm-sim">
      <header className="lm-sim-head">
        <div>
          <span className="eyebrow">Page 4 of 4 · Language models</span>
          <h1>FLoRIST Simulator</h1>
          <p className="muted">
            {byLevel(lvl, {
              beginner: 'Watch a server combine what many clients learned into one small update, and see what that saves and what it loses.',
              student:
                'Sample clients, give them LoRA adapters, and aggregate them three ways. Every number here is computed live by the backend on small synthetic adapters.',
              researcher:
                'One aggregation round per step: Dirichlet partition, client sampling, data-size weights, then FedIT, exact weighted stacking and FLoRIST (Algorithm 1) on synthetic 16 × 32 adapters, computed live in float64.',
            })}
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
            add-on on its own notes. A server combines the add-ons and sends one back to everybody. This page shows how big that download is, and how much of
            the hospitals’ work survives the combining.
          </p>
        </div>
      )}

      <div className="lm-sim-grid">
        <Controls settings={settings} update={update} level={lvl} />

        <div className="lm-sim-main">
          <Kpis result={current} level={lvl} fedVisible={fedVisible} />

          <section className="card lm-results" aria-label="Results">
            <div className="lm-results-head">
              <div className="segmented" role="tablist" aria-label="Result views">
                {TAB_ORDER.map((t) => (
                  <button key={t} type="button" role="tab" aria-selected={tab === t} onClick={() => setTab(t)}>
                    {TAB_LABELS[t][lvl]}
                  </button>
                ))}
              </div>
              {sim.state.status === 'loading' && <span className="small muted">Updating…</span>}
            </div>

            {sim.state.status === 'error' && <ApiErrorState error={sim.state.error} onRetry={() => sim.run(input)} />}
            {!data && sim.state.status === 'loading' && <LoadingState label="Simulating round…" />}

            {tab === 'accuracy' && (
              <EmptyState>
                {byLevel(lvl, {
                  beginner: (
                    <>
                      <strong>Coming later.</strong> The add-ons on this page are made-up examples, not trained on real text, so there is no chatbot to score
                      yet. We will not show invented numbers.
                    </>
                  ),
                  student: (
                    <>
                      <strong>Accuracy is available after the toy experiment (build step 5).</strong> The adapters on this page are synthetic, so there is no
                      model to score yet. We do not draw made-up curves. For accuracy measured in the paper, see Table 2 of the FLoRIST paper.
                    </>
                  ),
                  researcher: (
                    <>
                      <strong>Not computed: adapters are synthetic.</strong> Test accuracy per round needs local training (build step 5, toy model). Paper
                      results: Table 2 (MMLU, TinyLlama and LLaMA-3.2-1B, homogeneous and heterogeneous ranks).
                    </>
                  ),
                })}
              </EmptyState>
            )}

            {r && tab !== 'accuracy' && (
              <div className={sim.state.status === 'loading' ? 'lm-stale' : undefined}>
                {lvl === 'researcher' && <Refs tab={tab} />}
                {tab === 'spectrum' && (
                  <div className="lm-stack">
                    <SpectrumChart
                      values={r.singular_values}
                      energy={r.cumulative_energy}
                      p={r.p}
                      tau={current!.inputs.tau}
                      xLabel={byLevel(lvl, {
                        beginner: 'pieces of the update (most important first)',
                        student: 'component (largest first)',
                        researcher: 'i',
                      })}
                      plain={lvl === 'beginner'}
                    />
                    <ul className="legend">
                      <li>
                        <span className="swatch" style={{ background: 'var(--color-orange)' }} /> {lvl === 'beginner' ? 'kept' : 'kept by FLoRIST'}
                      </li>
                      <li>
                        <span
                          className="swatch"
                          style={{
                            background: 'var(--color-chart-light)',
                            border: '1px solid var(--color-chart-gray)',
                          }}
                        />{' '}
                        {lvl === 'beginner' ? 'dropped' : 'discarded'}
                      </li>
                      <li>
                        <span className="swatch" style={{ background: 'var(--color-ink)', height: 3 }} />{' '}
                        {lvl === 'beginner' ? 'total detail kept so far' : 'cumulative energy'}
                      </li>
                      <li>
                        <span className="swatch lm-swatch-dash" /> {lvl === 'beginner' ? 'target' : 'threshold τ'}
                      </li>
                    </ul>
                    {lvl === 'student' && <p className="small muted">Hover a bar to see its value.</p>}
                    {lvl === 'researcher' && <SingularValuesTable r={r} />}
                  </div>
                )}
                {tab === 'download' && <DownloadTab data={current!} settings={settings} level={lvl} paper={paper} fedVisible={fedVisible} />}
                {tab === 'data' && (
                  <div className="lm-stack">
                    <ClassMixGrid counts={r.class_counts} sampled={r.sampled_clients} ranks={r.all_client_ranks} />
                    {lvl === 'researcher' && <ClientsTable r={r} />}
                  </div>
                )}

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

          <Playback
            round={round}
            rounds={settings.rounds}
            playing={playing}
            level={lvl}
            onRun={runPlayback}
            onPause={() => setPlaying(false)}
            onStep={step}
            onReset={reset}
          />

          {r && r.notices.length > 0 && lvl !== 'beginner' && (
            <StatusMessage tone="warning" title="Notes for this round">
              <ul className="lm-list">
                {r.notices.map((n) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            </StatusMessage>
          )}
          {current && (
            <AssumptionsList
              title={lvl === 'beginner' ? 'What this page simplifies' : 'Assumptions behind these numbers'}
              assumptions={lvl === 'beginner' ? BEGINNER_ASSUMPTIONS : [...current.assumptions, ...current.results.payload_assumptions]}
            />
          )}
        </div>

        <aside className="lm-sim-side" aria-label="Experiments">
          <Experiments
            level={lvl}
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
                Each hospital sends a summary of what it learned. Sending everything staples all the summaries together: nothing is lost, but the packet keeps
                growing. FLoRIST rewrites them into one short summary that keeps the points most hospitals agree on (the tall bars).
              </p>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}

/* ---------- Paper references (researcher) ---------- */

function Refs({ tab }: { tab: Tab }) {
  const { term, openTerm } = useLab();
  const refs = TAB_REFS[tab].map((id) => term(id)).filter((t): t is NonNullable<typeof t> => !!t);
  if (!refs.length) return null;
  return (
    <p className="lm-refs small">
      {refs.map((t) => (
        <button key={t.id} type="button" className="lm-chip lm-chip-btn" onClick={() => openTerm(t.id)} title={t.label}>
          {t.label}: {t.reference}
        </button>
      ))}
    </p>
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
  ends,
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
  ends?: [string, string];
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
      {ends && (
        <span className="lm-ends small muted" aria-hidden>
          <span>{ends[0]}</span>
          <span>{ends[1]}</span>
        </span>
      )}
      {hint && <span className="small muted">{hint}</span>}
    </div>
  );
}

function Controls({ settings: s, update, level }: { settings: Settings; update: (p: Partial<Settings>) => void; level: Level }) {
  const beginner = level === 'beginner';
  const researcher = level === 'researcher';
  return (
    <aside className="lm-sim-controls" aria-label="Simulation settings">
      <Group title={beginner ? 'Who takes part' : 'Federation'}>
        {!beginner && (
          <Slider
            id="lm-total"
            label={
              <Term htmlFor="lm-total" definition="Every device or organisation that holds its own data and could take part in training.">
                {researcher ? 'Total clients K' : 'Total clients'}
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
              {beginner ? 'Hospitals per round' : 'Clients per round'}
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

      <Group title={beginner ? 'Their data' : 'Client data'}>
        <Slider
          id="lm-alpha"
          label={
            <Term
              htmlFor="lm-alpha"
              definition={
                beginner
                  ? 'How different the hospitals’ notes are. Very uneven: each hospital sees only a few topics. Very similar: everyone sees a bit of everything.'
                  : "Dirichlet α controls how uneven the clients' data is: small α gives each client only a few classes; large α gives everyone a similar mix."
              }
            >
              {byLevel(level, {
                beginner: 'How different is their data?',
                student: 'Dirichlet α',
                researcher: 'Dirichlet α (label skew)',
              })}
            </Term>
          }
          value={Math.log10(s.alpha)}
          display={beginner ? alphaWords(s.alpha) : formatNumber(s.alpha, s.alpha < 1 ? 2 : 1)}
          min={-2}
          max={3}
          step={0.05}
          onChange={(v) => update({ alpha: Number(Math.pow(10, v).toPrecision(2)) })}
          ends={beginner ? ['very uneven', 'very similar'] : undefined}
          hint={
            <>
              {beginner
                ? 'The paper uses a fairly uneven setting.'
                : researcher
                  ? 'Log scale, 0.01–1000 · paper: α = 0.5 (Sec. 4.1)'
                  : 'More uneven ← → more similar · paper uses 0.5'}{' '}
              {s.alpha !== 0.5 && (
                <button type="button" className="btn btn-ghost lm-inline-btn" onClick={() => update({ alpha: 0.5 })}>
                  reset
                </button>
              )}
            </>
          }
        />
      </Group>

      <Group title={beginner ? 'Add-on sizes' : 'LoRA ranks'}>
        <div className="lm-field">
          <Term termId="heterogeneous_ranks">{beginner ? 'Do all hospitals use the same size?' : researcher ? 'r_k' : 'Client ranks'}</Term>
          <div className="segmented lm-full" role="group" aria-label="Client ranks">
            <button type="button" aria-pressed={!s.heterogeneous} onClick={() => update({ heterogeneous: false })}>
              {beginner ? 'Same size' : 'Homogeneous'}
            </button>
            <button type="button" aria-pressed={s.heterogeneous} onClick={() => update({ heterogeneous: true })}>
              {beginner ? 'Mixed sizes' : 'Heterogeneous'}
            </button>
          </div>
          <span className="small muted">
            {byLevel(level, {
              beginner: s.heterogeneous
                ? 'Smaller hospitals use smaller add-ons, bigger ones use larger add-ons.'
                : 'Every hospital uses the same add-on size.',
              student: s.heterogeneous ? 'Mix of ranks 1, 2 and 4 (most clients small, like the paper).' : 'Every client uses rank 4.',
              researcher: s.heterogeneous ? 'r_k by client id: 50% r=1, 30% r=2, 20% r=4 (heavy-tailed, cf. Sec. 4.1).' : 'r_k = 4 for all k.',
            })}
          </span>
        </div>
      </Group>

      <Group title={beginner ? 'How the server combines' : 'Aggregation'}>
        <div className="lm-field" role="group" aria-label="Methods to compare">
          <span>{beginner ? 'Compare' : 'Methods'}</span>
          <label className="lm-check">
            <input type="checkbox" checked={s.show_florist} onChange={(e) => update({ show_florist: e.target.checked })} />
            <Term termId="selected_rank">FLoRIST</Term>
          </label>
          <label className="lm-check">
            <input
              type="checkbox"
              checked={beginner ? s.beginner_fedit : s.show_fedit}
              onChange={(e) => update(beginner ? { beginner_fedit: e.target.checked } : { show_fedit: e.target.checked })}
            />
            <Term termId="fedit">{beginner ? 'Simple averaging' : 'FedIT'}</Term>
          </label>
          <label className="lm-check">
            <input type="checkbox" checked disabled />
            <Term termId="weighted_stacking">{beginner ? 'Send everything' : 'Exact stacking'}</Term>
            {!beginner && <span className="small muted">(reference)</span>}
          </label>
        </div>
        <Slider
          id="lm-tau"
          label={
            <Term termId="energy_threshold" htmlFor="lm-tau">
              {byLevel(level, {
                beginner: 'How much detail to keep',
                student: 'Energy threshold τ',
                researcher: 'τ',
              })}
            </Term>
          }
          value={s.tau}
          display={beginner ? `${Math.round(s.tau * 100)}%` : s.tau.toFixed(2)}
          min={0.5}
          max={1}
          step={0.01}
          onChange={(v) => update({ tau: v })}
          ends={beginner ? ['send less', 'keep more detail'] : undefined}
          hint={byLevel(level, {
            beginner: 'The paper keeps 90%.',
            student: 'FLoRIST only · paper default 0.90',
            researcher: 'FLoRIST only · paper: τ = 0.90 and tuned τ* (Sec. 4.2)',
          })}
        />
      </Group>

      {!beginner && (
        <Group title="Run settings" defaultOpen={researcher}>
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

function alphaWords(alpha: number): string {
  if (alpha < 0.1) return 'very uneven';
  if (alpha < 1) return 'uneven';
  if (alpha < 10) return 'mixed';
  if (alpha < 100) return 'similar';
  return 'very similar';
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

function Kpis({ result, level, fedVisible }: { result: SimulatorRoundResponse | undefined; level: Level; fedVisible: boolean }) {
  const r = result?.results;
  const fl = payloadOf(result, 'florist');
  const st = payloadOf(result, 'exact_stack');
  const dash = '—';
  if (!r || !fl || !st) {
    return (
      <div className="lm-kpis">
        <Kpi label="Waiting for the first result" value={dash} sub="The backend computes every number on this page." />
      </div>
    );
  }
  const sigma = sum(r.client_ranks);
  const ratio = formatNumber(st.download_per_client_bytes / Math.max(fl.download_per_client_bytes, 1), 1);
  const fe = r.aggregation_error_fedit;
  const tau = result!.inputs.tau;
  const b = result!.inputs.bytes_per_param;

  if (level === 'beginner') {
    return (
      <div className="lm-kpis">
        <Kpi
          label={<Term termId="selected_rank">Pieces kept</Term>}
          value={`${r.p} of ${sigma}`}
          sub="FLoRIST keeps the most important pieces of the combined update and drops the rest."
        />
        <Kpi
          label="Each client downloads"
          value={formatBytes(fl.download_per_client_bytes)}
          sub={`${ratio}× less than sending everything (${formatBytes(st.download_per_client_bytes)})`}
        />
        <Kpi label="Lost by compressing" value={pct(r.reconstruction_error.relative)} sub="How far FLoRIST’s small update is from the full one." />
        {fedVisible && (
          <Kpi
            label="Error from simple averaging"
            value={fe ? pct(fe.relative) : 'not possible'}
            sub={
              fe
                ? 'Averaging the two parts separately mixes up different hospitals’ work.'
                : 'Hospitals use different add-on sizes, so they cannot be averaged.'
            }
          />
        )}
      </div>
    );
  }

  if (level === 'researcher') {
    return (
      <div className="lm-kpis">
        <Kpi
          label={<Term termId="selected_rank">p</Term>}
          value={r.p}
          sub={`max r_k = ${Math.max(...r.client_ranks)} · Σr_k = ${sigma} · τ = ${tau.toFixed(2)}`}
        />
        <Kpi
          label={<Term termId="communication_cost">download / client</Term>}
          value={formatBytes(fl.download_per_client_bytes)}
          sub={`p(m+n)·b = ${r.p}·${r.m + r.n}·${b} · stack Σr_k(m+n)·b = ${formatBytes(st.download_per_client_bytes)}`}
        />
        <Kpi
          label={<Term termId="eckart_young_bound">‖ΔW − B_gA_g‖_F</Term>}
          value={r.reconstruction_error.absolute.toPrecision(3)}
          sub={`rel ${pct(r.reconstruction_error.relative)} · = √Σ_{i>p} σ_i² = ${r.eq5_bound.toPrecision(3)} (Eq. 5)`}
        />
        {fedVisible && (
          <Kpi
            label={<Term termId="cross_term">‖ΔW − B̄Ā‖_F (FedIT)</Term>}
            value={fe ? fe.absolute.toPrecision(3) : 'undefined'}
            sub={fe ? `rel ${pct(fe.relative)} · stacking: ${r.aggregation_error_exact_stack.absolute.toExponential(1)}` : 'r_k differ: Σw_kB_k is not defined'}
          />
        )}
      </div>
    );
  }

  return (
    <div className="lm-kpis">
      <Kpi
        label={<Term termId="selected_rank">Global rank p</Term>}
        value={r.p}
        sub={`The smallest rank whose singular values hold ${Math.round(tau * 100)}% of the energy (max r_k = ${Math.max(...r.client_ranks)}, Σr_k = ${sigma}).`}
      />
      <Kpi
        label={<Term termId="communication_cost">Download per client</Term>}
        value={formatBytes(fl.download_per_client_bytes)}
        sub={`FLoRIST sends p(m+n) numbers: ${ratio}× less than exact stacking’s ${formatBytes(st.download_per_client_bytes)}.`}
      />
      <Kpi
        label={<Term termId="eckart_young_bound">FLoRIST error</Term>}
        value={pct(r.reconstruction_error.relative)}
        sub="Only the discarded singular values cause error, so it equals the Eq. 5 bound."
      />
      {fedVisible && (
        <Kpi
          label={<Term termId="cross_term">FedIT error</Term>}
          value={fe ? pct(fe.relative) : 'n/a'}
          sub={fe ? 'Multiplying averaged factors adds cross terms between clients.' : 'Mixed ranks: FedIT cannot average them.'}
        />
      )}
    </div>
  );
}

/* ---------- Download tab ---------- */

function DownloadTab({
  data,
  settings,
  level,
  paper,
  fedVisible,
}: {
  data: SimulatorRoundResponse;
  settings: Settings;
  level: Level;
  paper: PaperAction;
  fedVisible: boolean;
}) {
  const r = data.results;
  const shown = r.payload.filter((p) => (p.method === 'fedit' ? fedVisible : p.method === 'florist' ? settings.show_florist : true));
  const reported = paper.data?.reported_results;
  const table3 = reported?.table3;
  return (
    <div className="lm-stack">
      <DownloadBars payload={shown} labels={level === 'beginner' ? BEGINNER_METHOD_LABEL : METHOD_LABEL} showRank={level !== 'beginner'} />
      {level === 'researcher' && <ServerCostTable r={r} table4={reported?.table4} />}
      <div className="lm-paper">
        <div className="lm-paper-head">
          <span className="badge">Paper-reported</span>
          <strong>{level === 'beginner' ? 'At real scale' : (table3?.reference ?? 'Table 3')}</strong>
          <span className="small muted">
            {level === 'beginner' ? 'From the FLoRIST paper, for a real chatbot model.' : 'Not computed here: these are the paper’s numbers, for scale.'}
          </span>
        </div>
        {paper.state.status === 'loading' && <LoadingState label="Loading paper values…" />}
        {paper.state.status === 'error' && <ApiErrorState error={paper.state.error} onRetry={() => paper.run(null)} />}
        {table3 && level === 'beginner' && <BeginnerScale rows={table3.rows} />}
        {table3 && level !== 'beginner' && (
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
            {level === 'researcher' && <p className="small muted">{table3.setup}</p>}
          </>
        )}
      </div>
    </div>
  );
}

function BeginnerScale({ rows }: { rows: { method: string; download_mb: number }[] }) {
  const full = rows.find((x) => x.method === 'Full FT');
  const flo = rows.find((x) => x.method.startsWith('FLoRIST [τ=0.9'));
  if (!full || !flo) return null;
  return (
    <p>
      Sending the whole model back to the clients would take about <strong>{formatNumber(full.download_mb, 0)} MB</strong>. FLoRIST sends about{' '}
      <strong>{formatNumber(flo.download_mb, 1)} MB</strong>, roughly {formatNumber(full.download_mb / flo.download_mb, 0)} times less.
    </p>
  );
}

/* ---------- Reading captions and math ---------- */

function reading(tab: Tab, d: SimulatorRoundResponse, level: Level): string {
  const r = d.results;
  const sigma = sum(r.client_ranks);
  if (tab === 'spectrum') {
    const kept = r.p > 0 ? r.cumulative_energy[r.p - 1] : 0;
    return byLevel(level, {
      beginner: `Each bar is one piece of what the clients learned together; tall bars matter most. FLoRIST keeps the ${r.p} orange bar${
        r.p === 1 ? '' : 's'
      }, which hold ${formatPercent(kept)} of the total, and drops the rest.`,
      student: `The ${r.singular_values.length} bars are the singular values of the weighted average update ΔW. The first ${r.p} hold ${formatPercent(
        kept,
      )} of the energy, which meets τ = ${d.inputs.tau.toFixed(2)}, so FLoRIST sends rank ${r.p} instead of Σr_k = ${sigma}.`,
      researcher: `rank(ΔW) ≤ min(m, n, Σr_k) = ${Math.min(r.m, r.n, sigma)}; ${r.singular_values.length} singular values shown. E(p) = ${formatPercent(
        kept,
        2,
      )} ≥ τ = ${d.inputs.tau.toFixed(2)} at p = ${r.p}. σ(P) equals σ(ΔW) because B_stack and A_stack factor ΔW exactly.`,
    });
  }
  if (tab === 'download') {
    const fl = payloadOf(d, 'florist')!;
    const st = payloadOf(d, 'exact_stack')!;
    const fe = payloadOf(d, 'fedit');
    return byLevel(level, {
      beginner: `Each client downloads ${formatBytes(fl.download_per_client_bytes)} with FLoRIST, versus ${formatBytes(
        st.download_per_client_bytes,
      )} if the server sent everything.`,
      student: `Each client downloads ${formatBytes(fl.download_per_client_bytes)} with FLoRIST (rank ${fl.download_rank}) and ${formatBytes(
        st.download_per_client_bytes,
      )} with exact stacking (rank ${st.download_rank}).${fe ? ` FedIT sends ${formatBytes(fe.download_per_client_bytes)}, but its average is not exact.` : ''}`,
      researcher: `Per client per round: stack ${st.download_rank}(m+n), FLoRIST ${fl.download_rank}(m+n)${
        fe ? `, FedIT ${fe.download_rank}(m+n)` : ''
      } parameters at ${d.inputs.bytes_per_param} B each. Upload is r_k(m+n) for every method. Total upload this round: ${formatBytes(fl.upload_total_bytes)}.`,
    });
  }
  const held = r.sampled_clients.map((c) => classesHeld(r.class_counts[c]));
  const avg = held.length ? sum(held) / held.length : 0;
  return byLevel(level, {
    beginner: `Each row is one hospital and each column is one topic; darker means more examples. The highlighted rows (●) take part this round and have ${formatNumber(
      avg,
      1,
    )} of the 16 topics on average. Hospitals with more notes count for more when the server combines.`,
    student: `Rows are clients and columns are the 16 classes; darker cells hold more examples. With α = ${formatNumber(d.inputs.alpha, 2)}, this round's ${
      r.sampled_clients.length
    } sampled clients (●) hold ${formatNumber(avg, 1)} classes on average, and they are weighted by how many examples they have.`,
    researcher: `Per-class Dirichlet(α = ${formatNumber(d.inputs.alpha, 3)}) split of 16 × 60 samples over K = ${r.class_counts.length} clients. w_k = n_k / Σ n_j over the ${
      r.sampled_clients.length
    } sampled clients${r.sample_counts.some((n) => n === 0) ? '; clients with n_k = 0 get w_k = 0' : ''}.`,
  });
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
  level,
  onRun,
  onPause,
  onStep,
  onReset,
}: {
  round: number;
  rounds: number;
  playing: boolean;
  level: Level;
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
        {byLevel(level, {
          beginner: 'Each round picks different hospitals. The add-ons are made-up examples, so they do not improve from round to round yet.',
          student:
            'Each round samples a new set of clients and aggregates their adapters. The adapters are synthetic, so nothing is trained between rounds yet.',
          researcher:
            'Rounds re-sample clients (seed, t); adapters are fixed synthetic draws per client, with no local training or global-model carry-over yet (step 5).',
        })}
      </p>
    </div>
  );
}

/* ---------- Experiments ---------- */

function Experiments({
  level,
  data,
  current,
  prediction,
  activeExp,
  setActiveExp,
  onPredict,
}: {
  level: Level;
  data: SimulatorRoundResponse | undefined;
  current: SimulatorRoundResponse | undefined;
  prediction: Prediction | null;
  activeExp: string;
  setActiveExp: (id: string) => void;
  onPredict: (exp: Experiment, choice: string) => void;
}) {
  const exp = EXPERIMENTS.find((e) => e.id === activeExp) ?? EXPERIMENTS[0];
  const words = byLevel(level, exp.wording);
  const mine = prediction?.experiment === exp.id ? prediction : null;
  const after = mine && data && sameInput(data.inputs, mine.target) ? data : undefined;
  const verdict = mine && after ? exp.verdict(mine.before, after, level) : null;

  return (
    <>
      <div className="card lm-predict">
        <div className="lm-kicker">Predict, then run</div>
        <strong>{words.title}</strong>
        <p>{words.question}</p>
        <div className="lm-options" role="radiogroup" aria-label="Your prediction">
          {exp.optionIds.map((id) => (
            <button
              key={id}
              type="button"
              role="radio"
              className="choice"
              aria-checked={mine?.choice === id}
              disabled={!current || !!mine}
              onClick={() => onPredict(exp, id)}
            >
              {words.options[id]}
            </button>
          ))}
        </div>
        {mine && !verdict && <LoadingState label="Running…" />}
        {verdict && (
          <StatusMessage
            tone={verdict.correct === mine!.choice ? 'success' : 'warning'}
            title={verdict.correct === mine!.choice ? 'You called it' : 'Not this time'}
          >
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
                {byLevel(level, e.wording).title}
              </button>
            </li>
          ))}
        </ol>
        <p className="small muted">Every answer is checked against the live result, not a stored answer.</p>
      </div>
    </>
  );
}
