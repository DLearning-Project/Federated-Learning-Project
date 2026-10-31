import { useModuleState } from '../../app/AppStateContext';
import { Card, ExtensionCard, PageHeader, Section } from '../../components/Card';
import { Details } from '../../components/Details';
import { formatInt, formatNumber, formatPercent } from '../../components/format';
import { parseNumber } from '../../components/NumberField';
import { Quiz } from '../../components/Quiz';
import { AssumptionsList, ModeBadge } from '../../components/Results';
import { ApiErrorState, EmptyState, LoadingState, StatusMessage } from '../../components/StatusMessage';
import { useApiAction } from '../../hooks/useApiAction';
import { aggregateFedAvg } from './api';
import { StackedBars } from './Charts';
import distributions from './data/label_distributions.json';
import { SpectrogramExplorer } from './SpectrogramExplorer';
import type { ClientRow, FedAvgInput, Scenario } from './types';

const PIPELINE = [
  { title: 'Audio', text: 'A hydrophone records underwater sound, such as a passing ship.' },
  { title: 'Spectrogram', text: 'The sound becomes an image of frequency over time.' },
  { title: 'Local CNN training', text: 'Each station trains an audio model (planned: PANN MobileNetV2) on its own recordings.' },
  { title: 'FedAvg', text: 'The server averages model weights, weighted by how much data each station has.' },
  { title: 'Global model', text: 'The combined model goes back to every station for the next round.' },
];

function rowsFor(scenario: Scenario): ClientRow[] {
  const counts = distributions.scenarios[scenario].counts;
  return distributions.clients.map((name, i) => ({
    name,
    samples: String(counts[i].reduce((a, b) => a + b, 0)),
    vector: distributions.toy_vectors[i].map(String),
  }));
}

function parseRows(rows: ClientRow[]): { input: FedAvgInput | null; errors: string[] } {
  const errors: string[] = [];
  const clients = rows.map((r) => {
    const [n, nErr] = parseNumber(r.samples, { min: 0, max: 10_000_000, integer: true, label: `${r.name} samples` });
    if (nErr) errors.push(nErr);
    const vector = r.vector.map((v, j) => {
      const [x, xErr] = parseNumber(v, { min: -1e6, max: 1e6, label: `${r.name} parameter ${j + 1}` });
      if (xErr) errors.push(xErr);
      return x;
    });
    return { name: r.name, sample_count: n, vector };
  });
  if (!errors.length && clients.every((c) => c.sample_count === 0)) errors.push('At least one station needs more than zero samples.');
  return { input: errors.length ? null : { clients }, errors };
}

export default function AcousticsPage() {
  const [scenario, setScenario] = useModuleState<Scenario>('ac.scenario', 'balanced');
  const [rows, setRows] = useModuleState<ClientRow[]>('ac.rows', rowsFor('balanced'));
  const [showErrors, setShowErrors] = useModuleState('ac.showErrors', false);
  const { state, data, run, reset } = useApiAction('ac.calc', aggregateFedAvg);

  const { input, errors } = parseRows(rows);
  const stale = data && input && JSON.stringify(data.inputs.clients) !== JSON.stringify(input.clients);
  const dims = rows[0].vector.length;

  const setCell = (i: number, patch: (r: ClientRow) => ClientRow) => setRows((rs) => rs.map((r, k) => (k === i ? patch(r) : r)));

  const submit = () => {
    setShowErrors(true);
    if (input) void run(input);
  };

  return (
    <div className="page">
      <PageHeader
        eyebrow="Module 2 · Underwater acoustics"
        accent="var(--accent-acoustics)"
        title="Listening to the ocean, together"
        lead="Hydrophone stations record ships, but sharing raw recordings can be costly or restricted. Federated learning lets each station train locally and share only model weights."
      />

      <Card title="From sound to a shared model">
        <ol className="grid-3" style={{ listStyle: 'none', padding: 0, gridTemplateColumns: 'repeat(auto-fit, minmax(11rem, 1fr))', gap: 'var(--space-3)' }}>
          {PIPELINE.map((p, i) => (
            <li key={p.title} className="stat">
              <span className="stat-label">Step {i + 1}</span>
              <strong>{p.title}</strong>
              <span className="small muted">{p.text}</span>
            </li>
          ))}
        </ol>
        <Details summary="Technical details: spectrograms and PANN MobileNetV2">
          <p>
            A spectrogram splits audio into short overlapping windows and measures the energy at each frequency (a short-time Fourier transform). Ship
            engines and propellers create steady tones and harmonics, which appear as horizontal lines. A log-Mel spectrogram then groups the
            frequencies into bands that are narrow at low frequencies and wide at high ones, and measures each band in decibels.
          </p>
          <p>
            PANN MobileNetV2 is a compact convolutional network pretrained on AudioSet, a large collection of everyday sounds. It computes the log-Mel
            spectrogram itself from 32 kHz audio, and is the planned model for this module. <strong>This page does not train a CNN and does not predict accuracy.</strong>
          </p>
        </Details>
      </Card>

      <SpectrogramExplorer />

      <Card
        title="Who has which labels?"
        subtitle="Illustrative vessel labels at three stations"
        actions={
          <div className="segmented" role="group" aria-label="Label distribution">
            {(['balanced', 'skewed'] as Scenario[]).map((s) => (
              <button key={s} type="button" aria-pressed={scenario === s} onClick={() => setScenario(s)}>
                {s === 'balanced' ? 'Balanced' : 'Skewed'}
              </button>
            ))}
          </div>
        }
      >
        <StackedBars clients={distributions.clients} classes={distributions.classes} counts={distributions.scenarios[scenario].counts} />
        <p className="small muted">
          {scenario === 'balanced'
            ? 'Every station has the same mix of vessel types. Local models learn similar patterns.'
            : 'Each station mostly hears one vessel type (non-IID data). Local models drift apart, which makes aggregation harder.'}
        </p>
        <div>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => setRows((rs) => rs.map((r, i) => ({ ...r, samples: rowsFor(scenario)[i].samples })))}
          >
            Use these sample counts below
          </button>
        </div>
      </Card>

      <Section
        title="FedAvg calculator"
        intro="Edit how many samples each station trained on, and a tiny toy parameter vector for each. The backend returns each station's weight and the weighted global vector."
      >
        <div className="workbench">
          <Card title="Client updates" as="div">
            <form
              className="form"
              noValidate
              onSubmit={(e) => {
                e.preventDefault();
                submit();
              }}
            >
              {rows.map((r, i) => (
                <fieldset key={r.name} className="field" style={{ border: '1px solid var(--color-border)', borderRadius: 'var(--radius-md)', padding: 'var(--space-3) var(--space-4)', margin: 0 }}>
                  <legend style={{ fontWeight: 600, padding: '0 var(--space-1)' }}>{r.name}</legend>
                  <label className="field-label" htmlFor={`ac-n-${i}`}>
                    Samples <span className="field-unit">n</span>
                  </label>
                  <input
                    id={`ac-n-${i}`}
                    className="input"
                    type="number"
                    min={0}
                    step={1}
                    value={r.samples}
                    onChange={(e) => setCell(i, (row) => ({ ...row, samples: e.target.value }))}
                  />
                  <span className="field-label" style={{ marginTop: 'var(--space-2)' }}>
                    Toy parameters
                  </span>
                  <div style={{ display: 'grid', gridTemplateColumns: `repeat(${dims}, minmax(0, 1fr))`, gap: 'var(--space-2)' }}>
                    {r.vector.map((v, j) => (
                      <input
                        key={j}
                        className="input compact"
                        type="number"
                        step="any"
                        aria-label={`${r.name} parameter ${j + 1}`}
                        value={v}
                        onChange={(e) => setCell(i, (row) => ({ ...row, vector: row.vector.map((x, k) => (k === j ? e.target.value : x)) }))}
                      />
                    ))}
                  </div>
                </fieldset>
              ))}

              {showErrors && errors.length > 0 && (
                <StatusMessage tone="error" title="Please fix these inputs">
                  <ul>
                    {errors.map((e) => (
                      <li key={e}>{e}</li>
                    ))}
                  </ul>
                </StatusMessage>
              )}

              <div className="btn-row">
                <button type="submit" className="btn btn-primary" disabled={state.status === 'loading'}>
                  {state.status === 'loading' ? 'Aggregating…' : 'Aggregate'}
                </button>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => {
                    setRows(rowsFor(scenario));
                    setShowErrors(false);
                    reset();
                  }}
                >
                  Reset
                </button>
              </div>
            </form>
          </Card>

          <Card title="Aggregation result" actions={data && <ModeBadge mode={data.mode} />} as="div">
            {state.status === 'loading' && <LoadingState label="Aggregating…" />}
            {state.status === 'error' && <ApiErrorState error={state.error} onRetry={submit} />}
            {state.status === 'idle' && !data && <EmptyState>Press Aggregate to compute FedAvg weights and the global vector.</EmptyState>}
            {stale && state.status !== 'loading' && <StatusMessage tone="warning">Inputs changed since the last aggregation. Press Aggregate to update.</StatusMessage>}
            {data && (
              <>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
                  <h4>Client weights</h4>
                  {data.results.weights.map((w) => (
                    <div className="meter" key={w.name}>
                      <div className="meter-head">
                        <strong>{w.name}</strong>
                        <span>
                          {formatInt(w.sample_count)} / {formatInt(data.results.total_samples)} samples = {formatPercent(w.weight)}
                        </span>
                      </div>
                      <div className="meter-track" aria-hidden>
                        <span className="meter-fill" style={{ width: `${w.weight * 100}%`, background: 'var(--color-orange)' }} />
                      </div>
                    </div>
                  ))}
                </div>

                <div className="table-scroll">
                  <table className="result-table">
                    <caption className="visually-hidden">Client vectors, FedAvg global vector and unweighted mean</caption>
                    <thead>
                      <tr>
                        <th scope="col">Parameter</th>
                        {data.inputs.clients.map((c) => (
                          <th scope="col" key={c.name} className="num">
                            {c.name}
                          </th>
                        ))}
                        <th scope="col" className="num">
                          FedAvg global
                        </th>
                        <th scope="col" className="num">
                          Plain mean
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.results.global_vector.map((g, j) => (
                        <tr key={j}>
                          <th scope="row">θ{j + 1}</th>
                          {data.inputs.clients.map((c) => (
                            <td key={c.name} className="num">
                              {formatNumber(c.vector[j], 3)}
                            </td>
                          ))}
                          <td className="num">
                            <strong>{formatNumber(g, 3)}</strong>
                          </td>
                          <td className="num muted">{formatNumber(data.results.unweighted_mean[j], 3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="small muted">
                  The plain mean treats every station equally. FedAvg gives stations with more samples more influence.
                </p>
                <AssumptionsList assumptions={data.assumptions} />
              </>
            )}
          </Card>
        </div>
      </Section>

      <Quiz
        question="Why do stations contribute different weights in FedAvg?"
        options={[
          {
            id: 'samples',
            label: 'Each weight is proportional to how many training samples the station used.',
            correct: true,
            feedback: 'Right. A station with 3× more samples gets 3× the weight, so the global model reflects the total data, not just the number of stations.',
          },
          { id: 'accuracy', label: 'Stations with more accurate local models get more weight.', feedback: 'Standard FedAvg does not look at accuracy at all; it only uses sample counts.' },
          { id: 'labels', label: 'Stations with more vessel classes get more weight.', feedback: 'Label variety does not change the weight. Only the sample count does, even when labels are skewed.' },
          { id: 'random', label: 'The server picks weights at random each round.', feedback: 'Client selection may be random, but the aggregation weights are n_k / Σn.' },
        ]}
        explanation="Note that skewed labels can still hurt the global model even with correct weights: weighting fixes how much each station counts, not what it learned."
      />

      <ExtensionCard title="Future: PANN MobileNetV2 training and evaluation">
        <p>
          Owner: Acoustics teammate. Planned additions, trained offline in <code>training/acoustics/</code>, never inside an API request:
        </p>
        <ul>
          <li>Centralized fine-tuning of PANN MobileNetV2 for vessel-present versus no-vessel detection on QiandaoEar22.</li>
          <li>FedAvg with three simulated stations, using balanced (IID) and skewed (non-IID) splits of the same training data.</li>
          <li>Measured evaluation results on a held-out set that is kept separate from all client training data, exported as JSON for this page.</li>
        </ul>
      </ExtensionCard>
    </div>
  );
}
