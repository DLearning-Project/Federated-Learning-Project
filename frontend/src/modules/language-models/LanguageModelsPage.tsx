import { useModuleState } from '../../app/AppStateContext';
import { Card, ExtensionCard, PageHeader, Section } from '../../components/Card';
import { Details } from '../../components/Details';
import { StepThrough, type DiagramStep } from '../../components/FederatedDiagram';
import { formatBytes, formatInt, formatNumber } from '../../components/format';
import { NumberField, parseNumber } from '../../components/NumberField';
import { Quiz } from '../../components/Quiz';
import { AssumptionsList, ModeBadge, StatGrid } from '../../components/Results';
import { ApiErrorState, EmptyState, LoadingState, StatusMessage } from '../../components/StatusMessage';
import { useApiAction } from '../../hooks/useApiAction';
import { calculateLora } from './api';
import { MatrixShapes, SavingsBars } from './MatrixVisual';
import type { LoraCalcInput, LoraForm } from './types';

const DEFAULT_FORM: LoraForm = { d_in: '4096', d_out: '4096', rank: '8', clients: '10', rounds: '50', bytes_per_param: '2' };

const LIMITS = {
  d_in: { min: 1, max: 16384, label: 'Input dimension' },
  d_out: { min: 1, max: 16384, label: 'Output dimension' },
  rank: { min: 1, max: 16384, label: 'LoRA rank' },
  clients: { min: 1, max: 1000, label: 'Clients' },
  rounds: { min: 1, max: 1000, label: 'Rounds' },
} as const;

type NumericKey = keyof typeof LIMITS;

const STEPS: DiagramStep[] = [
  {
    title: 'Send model',
    active: ['download', 'aggregate'],
    body: (
      <p>
        The server sends every client the same <strong>frozen base model</strong> plus small starting adapters. This one-time download of the base model
        is not counted in the calculator below.
      </p>
    ),
  },
  {
    title: 'Train locally',
    active: ['train', 'data'],
    body: (
      <p>
        Each client fine-tunes on its own private text. The base weights <strong>W</strong> never change; only the low-rank adapters{' '}
        <strong>A</strong> and <strong>B</strong> are trained.
      </p>
    ),
  },
  {
    title: 'Upload adapters',
    active: ['upload'],
    body: <p>Clients send only their adapter weights to the server. The text they trained on stays on the device.</p>,
  },
  {
    title: 'Aggregate',
    active: ['aggregate'],
    body: (
      <p>
        The server combines the adapters into one global adapter. How it combines them matters: averaging A and B separately is not the same as
        averaging each client's product B·A.
      </p>
    ),
  },
  {
    title: 'Send back',
    active: ['download'],
    body: <p>The server sends the new global adapter back to the clients, and the next round begins.</p>,
  },
];

function parseForm(form: LoraForm): { input: LoraCalcInput | null; errors: Partial<Record<keyof LoraForm, string>> } {
  const errors: Partial<Record<keyof LoraForm, string>> = {};
  const values: Partial<Record<NumericKey, number>> = {};
  for (const key of Object.keys(LIMITS) as NumericKey[]) {
    const [n, err] = parseNumber(form[key], { ...LIMITS[key], integer: true });
    if (err) errors[key] = err;
    else values[key] = n;
  }
  if (Object.keys(errors).length) return { input: null, errors };
  return {
    input: {
      d_in: values.d_in!,
      d_out: values.d_out!,
      rank: values.rank!,
      clients: values.clients!,
      rounds: values.rounds!,
      bytes_per_param: Number(form.bytes_per_param) as 1 | 2 | 4,
    },
    errors,
  };
}

function sameInputs(a: LoraCalcInput, b: LoraCalcInput) {
  return (Object.keys(a) as (keyof LoraCalcInput)[]).every((k) => a[k] === b[k]);
}

export default function LanguageModelsPage() {
  const [form, setForm] = useModuleState<LoraForm>('lm.form', DEFAULT_FORM);
  const [showErrors, setShowErrors] = useModuleState('lm.showErrors', false);
  const { state, data, run, reset } = useApiAction('lm.calc', calculateLora);

  const { input, errors } = parseForm(form);
  const visibleErrors = showErrors ? errors : {};
  const stale = data && input && !sameInputs(data.inputs, input);

  const update = (key: keyof LoraForm) => (value: string) => setForm((f) => ({ ...f, [key]: value }));

  const submit = () => {
    setShowErrors(true);
    if (input) void run(input);
  };

  const shapes = data?.inputs ?? input ?? { d_in: 4096, d_out: 4096, rank: 8 };

  return (
    <div className="page">
      <PageHeader
        eyebrow="Module 1 · Language models"
        accent="var(--accent-language-models)"
        title="Federated fine-tuning with LoRA"
        lead="Large language models have billions of weights. LoRA freezes them and trains two small matrices instead, so each client only needs to send a tiny update."
      />

      <div className="grid-2" style={{ alignItems: 'start' }}>
        <Card title="The idea in three parts">
          <div className="prose">
            <p>
              <strong>Frozen base model.</strong> Every client starts from the same pretrained model. Its weights stay fixed.
            </p>
            <p>
              <strong>Trainable adapters.</strong> For a weight matrix W, LoRA learns two thin matrices B and A. Their product B·A is a small correction
              added to W.
            </p>
            <p>
              <strong>Federated rounds.</strong> Clients train adapters on private text. The server aggregates them and sends the result back.
            </p>
          </div>
          <Details summary="Technical details">
            <p>
              The adapted layer computes <code>h = W·x + B·A·x</code>, where W is <code>d_out × d_in</code> (frozen), B is <code>d_out × r</code>, and A
              is <code>r × d_in</code>. B usually starts at zero, so training begins from the unchanged base model.
            </p>
            <p>
              <strong>Rank limit vs. actual rank.</strong> The hyperparameter r is an upper bound: B·A can have rank at most r. The learned product may
              have a lower actual rank if some directions turn out unnecessary.
            </p>
            <p>
              <strong>Why aggregation is subtle.</strong> If clients send (A<sub>k</sub>, B<sub>k</sub>), averaging A's and B's separately gives
              mean(B)·mean(A), which is generally different from mean(B<sub>k</sub>·A<sub>k</sub>). FLoRIST-style methods address this at the server
              and use singular value thresholding to pick a compact rank. A real implementation must follow the official paper and code.
            </p>
          </Details>
        </Card>

        <Card title="Walk through one round" subtitle="Click a step or use Next">
          <StepThrough steps={STEPS} title="Federated LoRA fine-tuning" labels={{ data: 'Private text', serverDetail: 'Aggregates adapters' }} />
        </Card>
      </div>

      <Section title="Communication calculator" intro="Change the layer size, rank and federation settings, then calculate how much each client sends. The backend does the calculation.">
        <div className="workbench">
          <Card title="Settings" as="div">
            <form
              className="form"
              noValidate
              onSubmit={(e) => {
                e.preventDefault();
                submit();
              }}
            >
              <div className="grid-2" style={{ gap: 'var(--space-4)' }}>
                <NumberField label="Input dim." unit="d_in" value={form.d_in} onChange={update('d_in')} min={1} max={16384} error={visibleErrors.d_in} />
                <NumberField label="Output dim." unit="d_out" value={form.d_out} onChange={update('d_out')} min={1} max={16384} error={visibleErrors.d_out} />
              </div>
              <NumberField
                label="LoRA rank limit"
                unit="r"
                value={form.rank}
                onChange={update('rank')}
                min={1}
                max={1024}
                error={visibleErrors.rank}
                hint="Must be at most the smaller dimension."
              />
              <NumberField label="Participating clients" value={form.clients} onChange={update('clients')} min={1} max={1000} error={visibleErrors.clients} />
              <NumberField label="Communication rounds" value={form.rounds} onChange={update('rounds')} min={1} max={1000} error={visibleErrors.rounds} />
              <div className="field">
                <label className="field-label" htmlFor="lm-bytes">
                  Bytes per parameter
                </label>
                <select id="lm-bytes" className="select" value={form.bytes_per_param} onChange={(e) => update('bytes_per_param')(e.target.value)}>
                  <option value="4">4 bytes · fp32</option>
                  <option value="2">2 bytes · fp16 / bf16</option>
                  <option value="1">1 byte · int8</option>
                </select>
              </div>
              <div className="btn-row">
                <button type="submit" className="btn btn-primary" disabled={state.status === 'loading'}>
                  {state.status === 'loading' ? 'Calculating…' : 'Calculate'}
                </button>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => {
                    setForm(DEFAULT_FORM);
                    setShowErrors(false);
                    reset();
                  }}
                >
                  Reset
                </button>
              </div>
            </form>
          </Card>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-5)', minWidth: 0 }}>
            <Card title="Matrix shapes" subtitle="What is frozen and what is trained" as="div">
              <MatrixShapes dIn={shapes.d_in} dOut={shapes.d_out} rank={Math.min(shapes.rank, shapes.d_in, shapes.d_out)} />
              <ul className="legend">
                <li>
                  <span className="swatch" style={{ background: 'var(--color-border)', border: '1px solid var(--color-chart-gray)' }} /> Frozen, never sent
                </li>
                <li>
                  <span className="swatch" style={{ background: 'var(--color-yellow)', border: '1px solid var(--color-ink)' }} /> Trained and exchanged
                </li>
              </ul>
            </Card>

            <Card title="Results" actions={data && <ModeBadge mode={data.mode} />} as="div">
              {state.status === 'loading' && <LoadingState />}
              {state.status === 'error' && <ApiErrorState error={state.error} onRetry={submit} />}
              {state.status === 'idle' && !data && <EmptyState>Press Calculate to see parameter counts and data volumes.</EmptyState>}
              {stale && state.status !== 'loading' && (
                <StatusMessage tone="warning">Settings changed since the last calculation. Press Calculate to update.</StatusMessage>
              )}
              {data && (
                <>
                  <SavingsBars results={data.results} />
                  <StatGrid
                    stats={[
                      { label: 'Adapter parameters', value: formatInt(data.results.adapter_params), note: `A: ${formatInt(data.results.a_params)} · B: ${formatInt(data.results.b_params)}` },
                      { label: 'Fewer parameters than W', value: `${formatNumber(data.results.savings_ratio, 1)}×` },
                      {
                        label: 'Update per client, per round',
                        value: formatBytes(data.results.per_client_update_bytes),
                        note: `vs. ${formatBytes(data.results.full_update_bytes)} for full W`,
                      },
                      { label: 'Total upload (all clients, all rounds)', value: formatBytes(data.results.total_upload_bytes) },
                      { label: 'Total download', value: formatBytes(data.results.total_download_bytes) },
                      { label: 'Upload + download', value: formatBytes(data.results.total_bytes) },
                    ]}
                  />
                  <p className="small muted">Bytes use decimal units: 1 MB = 1,000,000 bytes.</p>
                  <AssumptionsList assumptions={data.assumptions} />
                </>
              )}
            </Card>
          </div>
        </div>
      </Section>

      <Quiz
        question="In federated LoRA fine-tuning, which parameters does each client train and send to the server?"
        options={[
          { id: 'full', label: 'All weights of the base model', feedback: 'The base model stays frozen. Sending it every round is exactly the cost LoRA avoids.' },
          {
            id: 'adapters',
            label: 'Only the low-rank adapter matrices A and B',
            correct: true,
            feedback: 'Right. Clients train and exchange only A and B; W stays frozen and identical on every client.',
          },
          { id: 'data', label: 'Their private training text', feedback: 'Raw data never leaves the client in federated learning. Only model updates are shared.' },
          { id: 'both', label: 'A, B and the updated W', feedback: 'W is never updated in LoRA, so there is nothing to send for it.' },
        ]}
        explanation="Because A and B together are r·(d_in + d_out) numbers, the update grows with the rank instead of with d_in × d_out."
      />

      <ExtensionCard title="Future: singular value thresholding & measured performance">
        <p>
          Owner: Language-models teammate. Planned additions behind <code>backend/app/modules/language_models/service.py</code>:
        </p>
        <ul>
          <li>FLoRIST-style server aggregation with singular value thresholding, following the official paper and code.</li>
          <li>Measured model quality (e.g. validation loss) from real fine-tuning runs, run in a separate worker.</li>
        </ul>
      </ExtensionCard>
    </div>
  );
}
