import { useEffect } from 'react';
import { useModuleState } from '../../app/AppStateContext';
import { Card, ExtensionCard, PageHeader, Section } from '../../components/Card';
import { Details } from '../../components/Details';
import { formatNumber, formatPercent, formatSeconds } from '../../components/format';
import { CheckIcon, CrossIcon, StarIcon } from '../../components/Icons';
import { NumberField, parseNumber } from '../../components/NumberField';
import { Quiz } from '../../components/Quiz';
import { AssumptionsList, Meter, ModeBadge, StatGrid } from '../../components/Results';
import { ApiErrorState, EmptyState, LoadingState, StatusMessage } from '../../components/StatusMessage';
import { useApiAction } from '../../hooks/useApiAction';
import { estimateResources, getDeviceProfiles } from './api';
import type { HardwareEstimateInput, HardwareEstimateResult, HardwareForm } from './types';

const DEFAULT_FORM: HardwareForm = {
  profile_id: 'phone',
  update_mb: '20',
  bandwidth_mbps: '20',
  local_epochs: '3',
  energy_budget_wh: '0.5',
  time_budget_s: '600',
};

const CHALLENGE_FORM: HardwareForm = {
  profile_id: 'phone',
  update_mb: '50',
  bandwidth_mbps: '5',
  local_epochs: '10',
  energy_budget_wh: '0.5',
  time_budget_s: '900',
};

const LESSON = [
  {
    title: 'Local training',
    body: 'Each round, a device trains on its own data for a few local epochs. More epochs means more computation, so more time and more battery.',
  },
  {
    title: 'Communication',
    body: 'After training, the device uploads its model update and later downloads the new global model. Large updates over slow links take a long time, and the radio uses power while it transmits.',
  },
  {
    title: 'Device constraints',
    body: 'Phones and small boards have limited energy, compute and bandwidth. A good federated setup picks update sizes and epoch counts that each device can actually afford.',
  },
];

function parseForm(form: HardwareForm): { input: HardwareEstimateInput | null; errors: Partial<Record<keyof HardwareForm, string>> } {
  const errors: Partial<Record<keyof HardwareForm, string>> = {};
  const [update, e1] = parseNumber(form.update_mb, { min: 0.001, max: 10_000, label: 'Update size' });
  const [bw, e2] = parseNumber(form.bandwidth_mbps, { min: 0.001, max: 10_000, label: 'Bandwidth' });
  const [epochs, e3] = parseNumber(form.local_epochs, { min: 1, max: 100, integer: true, label: 'Local epochs' });
  const [energy, e4] = parseNumber(form.energy_budget_wh, { min: 0.001, max: 1000, label: 'Energy budget' });
  let time: number | null = null;
  if (form.time_budget_s.trim() !== '') {
    const [t, e5] = parseNumber(form.time_budget_s, { min: 1, max: 86_400, label: 'Time budget' });
    if (e5) errors.time_budget_s = e5;
    time = t;
  }
  if (e1) errors.update_mb = e1;
  if (e2) errors.bandwidth_mbps = e2;
  if (e3) errors.local_epochs = e3;
  if (e4) errors.energy_budget_wh = e4;
  if (Object.keys(errors).length) return { input: null, errors };
  return {
    input: { profile_id: form.profile_id, update_mb: update, bandwidth_mbps: bw, local_epochs: epochs, energy_budget_wh: energy, time_budget_s: time },
    errors,
  };
}

function sameInputs(a: HardwareEstimateInput, b: HardwareEstimateInput) {
  return (Object.keys(a) as (keyof HardwareEstimateInput)[]).every((k) => a[k] === b[k]);
}

function BudgetStatus({ ok, used }: { ok: boolean; used: number }) {
  return ok ? (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 'var(--space-2)', color: 'var(--color-success)' }}>
      <CheckIcon /> Within budget ({formatPercent(used, 0)} used)
    </span>
  ) : (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 'var(--space-2)', color: 'var(--color-error)' }}>
      <CrossIcon /> Over budget ({formatPercent(used, 0)} of budget)
    </span>
  );
}

function ResourceBars({ inputs, r }: { inputs: HardwareEstimateInput; r: HardwareEstimateResult }) {
  const energyMax = Math.max(r.total_energy_wh, inputs.energy_budget_wh) * 1.1;
  const timeBudget = inputs.time_budget_s;
  const timeMax = Math.max(r.total_time_s, timeBudget ?? 0) * 1.1;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
      <Meter
        label="Energy per round"
        max={energyMax}
        limit={inputs.energy_budget_wh}
        valueText={`${formatNumber(r.total_energy_wh, 3)} Wh of ${formatNumber(inputs.energy_budget_wh, 3)} Wh budget`}
        segments={[
          { label: 'Training energy', value: r.training_energy_wh, color: 'var(--color-ink)' },
          { label: 'Communication energy', value: r.communication_energy_wh, color: 'var(--color-orange)' },
        ]}
        statusText={<BudgetStatus ok={r.meets_energy_budget} used={r.energy_budget_used} />}
      />
      <Meter
        label="Time per round"
        max={timeMax}
        limit={timeBudget ?? undefined}
        valueText={timeBudget ? `${formatSeconds(r.total_time_s)} of ${formatSeconds(timeBudget)} budget` : formatSeconds(r.total_time_s)}
        segments={[
          { label: 'Training time', value: r.training_time_s, color: 'var(--color-ink)' },
          { label: 'Communication time', value: r.communication_time_s, color: 'var(--color-orange)' },
        ]}
        statusText={r.meets_time_budget === null || r.time_budget_used === null ? <span className="muted">No time budget set.</span> : <BudgetStatus ok={r.meets_time_budget} used={r.time_budget_used} />}
      />
      <ul className="legend">
        <li>
          <span className="swatch" style={{ background: 'var(--color-ink)' }} /> Training
        </li>
        <li>
          <span className="swatch" style={{ background: 'var(--color-orange)' }} /> Communication
        </li>
        <li>
          <span className="swatch" style={{ width: 2, background: 'var(--color-ink)' }} /> Budget limit
        </li>
      </ul>
    </div>
  );
}

export default function HardwarePage() {
  const [lessonStep, setLessonStep] = useModuleState('hw.lesson', 0);
  const [form, setForm] = useModuleState<HardwareForm>('hw.form', DEFAULT_FORM);
  const [showErrors, setShowErrors] = useModuleState('hw.showErrors', false);
  const [challengeOn, setChallengeOn] = useModuleState('hw.challenge', false);
  const profiles = useApiAction('hw.profiles', getDeviceProfiles);
  const estimate = useApiAction('hw.estimate', estimateResources);

  const loadProfiles = profiles.run;
  useEffect(() => {
    if (profiles.state.status === 'idle') void loadProfiles(undefined);
  }, [profiles.state.status, loadProfiles]);

  const { input, errors } = parseForm(form);
  const visibleErrors = showErrors ? errors : {};
  const data = estimate.data;
  const stale = data && input && !sameInputs(data.inputs, input);
  const profileList = profiles.data?.profiles ?? [];
  const selected = profileList.find((p) => p.id === form.profile_id);

  const challengeDone =
    challengeOn &&
    !!data &&
    !stale &&
    data.inputs.profile_id === 'phone' &&
    data.inputs.update_mb === 50 &&
    data.inputs.energy_budget_wh === 0.5 &&
    data.results.meets_energy_budget;

  const update = (key: keyof HardwareForm) => (value: string) => setForm((f) => ({ ...f, [key]: value }));
  const submit = () => {
    setShowErrors(true);
    if (input) void estimate.run(input);
  };

  return (
    <div className="page">
      <PageHeader
        eyebrow="Module 3 · Hardware-aware learning"
        accent="var(--accent-hardware)"
        title="Can this device afford a training round?"
        lead="Federated clients are real devices with batteries, slow links and small processors. This module estimates the time and energy one round costs on illustrative device profiles."
      />

      <Card title="Short guided lesson">
        <div className="segmented" role="tablist" aria-label="Lesson steps">
          {LESSON.map((l, i) => (
            <button key={l.title} type="button" role="tab" id={`hw-tab-${i}`} aria-selected={lessonStep === i} aria-controls="hw-lesson-panel" onClick={() => setLessonStep(i)}>
              {i + 1}. {l.title}
            </button>
          ))}
        </div>
        <div className="progress" aria-hidden>
          <span style={{ width: `${((lessonStep + 1) / LESSON.length) * 100}%` }} />
        </div>
        <div id="hw-lesson-panel" role="tabpanel" aria-labelledby={`hw-tab-${lessonStep}`} className="prose">
          <p>{LESSON[lessonStep].body}</p>
        </div>
        <div className="btn-row">
          <button type="button" className="btn btn-secondary" disabled={lessonStep === 0} onClick={() => setLessonStep(lessonStep - 1)}>
            Previous
          </button>
          <button type="button" className="btn btn-primary" disabled={lessonStep === LESSON.length - 1} onClick={() => setLessonStep(lessonStep + 1)}>
            Next
          </button>
        </div>
        <Details summary="Formulas and units">
          <ul className="assumptions">
            <li>
              Upload time = download time = update size (MB) × 8 ÷ bandwidth (Mbps), in seconds.
            </li>
            <li>Training time = local epochs × seconds per epoch (from the device profile).</li>
            <li>Energy (Wh) = (training time × training power + communication time × radio power) ÷ 3600.</li>
          </ul>
          <p className="small muted">The backend applies these formulas; the page only displays the results.</p>
        </Details>
      </Card>

      <Section title="Resource estimator" intro="Choose a device and a round configuration, then estimate whether it fits the budget. All profiles are illustrative, not calibrated measurements.">
        <Card title="Device profile" as="div">
          {profiles.state.status === 'loading' && <LoadingState label="Loading device profiles…" />}
          {profiles.state.status === 'error' && <ApiErrorState error={profiles.state.error} onRetry={() => void loadProfiles(undefined)} />}
          {profiles.state.status === 'success' && profileList.length === 0 && <EmptyState>No device profiles are available.</EmptyState>}
          {profileList.length > 0 && (
            <>
              <div className="choice-list" role="radiogroup" aria-label="Device profile">
                {profileList.map((p) => (
                  <button key={p.id} type="button" role="radio" aria-checked={form.profile_id === p.id} className="choice" onClick={() => update('profile_id')(p.id)}>
                    <span className="choice-title">
                      {p.name}
                      {form.profile_id === p.id && <CheckIcon />}
                    </span>
                    <span className="small muted">{p.description}</span>
                    <span className="small">
                      {p.seconds_per_epoch} s/epoch · {p.train_power_w} W train · {p.radio_power_w} W radio
                    </span>
                  </button>
                ))}
              </div>
              <p className="small muted">{profiles.data?.note}</p>
            </>
          )}
        </Card>

        <div className="workbench">
          <Card title="Round settings" as="div">
            <form
              className="form"
              noValidate
              onSubmit={(e) => {
                e.preventDefault();
                submit();
              }}
            >
              <NumberField label="Model update size" unit="MB" value={form.update_mb} onChange={update('update_mb')} step="any" min={0} error={visibleErrors.update_mb} />
              <NumberField
                label="Bandwidth"
                unit="Mbps"
                value={form.bandwidth_mbps}
                onChange={update('bandwidth_mbps')}
                step="any"
                min={0}
                error={visibleErrors.bandwidth_mbps}
                hint={selected ? `Same speed assumed for upload and download. Typical for this profile: ${selected.typical_bandwidth_mbps} Mbps.` : 'Same speed assumed for upload and download.'}
              />
              <NumberField label="Local epochs" value={form.local_epochs} onChange={update('local_epochs')} min={1} max={100} error={visibleErrors.local_epochs} />
              <NumberField label="Energy budget per round" unit="Wh" value={form.energy_budget_wh} onChange={update('energy_budget_wh')} step="any" min={0} error={visibleErrors.energy_budget_wh} />
              <NumberField
                label="Time budget per round"
                unit="seconds, optional"
                value={form.time_budget_s}
                onChange={update('time_budget_s')}
                min={1}
                error={visibleErrors.time_budget_s}
                hint="Leave empty to skip the time check."
              />
              <div className="btn-row">
                <button type="submit" className="btn btn-primary" disabled={estimate.state.status === 'loading' || !selected}>
                  {estimate.state.status === 'loading' ? 'Estimating…' : 'Estimate'}
                </button>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => {
                    setForm(DEFAULT_FORM);
                    setShowErrors(false);
                    setChallengeOn(false);
                    estimate.reset();
                  }}
                >
                  Reset
                </button>
              </div>
            </form>
          </Card>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-5)', minWidth: 0 }}>
            <Card title="Estimate" subtitle={data ? `${data.results.profile_name} · illustrative estimate` : undefined} actions={data && <ModeBadge mode={data.mode} />} as="div">
              {estimate.state.status === 'loading' && <LoadingState label="Estimating…" />}
              {estimate.state.status === 'error' && <ApiErrorState error={estimate.state.error} onRetry={submit} />}
              {estimate.state.status === 'idle' && !data && <EmptyState>Pick a device and press Estimate.</EmptyState>}
              {stale && estimate.state.status !== 'loading' && <StatusMessage tone="warning">Settings changed since the last estimate. Press Estimate to update.</StatusMessage>}
              {data && (
                <>
                  <ResourceBars inputs={data.inputs} r={data.results} />
                  <StatGrid
                    stats={[
                      { label: 'Training time', value: formatSeconds(data.results.training_time_s) },
                      { label: 'Communication time', value: formatSeconds(data.results.communication_time_s), note: `${formatSeconds(data.results.upload_time_s)} up + ${formatSeconds(data.results.download_time_s)} down` },
                      { label: 'Training energy', value: `${formatNumber(data.results.training_energy_wh, 3)} Wh` },
                      { label: 'Communication energy', value: `${formatNumber(data.results.communication_energy_wh, 3)} Wh`, note: 'Included in the total' },
                    ]}
                  />
                  <AssumptionsList assumptions={data.assumptions} />
                </>
              )}
            </Card>

            <Card title="Guided challenge" subtitle="Make a struggling phone fit its energy budget">
              <p>
                A smartphone must send a <strong>50 MB</strong> update over a <strong>5 Mbps</strong> link after <strong>10 local epochs</strong>, with a{' '}
                <strong>0.5 Wh</strong> energy budget. Load the scenario, estimate it, then change <em>only</em> the local epochs or the bandwidth until it
                fits.
              </p>
              {challengeDone ? (
                <div className="achievement" role="status">
                  <StarIcon width={28} height={28} />
                  <span>
                    Challenge complete! {formatNumber(data!.results.total_energy_wh, 3)} Wh fits within 0.5 Wh. Training dominates energy here, so fewer
                    epochs help most. A faster link mostly saves time.
                  </span>
                </div>
              ) : (
                challengeOn && (
                  <StatusMessage tone="info">
                    Challenge active. Keep the phone, 50 MB and 0.5 Wh fixed. Hint: compare the training and communication segments in the energy bar.
                  </StatusMessage>
                )
              )}
              <div className="btn-row">
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => {
                    setForm(CHALLENGE_FORM);
                    setChallengeOn(true);
                    setShowErrors(false);
                  }}
                >
                  {challengeOn ? 'Restart challenge' : 'Load challenge'}
                </button>
              </div>
            </Card>
          </div>
        </div>
      </Section>

      <Quiz
        question="A phone is over its energy budget, and almost all of the energy comes from training. What is the most effective change?"
        options={[
          {
            id: 'epochs',
            label: 'Reduce the number of local epochs.',
            correct: true,
            feedback: 'Right. Training energy grows with local epochs, so fewer epochs cut the largest part of the cost.',
          },
          { id: 'bandwidth', label: 'Use a faster network connection.', feedback: 'A faster link shortens communication, but that was only a small part of the energy here.' },
          { id: 'budget', label: 'Ignore it; the estimate predicts accuracy, not energy.', feedback: 'The estimate is about time and energy only. It makes no claim about accuracy.' },
          { id: 'server', label: 'Upgrade the server.', feedback: 'The server does not change how much energy the phone spends training locally.' },
        ]}
        explanation="Always look at which part dominates (training or communication) before deciding what to optimise."
      />

      <ExtensionCard title="Future: real measurements and a calibrated digital twin">
        <p>
          Owner: Hardware teammate. Planned additions behind <code>backend/app/modules/hardware/service.py</code>:
        </p>
        <ul>
          <li>Importing measured epoch times, power draw and link throughput from real devices.</li>
          <li>Calibrating profiles against those measurements to build a device digital twin.</li>
          <li>Until then, every profile and output here is an illustrative estimate.</li>
        </ul>
      </ExtensionCard>
    </div>
  );
}
