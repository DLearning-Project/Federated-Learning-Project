import { Link } from 'react-router-dom';
import { Card, PageHeader, Section } from '../components/Card';
import { FederatedDiagram } from '../components/FederatedDiagram';
import { ArrowRightIcon } from '../components/Icons';
import { StatusMessage } from '../components/StatusMessage';
import { MODULES } from './modules';

export function HomePage() {
  return (
    <div className="page">
      <PageHeader
        eyebrow="Course project · Federated learning"
        title="Learn how models train together without pooling data"
        lead="Federated learning trains one shared model across many devices or organisations. Each participant learns from its own data, and only model updates are sent to a central server."
      />

      <div className="grid-2" style={{ alignItems: 'start' }}>
        <Card title="How one round works">
          <ol className="prose" style={{ gap: 'var(--space-2)' }}>
            <li>The server sends the current model to a group of clients.</li>
            <li>Each client trains on its own local data for a short while.</li>
            <li>Clients send back their model <em>updates</em>, not their raw data.</li>
            <li>The server combines the updates into a new global model and repeats.</li>
          </ol>
          <StatusMessage tone="warning" title="Local data is not automatic privacy">
            Model updates can still leak information about the data they were trained on. Real systems add protections such as secure aggregation or
            differential privacy. This explorer does not implement those protections.
          </StatusMessage>
        </Card>
        <Card title="Data stays, updates travel" subtitle="Simplified view of one federated round">
          <FederatedDiagram title="Three clients keep their data locally and exchange model updates with a central server" />
        </Card>
      </div>

      <Section title="Explore the three modules" intro="Each module is built by one team member and can be explored independently, in any order.">
        <div className="grid-3">
          {MODULES.map((m) => (
            <Link key={m.id} to={m.path} className="card module-card">
              <span className="accent-line" style={{ background: m.accent }} aria-hidden />
              <h3>{m.title}</h3>
              <p className="muted">{m.summary}</p>
              <ul className="legend" aria-label="Topics">
                {m.topics.map((t) => (
                  <li key={t} className="badge">
                    {t}
                  </li>
                ))}
              </ul>
              <span className="card-link">
                Open module <ArrowRightIcon />
              </span>
            </Link>
          ))}
        </div>
      </Section>

      <Card variant="inset" title="About this MVP">
        <p className="small">
          All numbers are <strong>illustrative</strong>. Calculations run on a small FastAPI backend, and no real model training, dataset downloads or GPU
          work happens yet. Your settings are kept while you move between pages, but reset when you refresh.
        </p>
      </Card>
    </div>
  );
}
