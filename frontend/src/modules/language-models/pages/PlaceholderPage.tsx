import { Card, PageHeader } from '../../../components/Card';
import { useLab } from '../lab/LabContext';
import { Rich } from '../lab/Math';

/** A page of the learning path that is planned but not built yet. */
export function PlaceholderPage({ number, title, lead, planned }: { number: 2 | 3; title: string; lead: string; planned: string[] }) {
  const { setPage } = useLab();
  return (
    <div className="page">
      <PageHeader eyebrow={`Page ${number} of 4 · Coming next`} accent="var(--accent-language-models)" title={title} lead={<Rich text={lead} />} />
      <Card title="Planned for this page">
        <ul className="lm-list">
          {planned.map((p) => (
            <li key={p}>
              <Rich text={p} />
            </li>
          ))}
        </ul>
        <p className="small muted">This page is not built yet. The Simulator (page 4) already uses the same backend calculations.</p>
        <div className="lm-row">
          <button type="button" className="btn btn-secondary" onClick={() => setPage((number - 1) as 1 | 2)}>
            ← Previous page
          </button>
          <button type="button" className="btn btn-primary" onClick={() => setPage(4)}>
            Open the Simulator
          </button>
        </div>
      </Card>
    </div>
  );
}
