import type { Level } from '../types';
import { useLab, type PageNumber } from './LabContext';

export const PAGES: { n: PageNumber; title: string }[] = [
  { n: 1, title: 'LoRA basics' },
  { n: 2, title: 'The aggregation problem' },
  { n: 3, title: 'FLoRIST step-by-step' },
  { n: 4, title: 'Simulator' },
];

export const LEVELS: { id: Level; label: string; blurb: string }[] = [
  { id: 'beginner', label: 'Beginner', blurb: 'Plain language and analogies. No math needed.' },
  { id: 'student', label: 'Student', blurb: 'Intuition plus the key equations, on request.' },
  { id: 'researcher', label: 'Researcher', blurb: 'Full notation, paper references and all the numbers.' },
];

/** Module toolbar: the four-page learning path, audience level, and glossary. */
export function LabToolbar() {
  const { page, setPage, level, setLevel, openGlossary } = useLab();
  const current = PAGES.find((p) => p.n === page)!;
  return (
    <div className="lm-toolbar">
      <nav aria-label="FLoRIST Lab pages" className="lm-path">
        <ol>
          {PAGES.map((p) => (
            <li key={p.n}>
              <button
                type="button"
                className={`lm-path-step ${p.n === page ? 'current' : ''} ${p.n < page ? 'done' : ''}`}
                aria-current={p.n === page ? 'page' : undefined}
                onClick={() => setPage(p.n)}
              >
                <span className="lm-path-num" aria-hidden>
                  {p.n}
                </span>
                <span className="lm-path-title">{p.title}</span>
              </button>
            </li>
          ))}
        </ol>
        <span className="small muted lm-path-where">
          Page {page} of 4 · {current.title}
        </span>
      </nav>
      <div className="lm-toolbar-right">
        <div className="segmented" role="group" aria-label="Explanation level">
          {LEVELS.map((l) => (
            <button key={l.id} type="button" aria-pressed={level === l.id} onClick={() => setLevel(l.id)}>
              {l.label}
            </button>
          ))}
        </div>
        <button type="button" className="btn btn-secondary" onClick={openGlossary}>
          Glossary
        </button>
      </div>
    </div>
  );
}
