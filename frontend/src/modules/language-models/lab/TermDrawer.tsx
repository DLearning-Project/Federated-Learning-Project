import { useEffect, useRef } from 'react';
import { CloseIcon } from '../../../components/Icons';
import { ApiErrorState, LoadingState } from '../../../components/StatusMessage';
import { useLab } from './LabContext';
import { MathBlock, Rich, toPlain } from './Math';

/** Side drawer with a term's paper explanation, or the full glossary. */
export function TermDrawer() {
  const { drawer, closeDrawer, openTerm, openGlossary, glossary, term, level } = useLab();
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!drawer) return;
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && closeDrawer();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [drawer, closeDrawer]);

  if (!drawer) return null;
  const t = drawer.kind === 'term' ? term(drawer.id) : undefined;
  const title = drawer.kind === 'glossary' ? 'Glossary' : t?.label ?? 'Term';
  const showLong = level !== 'beginner';
  const showAnalogy = level === 'beginner' || level === null;

  return (
    <div role="dialog" aria-modal="false" aria-label={toPlain(title)} className="lm-drawer">
      <div className="lm-drawer-head">
        <h2>
          <Rich text={title} />
        </h2>
        <button ref={closeRef} type="button" className="btn btn-secondary lm-icon-btn" aria-label="Close" onClick={closeDrawer}>
          <CloseIcon />
        </button>
      </div>

      {glossary.status === 'loading' && <LoadingState label="Loading terms…" />}
      {glossary.status === 'error' && <ApiErrorState error={glossary.error} />}

      {glossary.status === 'ready' && drawer.kind === 'glossary' && (
        <>
          <p className="muted small">Every LoRA and FLoRIST term in this module, with definitions from the FLoRIST paper.</p>
          <ul className="lm-glossary">
            {glossary.terms.map((g) => (
              <li key={g.id}>
                <button type="button" onClick={() => openTerm(g.id)}>
                  <strong>
                    <Rich text={g.label} />
                  </strong>
                  <span className="muted small">
                    <Rich text={g.short} />
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </>
      )}

      {glossary.status === 'ready' && drawer.kind === 'term' && !t && <p>This term is not in the glossary.</p>}
      {t && (
        <>
          <span className="lm-chip">Paper: {t.reference}</span>
          <div>
            <div className="lm-kicker">In short</div>
            <p>
              <Rich text={level ? t.levels[level] : t.short} />
            </p>
          </div>
          {showLong && (
            <div>
              <div className="lm-kicker">From the paper</div>
              <p>
                <Rich text={t.long} />
              </p>
            </div>
          )}
          <MathBlock tex={t.equation} label={`Equation: ${t.equation_plain}`} />
          {showAnalogy && t.analogy && (
            <div className="lm-analogy">
              <div className="lm-kicker">Analogy</div>
              <p>
                <Rich text={t.analogy} />
              </p>
            </div>
          )}
          <button type="button" className="btn btn-ghost lm-link-btn" onClick={openGlossary}>
            Browse all terms →
          </button>
        </>
      )}
    </div>
  );
}
