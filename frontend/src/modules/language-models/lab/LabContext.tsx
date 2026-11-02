import { createContext, useContext, useEffect, type ReactNode } from 'react';
import { useModuleState } from '../../../app/AppStateContext';
import { ApiError, isAbortError } from '../../../services/api';
import { fetchGlossary } from '../api';
import type { GlossaryTerm, Level } from '../types';

/**
 * Module-local state for the FLoRIST Lab: audience level, the open term drawer,
 * the current page of the learning path, and the glossary (loaded once from the
 * backend's paper_facts.json, the single source of truth for term text).
 *
 * These are module-local placeholders for the shared level/term/glossary pieces
 * the team may build later (README_FL step 7).
 */

export type PageNumber = 1 | 2 | 3 | 4;
export type DrawerTarget = { kind: 'glossary' } | { kind: 'term'; id: string } | null;
type GlossaryState = { status: 'loading' } | { status: 'ready'; terms: GlossaryTerm[] } | { status: 'error'; error: ApiError };

interface LabValue {
  level: Level | null;
  setLevel: (level: Level) => void;
  page: PageNumber;
  setPage: (page: PageNumber) => void;
  drawer: DrawerTarget;
  openTerm: (id: string) => void;
  openGlossary: () => void;
  closeDrawer: () => void;
  glossary: GlossaryState;
  retryGlossary: () => void;
  term: (id: string) => GlossaryTerm | undefined;
}

const LabContext = createContext<LabValue | null>(null);

export function LabProvider({ children }: { children: ReactNode }) {
  const [level, setLevel] = useModuleState<Level | null>('lm.level', null);
  const [page, setPage] = useModuleState<PageNumber>('lm.page', 1);
  const [drawer, setDrawer] = useModuleState<DrawerTarget>('lm.drawer', null);
  const [glossary, setGlossary] = useModuleState<GlossaryState>('lm.glossary', { status: 'loading' });
  const [attempt, setAttempt] = useModuleState('lm.glossaryAttempt', 0);

  useEffect(() => {
    if (glossary.status === 'ready') return;
    const controller = new AbortController();
    setGlossary({ status: 'loading' });
    fetchGlossary(controller.signal)
      .then((terms) => setGlossary({ status: 'ready', terms }))
      .catch((err) => {
        if (isAbortError(err)) return;
        setGlossary({ status: 'error', error: err instanceof ApiError ? err : new ApiError('server', String(err)) });
      });
    return () => controller.abort();
    // Only refetch on an explicit retry.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [attempt]);

  const value: LabValue = {
    level,
    setLevel,
    page,
    setPage: (p) => {
      setPage(p);
      window.scrollTo(0, 0);
    },
    drawer,
    openTerm: (id) => setDrawer({ kind: 'term', id }),
    openGlossary: () => setDrawer({ kind: 'glossary' }),
    closeDrawer: () => setDrawer(null),
    glossary,
    retryGlossary: () => setAttempt((a) => a + 1),
    term: (id) => (glossary.status === 'ready' ? glossary.terms.find((t) => t.id === id) : undefined),
  };
  return <LabContext.Provider value={value}>{children}</LabContext.Provider>;
}

export function useLab(): LabValue {
  const ctx = useContext(LabContext);
  if (!ctx) throw new Error('useLab must be used inside LabProvider');
  return ctx;
}
