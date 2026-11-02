import { useEffect, useRef } from 'react';
import { useLab } from './LabContext';
import { LEVELS } from './LabToolbar';

/** First-visit dialog: pick how much detail explanations should have. */
export function WelcomeModal() {
  const { setLevel } = useLab();
  const firstRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    firstRef.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setLevel('student');
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [setLevel]);

  return (
    <div className="lm-backdrop">
      <div role="dialog" aria-modal="true" aria-labelledby="lm-welcome-title" className="lm-modal">
        <span className="eyebrow">FLoRIST Learning Lab</span>
        <h2 id="lm-welcome-title">How much detail would you like?</h2>
        <p className="muted">
          This module explains how language models are fine-tuned together with LoRA, and how FLoRIST combines the clients'
          adapters. You can change the level at any time from the toolbar.
        </p>
        <div className="lm-level-cards">
          {LEVELS.map((l, i) => (
            <button key={l.id} ref={i === 0 ? firstRef : undefined} type="button" className="lm-level-card" onClick={() => setLevel(l.id)}>
              <strong>{l.label}</strong>
              <span className="small muted">{l.blurb}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
