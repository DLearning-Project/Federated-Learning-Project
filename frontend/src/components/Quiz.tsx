import { useId, useState, type ReactNode } from 'react';
import { Card } from './Card';
import { StatusMessage } from './StatusMessage';

export interface QuizOption {
  id: string;
  label: ReactNode;
  correct?: boolean;
  /** Why this particular answer is right or wrong. */
  feedback: ReactNode;
}

/** Single-choice knowledge check with explanatory feedback. Runs fully in the browser. */
export function Quiz({ question, options, explanation }: { question: ReactNode; options: QuizOption[]; explanation?: ReactNode }) {
  const name = useId();
  const [selected, setSelected] = useState<string | null>(null);
  const [checked, setChecked] = useState(false);
  const choice = options.find((o) => o.id === selected);

  return (
    <Card title="Knowledge check" subtitle="Pick one answer, then check it.">
      <form
        className="form"
        onSubmit={(e) => {
          e.preventDefault();
          if (selected) setChecked(true);
        }}
      >
        <fieldset className="quiz-options">
          <legend style={{ fontWeight: 600, marginBottom: 'var(--space-3)', lineHeight: 1.5 }}>{question}</legend>
          {options.map((o) => {
            const state = checked && o.id === selected ? (o.correct ? 'correct' : 'incorrect') : '';
            return (
              <label key={o.id} className={`quiz-option ${state}`}>
                <input
                  type="radio"
                  name={name}
                  value={o.id}
                  checked={selected === o.id}
                  onChange={() => {
                    setSelected(o.id);
                    setChecked(false);
                  }}
                />
                <span>{o.label}</span>
              </label>
            );
          })}
        </fieldset>

        {checked && choice && (
          <StatusMessage tone={choice.correct ? 'success' : 'error'} title={choice.correct ? 'Correct' : 'Not quite'}>
            <p>{choice.feedback}</p>
            {choice.correct && explanation && <p style={{ marginTop: 'var(--space-2)' }}>{explanation}</p>}
          </StatusMessage>
        )}

        <div className="btn-row">
          <button type="submit" className="btn btn-primary" disabled={!selected}>
            Check answer
          </button>
          {(selected || checked) && (
            <button
              type="button"
              className="btn btn-ghost"
              onClick={() => {
                setSelected(null);
                setChecked(false);
              }}
            >
              Reset
            </button>
          )}
        </div>
      </form>
    </Card>
  );
}
