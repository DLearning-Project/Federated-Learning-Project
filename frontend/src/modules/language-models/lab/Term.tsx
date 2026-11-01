import { useId, useState, type ReactNode } from 'react';
import { useLab } from './LabContext';

/**
 * A dotted-underlined term. Hover or keyboard focus shows a one-sentence
 * definition; the ? button opens the full explanation in the drawer.
 *
 * - termId: a LoRA/FLoRIST term from paper_facts.json (definition + drawer).
 * - definition: a plain definition for general FL terms (client, round, α …),
 *   which are taught in the teammates' modules, so they have no drawer here.
 */
export function Term({
  children,
  termId,
  definition,
  htmlFor,
  align = 'left',
}: {
  children: ReactNode;
  termId?: string;
  definition?: string;
  htmlFor?: string;
  align?: 'left' | 'right';
}) {
  const { term, openTerm } = useLab();
  const [open, setOpen] = useState(false);
  const tipId = useId();
  const t = termId ? term(termId) : undefined;
  const text = definition ?? t?.short;
  const label = typeof children === 'string' ? children : t?.label ?? 'this term';

  const show = () => setOpen(true);
  const hide = () => setOpen(false);
  const Tag = htmlFor ? 'label' : 'span';

  return (
    <span className="lm-term" onMouseLeave={hide}>
      <Tag
        className="lm-term-label"
        htmlFor={htmlFor}
        tabIndex={htmlFor ? undefined : 0}
        aria-describedby={open && text ? tipId : undefined}
        onMouseEnter={show}
        onFocus={show}
        onBlur={hide}
      >
        {children}
      </Tag>
      {termId && (
        <button type="button" className="lm-help" aria-label={`More about ${label}`} onClick={() => openTerm(termId)} onFocus={hide}>
          ?
        </button>
      )}
      {open && text && (
        <span role="tooltip" id={tipId} className={`lm-tooltip ${align === 'right' ? 'right' : ''}`}>
          {text}
        </span>
      )}
    </span>
  );
}
