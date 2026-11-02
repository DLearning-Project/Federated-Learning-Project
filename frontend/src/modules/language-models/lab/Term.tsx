import { useCallback, useId, useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { useLab } from './LabContext';
import { Rich, toPlain } from './Math';

/** Gap between the term and its tooltip, and the minimum margin to the viewport edges (px). */
const GAP = 6;
const MARGIN = 8;

/**
 * A dotted-underlined term. Hover or keyboard focus shows a one-sentence
 * definition; the ? button opens the full explanation in the drawer.
 *
 * - termId: a LoRA/FLoRIST term from paper_facts.json (definition + drawer).
 * - definition: a plain definition for general FL terms (client, round, α …),
 *   which are taught in the teammates' modules, so they have no drawer here.
 *
 * The tooltip is rendered into document.body with fixed positioning, so cards
 * with overflow: hidden cannot clip it. It is measured after rendering and
 * placed below the term, flipped above when there is no room, and shifted
 * horizontally to stay inside the viewport.
 */
export function Term({ children, termId, definition, htmlFor }: { children: ReactNode; termId?: string; definition?: string; htmlFor?: string }) {
  const { term, openTerm } = useLab();
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);
  const anchorRef = useRef<HTMLSpanElement>(null);
  const tipRef = useRef<HTMLSpanElement>(null);
  const tipId = useId();
  const t = termId ? term(termId) : undefined;
  const text = definition ?? t?.short;
  // Accessible name: the glossary label, or the visible text without TeX markup.
  const label = toPlain(t?.label ?? (typeof children === 'string' ? children : 'this term'));

  const place = useCallback(() => {
    const anchor = anchorRef.current?.getBoundingClientRect();
    const tip = tipRef.current?.getBoundingClientRect();
    if (!anchor || !tip) return;
    const vw = document.documentElement.clientWidth;
    const vh = window.innerHeight;
    const left = Math.max(MARGIN, Math.min(anchor.left, vw - tip.width - MARGIN));
    let top = anchor.bottom + GAP;
    if (top + tip.height > vh - MARGIN && anchor.top - GAP - tip.height >= MARGIN) top = anchor.top - GAP - tip.height;
    top = Math.max(MARGIN, Math.min(top, vh - tip.height - MARGIN));
    setPos({ left, top });
  }, []);

  useLayoutEffect(() => {
    if (!open) {
      setPos(null);
      return;
    }
    place();
    window.addEventListener('resize', place);
    window.addEventListener('scroll', place, true);
    return () => {
      window.removeEventListener('resize', place);
      window.removeEventListener('scroll', place, true);
    };
  }, [open, place, text]);

  const show = () => setOpen(true);
  const hide = () => setOpen(false);
  const Tag = htmlFor ? 'label' : 'span';

  return (
    <span className="lm-term" ref={anchorRef} onMouseLeave={hide}>
      <Tag
        className="lm-term-label"
        htmlFor={htmlFor}
        tabIndex={htmlFor ? undefined : 0}
        aria-describedby={open && text ? tipId : undefined}
        onMouseEnter={show}
        onFocus={show}
        onBlur={hide}
        onKeyDown={(e) => e.key === 'Escape' && hide()}
      >
        {typeof children === 'string' ? <Rich text={children} /> : children}
      </Tag>
      {termId && (
        <button type="button" className="lm-help" aria-label={`More about ${label}`} onClick={() => openTerm(termId)} onFocus={hide}>
          ?
        </button>
      )}
      {open &&
        text &&
        createPortal(
          <span
            ref={tipRef}
            role="tooltip"
            id={tipId}
            className="lm-tooltip"
            style={pos ? { left: pos.left, top: pos.top } : { left: 0, top: 0, visibility: 'hidden' }}
          >
            <Rich text={text} />
          </span>,
          document.body,
        )}
    </span>
  );
}
