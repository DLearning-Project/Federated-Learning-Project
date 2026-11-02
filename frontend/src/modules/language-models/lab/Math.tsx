import katex from 'katex';
import 'katex/dist/katex.min.css';
import { Fragment, useLayoutEffect, useRef, type ReactNode } from 'react';
import { Details } from '../../../components/Details';

/**
 * KaTeX rendering for the language-models module.
 *
 * - <M tex="…" />          inline math inside a sentence
 * - <MathBlock tex="…" />  centered display math for key equations
 * - <Rich text="…" />      text with $inline$ and $$display$$ math segments
 *
 * Text from the backend (glossary, assumptions, notices) uses the same $…$
 * convention, so one renderer covers all notation. Output is HTML plus MathML,
 * so screen readers read the formula. Invalid TeX is shown in place (no throw).
 */

const cache = new Map<string, string>();

function render(tex: string, display: boolean): string {
  const key = `${display ? 'D' : 'I'}${tex}`;
  let html = cache.get(key);
  if (html === undefined) {
    html = katex.renderToString(tex, { displayMode: display, throwOnError: false, output: 'htmlAndMathml', strict: 'ignore' });
    if (cache.size > 2000) cache.clear();
    cache.set(key, html);
  }
  return html;
}

export function M({ tex }: { tex: string }) {
  return <span className="lm-math-inline" dangerouslySetInnerHTML={{ __html: render(tex, false) }} />;
}

/** Smallest scale a display equation may shrink to before it scrolls instead. */
const MIN_SCALE = 0.7;

/**
 * Centered display equation.
 *
 * It is typeset in display style but in KaTeX's inline mode, which lets a long
 * formula wrap onto a new line after a relation or operator (=, +, ≤ …) on
 * narrow screens. If one unbreakable piece is still wider than the container,
 * the equation shrinks to fit, down to MIN_SCALE; beyond that it scrolls
 * sideways inside its own box rather than being clipped.
 */
export function MathBlock({ tex, label }: { tex: string; label?: string }) {
  const boxRef = useRef<HTMLDivElement>(null);
  const innerRef = useRef<HTMLSpanElement>(null);

  useLayoutEffect(() => {
    const box = boxRef.current;
    const inner = innerRef.current;
    if (!box || !inner) return;
    let lastWidth = -1;
    const fit = () => {
      const width = box.clientWidth;
      if (width <= 0 || width === lastWidth) return;
      lastWidth = width;
      inner.style.fontSize = '';
      const style = getComputedStyle(box);
      const avail = width - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
      // KaTeX does not shrink exactly in proportion, so re-measure a few times.
      let scale = 1;
      for (let pass = 0; pass < 4; pass++) {
        const over = box.scrollWidth - width;
        if (over <= 1 || scale <= MIN_SCALE) break;
        scale = Math.max(MIN_SCALE, scale * (avail / (avail + over)) - 0.005);
        inner.style.fontSize = `${scale}em`;
      }
    };
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(box);
    // KaTeX's fonts can finish loading after the first measurement, which makes
    // the formula wider without resizing the box, so measure again then.
    let live = true;
    document.fonts?.ready.then(() => {
      if (!live) return;
      lastWidth = -1;
      fit();
    });
    return () => {
      live = false;
      ro.disconnect();
    };
  }, [tex]);

  return (
    <div ref={boxRef} className="lm-math-display" role={label ? 'group' : undefined} aria-label={label} tabIndex={0}>
      <span ref={innerRef} className="lm-math-fit" dangerouslySetInnerHTML={{ __html: render(`\\displaystyle ${tex}`, false) }} />
    </div>
  );
}

const SEGMENT = /(\$\$[\s\S]+?\$\$|\$[^$]+?\$)/g;
const LEADING = /[([{“‘"']+$/; // opening punctuation right before a formula
const TRAILING = /^[)\]}”’"'.,;:!?%]+/; // closing punctuation right after a formula

/**
 * Renders a string whose math is marked with $…$ (inline) or $$…$$ (display).
 * Punctuation touching an inline formula, as in "(rank $p$).", is kept on the
 * same line as the formula so a line never starts with ")" or ends with "(".
 */
export function Rich({ text }: { text: string }) {
  if (!text.includes('$')) return <>{text}</>;
  const parts = text.split(SEGMENT);
  const out: ReactNode[] = [];
  for (let i = 0; i < parts.length; i++) {
    const part = parts[i];
    const isDisplay = part.startsWith('$$') && part.endsWith('$$') && part.length > 4;
    const isInline = !isDisplay && part.startsWith('$') && part.endsWith('$') && part.length > 2;
    if (isDisplay) {
      out.push(<MathBlock key={i} tex={part.slice(2, -2)} />);
    } else if (isInline) {
      const prev = out.length && typeof out[out.length - 1] === 'string' ? (out[out.length - 1] as string) : '';
      const lead = prev.match(LEADING)?.[0] ?? '';
      if (lead) out[out.length - 1] = prev.slice(0, -lead.length);
      const next = parts[i + 1] ?? '';
      const trail = next.match(TRAILING)?.[0] ?? '';
      if (trail) parts[i + 1] = next.slice(trail.length);
      out.push(
        lead || trail ? (
          <span key={i} className="lm-nowrap">
            {lead}
            <M tex={part.slice(1, -1)} />
            {trail}
          </span>
        ) : (
          <M key={i} tex={part.slice(1, -1)} />
        ),
      );
    } else if (part) {
      out.push(part);
    }
  }
  return (
    <>
      {out.map((node, k) => (typeof node === 'string' ? <Fragment key={`t${k}`}>{node}</Fragment> : node))}
    </>
  );
}

const PLAIN: Record<string, string> = {
  '\\Delta': 'Δ',
  '\\Sigma': 'Σ',
  '\\sum': 'Σ',
  '\\alpha': 'α',
  '\\sigma': 'σ',
  '\\tau': 'τ',
  '\\times': '×',
  '\\le': '≤',
  '\\ge': '≥',
};

/** Plain-text version of $…$-marked text, for aria-labels and native title tooltips. */
export function toPlain(text: string): string {
  return text
    .replace(/\$\$?([^$]+)\$\$?/g, (_, tex: string) =>
      tex
        .replace(/\\[a-zA-Z]+/g, (cmd) => PLAIN[cmd] ?? '')
        .replace(/[{}]/g, '')
        .replace(/\s+/g, ' ')
        .trim(),
    )
    .trim();
}

/** Renders a value that may be a $…$-marked string or any other React node. */
export function rich(node: ReactNode): ReactNode {
  return typeof node === 'string' ? <Rich text={node} /> : node;
}

/** The team's assumptions panel, with math in the items rendered by KaTeX. */
export function MathAssumptions({ assumptions, title = 'Assumptions' }: { assumptions: string[]; title?: string }) {
  return (
    <Details summary={title} defaultOpen>
      <ul className="assumptions">
        {assumptions.map((a) => (
          <li key={a}>
            <Rich text={a} />
          </li>
        ))}
      </ul>
    </Details>
  );
}
