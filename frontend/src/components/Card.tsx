import type { ReactNode } from 'react';

interface CardProps {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  variant?: 'default' | 'inset' | 'extension';
  as?: 'section' | 'div' | 'aside';
  className?: string;
  children?: ReactNode;
  headingLevel?: 2 | 3;
}

export function Card({ title, subtitle, actions, variant = 'default', as: Tag = 'section', className = '', children, headingLevel = 3 }: CardProps) {
  const Heading = headingLevel === 2 ? 'h2' : 'h3';
  const variantClass = variant === 'default' ? '' : variant;
  return (
    <Tag className={`card ${variantClass} ${className}`.trim()}>
      {(title || actions) && (
        <div className="card-header">
          <div>
            {title && <Heading style={headingLevel === 2 ? { fontSize: 'var(--text-card-heading)' } : undefined}>{title}</Heading>}
            {subtitle && <p className="card-subtitle">{subtitle}</p>}
          </div>
          {actions}
        </div>
      )}
      {children}
    </Tag>
  );
}

export function PageHeader({ eyebrow, title, lead, accent }: { eyebrow?: ReactNode; title: ReactNode; lead?: ReactNode; accent?: string }) {
  return (
    <header className="page-header">
      {eyebrow && (
        <span className="eyebrow">
          {accent && <span className="swatch" style={{ background: accent, width: '1.5rem', height: '3px', borderRadius: 2 }} />}
          {eyebrow}
        </span>
      )}
      <h1>{title}</h1>
      {lead && <p className="lead">{lead}</p>}
    </header>
  );
}

export function Section({ title, intro, children }: { title: ReactNode; intro?: ReactNode; children: ReactNode }) {
  return (
    <section className="section">
      <div className="section-header">
        <h2>{title}</h2>
        {intro && <p className="muted">{intro}</p>}
      </div>
      {children}
    </section>
  );
}

/** Dashed card marking where a teammate will plug in real models or data later. */
export function ExtensionCard({ title, children }: { title: string; children: ReactNode }) {
  return (
    <Card variant="extension" as="aside" title={title} actions={<span className="badge">Extension point · not implemented</span>}>
      <div className="prose small">{children}</div>
    </Card>
  );
}
