import type { ReactNode } from 'react';
import type { ApiError } from '../services/api';
import { AlertIcon, CheckIcon, CrossIcon, InfoIcon } from './Icons';

type Tone = 'info' | 'success' | 'error' | 'warning';

const icons = { info: InfoIcon, success: CheckIcon, error: CrossIcon, warning: AlertIcon };

export function StatusMessage({ tone = 'info', title, children, action }: { tone?: Tone; title?: ReactNode; children?: ReactNode; action?: ReactNode }) {
  const Icon = icons[tone];
  return (
    <div className={`status ${tone}`} role={tone === 'error' ? 'alert' : 'status'}>
      <Icon className="status-icon" />
      <div className="status-body">
        {title && <span className="status-title">{title}</span>}
        {children && <div>{children}</div>}
        {action && <div>{action}</div>}
      </div>
    </div>
  );
}

export function LoadingState({ label = 'Calculating…' }: { label?: string }) {
  return (
    <div className="status" role="status" aria-live="polite">
      <span className="spinner" aria-hidden />
      <span>{label}</span>
    </div>
  );
}

export function EmptyState({ children }: { children: ReactNode }) {
  return <StatusMessage tone="info">{children}</StatusMessage>;
}

/** Explains an API failure and, for connection problems, offers a retry. */
export function ApiErrorState({ error, onRetry }: { error: ApiError; onRetry?: () => void }) {
  const titles = {
    network: 'Backend not reachable',
    validation: 'Please check your inputs',
    not_found: 'Not found',
    server: 'Something went wrong on the server',
  } as const;
  return (
    <StatusMessage
      tone="error"
      title={titles[error.kind]}
      action={
        onRetry && error.kind !== 'validation' ? (
          <button type="button" className="btn btn-secondary" onClick={onRetry}>
            Retry
          </button>
        ) : undefined
      }
    >
      {error.message}
      {error.kind === 'network' && (
        <p className="small muted" style={{ marginTop: 'var(--space-1)' }}>
          Lessons, diagrams and quizzes on this page still work without the backend.
        </p>
      )}
    </StatusMessage>
  );
}
