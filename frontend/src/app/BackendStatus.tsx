import { useCallback, useEffect, useState } from 'react';
import { API_BASE_URL, apiGet } from '../services/api';

type Health = 'checking' | 'ok' | 'down';

/** Small indicator showing whether the FastAPI backend responds, with a manual re-check. */
export function BackendStatus() {
  const [health, setHealth] = useState<Health>('checking');

  const check = useCallback(async () => {
    setHealth('checking');
    try {
      await apiGet<{ status: string }>('/health');
      setHealth('ok');
    } catch {
      setHealth('down');
    }
  }, []);

  useEffect(() => {
    void check();
  }, [check]);

  const text = { checking: 'Checking backend…', ok: 'Backend connected', down: 'Backend offline' }[health];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)', alignItems: 'flex-start' }}>
      <span className="pill" role="status" title={API_BASE_URL}>
        <span className={`pill-dot ${health}`} aria-hidden />
        {text}
      </span>
      {health === 'down' && (
        <button type="button" className="btn btn-ghost small" style={{ minHeight: '2.75rem' }} onClick={check}>
          Check again
        </button>
      )}
    </div>
  );
}
