/** Shared API client: base URL, JSON handling and typed errors for every module. */

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');
export const API_PREFIX = '/api/v1';

export type ApiErrorKind = 'network' | 'validation' | 'not_found' | 'server';

export class ApiError extends Error {
  constructor(
    public kind: ApiErrorKind,
    message: string,
    public status?: number,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

/** Matches backend/app/schemas/common.py CalcEnvelope. */
export interface CalcEnvelope<I, R> {
  mode: 'illustrative' | 'measured';
  inputs: I;
  results: R;
  units: Record<string, string>;
  assumptions: string[];
}

const REQUEST_TIMEOUT_MS = 10_000;

async function request<T>(path: string, init: RequestInit = {}, signal?: AbortSignal): Promise<T> {
  const timeout = AbortSignal.timeout(REQUEST_TIMEOUT_MS);
  const combined = signal ? anySignal([signal, timeout]) : timeout;

  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${API_PREFIX}${path}`, {
      ...init,
      signal: combined,
      headers: { 'Content-Type': 'application/json', ...init.headers },
    });
  } catch (err) {
    if (signal?.aborted) throw err; // Caller cancelled: let them ignore it.
    throw new ApiError(
      'network',
      `Cannot reach the backend at ${API_BASE_URL}. Make sure the FastAPI server is running, then retry.`,
    );
  }

  if (res.ok) return (await res.json()) as T;

  let detail = res.statusText;
  try {
    const body = await res.json();
    if (typeof body.detail === 'string') detail = body.detail;
  } catch {
    /* non-JSON error body */
  }
  if (res.status === 422) throw new ApiError('validation', detail, res.status);
  if (res.status === 404) throw new ApiError('not_found', detail, res.status);
  throw new ApiError('server', `The backend returned an error (${res.status}): ${detail}`, res.status);
}

function anySignal(signals: AbortSignal[]): AbortSignal {
  const controller = new AbortController();
  for (const s of signals) {
    if (s.aborted) {
      controller.abort(s.reason);
      break;
    }
    s.addEventListener('abort', () => controller.abort(s.reason), { once: true });
  }
  return controller.signal;
}

export function apiGet<T>(path: string, signal?: AbortSignal): Promise<T> {
  return request<T>(path, { method: 'GET' }, signal);
}

export function apiPost<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  return request<T>(path, { method: 'POST', body: JSON.stringify(body) }, signal);
}

export function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError';
}
