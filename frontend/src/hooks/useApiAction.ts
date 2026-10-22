import { useCallback, useRef } from 'react';
import { useModuleState } from '../app/AppStateContext';
import { ApiError, isAbortError } from '../services/api';

export type ActionState<T> =
  | { status: 'idle' }
  | { status: 'loading'; requestId: number; data?: T }
  | { status: 'success'; requestId: number; data: T }
  | { status: 'error'; requestId: number; error: ApiError; data?: T };

let nextRequestId = 1;

/**
 * Runs an API call on demand (e.g. a Calculate button) and keeps the result in shared
 * module state so it survives navigation. Starting a new run cancels the previous
 * request, and late responses from older requests are ignored.
 */
export function useApiAction<I, O>(key: string, fn: (input: I, signal: AbortSignal) => Promise<O>) {
  const [state, setState] = useModuleState<ActionState<O>>(key, { status: 'idle' });
  const controllerRef = useRef<AbortController | null>(null);
  const fnRef = useRef(fn);
  fnRef.current = fn;

  // Requests are deliberately not aborted on unmount: if the user navigates away
  // mid-request, the response still lands in shared state for when they return.

  const run = useCallback(
    async (input: I) => {
      controllerRef.current?.abort();
      const controller = new AbortController();
      controllerRef.current = controller;
      const requestId = nextRequestId++;
      setState((prev) => ({ status: 'loading', requestId, data: 'data' in prev ? prev.data : undefined }));

      const settle = (next: ActionState<O>) =>
        setState((prev) => ('requestId' in prev && prev.requestId !== requestId ? prev : next));

      try {
        const data = await fnRef.current(input, controller.signal);
        settle({ status: 'success', requestId, data });
      } catch (err) {
        // Superseded by a newer request or reset: the newer state already won.
        if (isAbortError(err)) return;
        const error = err instanceof ApiError ? err : new ApiError('server', String(err));
        setState((prev) =>
          'requestId' in prev && prev.requestId !== requestId
            ? prev
            : { status: 'error', requestId, error, data: 'data' in prev ? prev.data : undefined },
        );
      }
    },
    [setState],
  );

  const reset = useCallback(() => {
    controllerRef.current?.abort();
    setState({ status: 'idle' });
  }, [setState]);

  const data = 'data' in state ? state.data : undefined;
  return { state, data, run, reset };
}
