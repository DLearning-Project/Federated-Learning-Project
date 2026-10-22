import { createContext, useCallback, useContext, useRef, useState, type ReactNode } from 'react';

/**
 * Temporary, in-memory state shared across navigation.
 * Each module stores its settings and results under its own key.
 * Everything resets on page refresh: there is no persistence by design.
 */
type Store = Record<string, unknown>;

interface AppState {
  store: Store;
  set: (key: string, updater: (prev: unknown) => unknown) => void;
}

const AppStateContext = createContext<AppState | null>(null);

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [store, setStore] = useState<Store>({});
  const set = useCallback((key: string, updater: (prev: unknown) => unknown) => {
    setStore((s) => ({ ...s, [key]: updater(s[key]) }));
  }, []);
  return <AppStateContext.Provider value={{ store, set }}>{children}</AppStateContext.Provider>;
}

/** Like useState, but the value survives navigating between pages. */
export function useModuleState<T>(key: string, initial: T): [T, (next: T | ((prev: T) => T)) => void] {
  const ctx = useContext(AppStateContext);
  if (!ctx) throw new Error('useModuleState must be used inside AppStateProvider');
  const initialRef = useRef(initial);
  const value = (key in ctx.store ? ctx.store[key] : initialRef.current) as T;
  const { set } = ctx;

  const setValue = useCallback(
    (next: T | ((prev: T) => T)) => {
      set(key, (prev) => {
        const current = (prev === undefined ? initialRef.current : prev) as T;
        return typeof next === 'function' ? (next as (p: T) => T)(current) : next;
      });
    },
    [key, set],
  );

  return [value, setValue];
}
