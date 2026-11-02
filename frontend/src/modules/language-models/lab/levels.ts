import type { Level } from '../types';

/**
 * Picks the text (or anything else) for the current audience level.
 * Researcher falls back to the student version when it is not given.
 */
export function byLevel<T>(level: Level, v: { beginner: T; student: T; researcher?: T }): T {
  if (level === 'beginner') return v.beginner;
  if (level === 'researcher' && v.researcher !== undefined) return v.researcher;
  return v.student;
}
