/**
 * Theme switch over the token sheet. Sets `data-theme` on <html> (so exactly one
 * semantic-role sheet is live), persists the choice, and restores it on load.
 * Runtime-provable: switching changes a real computed value (see DAG/components
 * reading the semantic vars), not just a compile-time class swap.
 */
import { useCallback, useSyncExternalStore } from "react";

export type ThemeName = "dark" | "light";

const STORAGE_KEY = "ra-theme";

function current(): ThemeName {
  const attr = document.documentElement.getAttribute("data-theme");
  return attr === "dark" ? "dark" : "light";
}

function apply(theme: ThemeName): void {
  document.documentElement.setAttribute("data-theme", theme);
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    // storage unavailable (private mode) — theme still applies for this session
  }
}

/** Restore the persisted theme once at module load (before first paint of React). */
export function initTheme(): void {
  let stored: string | null = null;
  try {
    stored = localStorage.getItem(STORAGE_KEY);
  } catch {
    stored = null;
  }
  apply(stored === "dark" ? "dark" : "light");
}

const listeners = new Set<() => void>();
function subscribe(cb: () => void): () => void {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

export function useTheme(): { theme: ThemeName; toggle: () => void } {
  const theme = useSyncExternalStore(subscribe, current, () => "light" as ThemeName);
  const toggle = useCallback(() => {
    apply(current() === "dark" ? "light" : "dark");
    for (const cb of listeners) cb();
  }, []);
  return { theme, toggle };
}
