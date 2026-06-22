import { useEffect, useState } from 'react';

const PROBE_URL = 'https://clients3.google.com/generate_204';
const DEFAULT_POLL_MS = 5000;
const PROBE_TIMEOUT_MS = 4000;

/**
 * Best-effort internet reachability (null = still probing).
 * Lightweight HTTP probe — no extra native NetInfo dependency.
 */
export function useNetworkReachable(pollMs = DEFAULT_POLL_MS): boolean | null {
  const [reachable, setReachable] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;

    const probe = async () => {
      try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), PROBE_TIMEOUT_MS);
        await fetch(PROBE_URL, {
          method: 'GET',
          signal: controller.signal,
          cache: 'no-store',
        });
        clearTimeout(timer);
        if (!cancelled) setReachable(true);
      } catch {
        if (!cancelled) setReachable(false);
      }
    };

    void probe();
    const id = setInterval(probe, pollMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [pollMs]);

  return reachable;
}

export function isLikelyNetworkError(message: string): boolean {
  const lower = message.toLowerCase();
  return (
    lower.includes('network request failed') ||
    lower.includes('failed to fetch') ||
    lower.includes('network error') ||
    lower.includes('internet') ||
    lower.includes('abort') ||
    lower.includes('timeout') ||
    lower.includes('enotfound') ||
    lower.includes('unable to resolve host')
  );
}
