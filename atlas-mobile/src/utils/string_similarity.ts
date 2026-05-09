/**
 * Atlas Mobile - String Similarity (Zero Dependencies)
 *
 * Provides Levenshtein edit distance and a normalized 0-1 similarity
 * metric.  Used by the OCR Smart Memory system to decide whether newly
 * extracted text is "substantially the same" as what was already read.
 *
 * The algorithm uses a single-row Wagner-Fischer DP approach: O(m*n) time,
 * O(min(m,n)) space.  For typical OCR strings (< 500 chars) this runs
 * well under 1 ms on modern phones.
 */

export function levenshtein(a: string, b: string): number {
  if (a === b) return 0;
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;

  // Ensure `a` is the shorter string so the DP row is minimal.
  if (a.length > b.length) [a, b] = [b, a];

  const m = a.length;
  const n = b.length;
  const dp = new Array<number>(m + 1);

  for (let i = 0; i <= m; i++) dp[i] = i;

  for (let j = 1; j <= n; j++) {
    let prev = dp[0];
    dp[0] = j;
    for (let i = 1; i <= m; i++) {
      const temp = dp[i];
      dp[i] =
        a[i - 1] === b[j - 1]
          ? prev
          : 1 + Math.min(prev, dp[i], dp[i - 1]);
      prev = temp;
    }
  }

  return dp[m];
}

/**
 * Normalized similarity in [0, 1].
 * 1.0 = identical strings, 0.0 = completely different.
 */
export function similarity(a: string, b: string): number {
  const maxLen = Math.max(a.length, b.length);
  if (maxLen === 0) return 1.0;
  return 1 - levenshtein(a, b) / maxLen;
}
