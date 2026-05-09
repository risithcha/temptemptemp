/**
 * Atlas Mobile - Autocorrelation Pitch Estimator (Zero Dependencies)
 *
 * Estimates the fundamental frequency (F0) of a voice from raw PCM
 * time-domain samples using the autocorrelation method, and returns a
 * harmonicity confidence score derived from the normalized peak
 * autocorrelation (bestCorr / r0).
 *
 * Called every 100 ms inside the alarm detector's polling loop.
 * For a 2048-sample buffer this completes in < 2 ms on modern phones.
 */

const SILENCE_RMS_THRESHOLD = 0.0005;

export interface PitchResult {
  hz: number;
  /** Harmonicity / periodicity score in [0, 1].  1.0 = pure tone, 0.0 = noise. */
  confidence: number;
}

/**
 * Estimate the fundamental frequency of the signal in the buffer.
 *
 * @returns Pitch in Hz plus a confidence score, or `null` if the buffer
 *          is silent or no positive autocorrelation peak is found.
 */
export function estimatePitch(
  buffer: Float32Array,
  sampleRate: number,
  minHz: number = 75,
  maxHz: number = 400,
): PitchResult | null {
  const n = buffer.length;
  if (n === 0) return null;

  // --- Energy gate: skip silent frames ---
  let energy = 0;
  for (let i = 0; i < n; i++) energy += buffer[i] * buffer[i];
  if (Math.sqrt(energy / n) < SILENCE_RMS_THRESHOLD) return null;

  const minLag = Math.floor(sampleRate / maxHz);
  const maxLag = Math.min(Math.ceil(sampleRate / minHz), n - 1);

  // --- Autocorrelation over the valid lag range ---
  let bestCorr = 0;
  let bestLag = -1;

  for (let lag = minLag; lag <= maxLag; lag++) {
    let sum = 0;
    const limit = n - lag;
    for (let i = 0; i < limit; i++) {
      sum += buffer[i] * buffer[i + lag];
    }
    const norm = sum / limit;
    if (norm > bestCorr) {
      bestCorr = norm;
      bestLag = lag;
    }
  }

  if (bestLag < 0 || bestCorr <= 0) return null;

  // --- Confidence: normalized peak autocorrelation ---
  // r(0) = energy / n is the autocorrelation at lag 0.
  // confidence = bestCorr / r(0) measures the fraction of signal energy
  // explained by the detected periodicity.  Voiced speech: 0.5-0.95.
  // Aperiodic noise (clacks, clicks): < 0.2.
  const r0 = energy / n;
  const confidence = Math.min(bestCorr / r0, 1.0);

  // --- Parabolic interpolation for sub-sample accuracy ---
  if (bestLag > minLag && bestLag < maxLag) {
    const corrPrev = autocorrAt(buffer, bestLag - 1);
    const corrNext = autocorrAt(buffer, bestLag + 1);
    const shift = (corrPrev - corrNext) / (2 * (corrPrev - 2 * bestCorr * (n - bestLag) + corrNext));
    if (Number.isFinite(shift)) {
      const refinedLag = bestLag + shift;
      return { hz: sampleRate / refinedLag, confidence };
    }
  }

  return { hz: sampleRate / bestLag, confidence };
}

function autocorrAt(buffer: Float32Array, lag: number): number {
  const n = buffer.length;
  const limit = n - lag;
  let sum = 0;
  for (let i = 0; i < limit; i++) sum += buffer[i] * buffer[i + lag];
  return sum;
}
