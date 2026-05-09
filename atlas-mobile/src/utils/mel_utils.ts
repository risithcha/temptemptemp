/**
 * Atlas Mobile - Lightweight Mel-Band Energy Extractor
 *
 * Computes a compact array of log Mel-band energies directly from the
 * FFT magnitude buffer already populated by the alarm detector's polling
 * loop (`analyser.getByteFrequencyData()` -> Uint8Array, values 0-255).
 *
 * Design choices:
 *  - Triangular filter bank pre-computed ONCE at module load time -> zero
 *    marginal cost at poll time (only ~375 multiply-adds per 100 ms frame).
 *  - Log Mel-band *energies* (not DCT'd MFCCs): the DCT step exists to
 *    decorrelate inputs for GMM classifiers; our Euclidean distance
 *    clustering has no such requirement, and the DCT is a lossy linear
 *    transform that would discard discriminative cross-band correlation.
 *  - Sparse filter bank representation: only non-zero {bin, weight} pairs
 *    are stored, so iteration cost stays O(non-zero entries) not O(numBins).
 *  - Power spectrum: each byte magnitude is squared before weighting,
 *    matching the standard Mel filter bank derivation (energy = amplitude^2).
 *  - Epsilon floor prevents log(0) on fully-silent frames.
 *
 * Mel scale: mel = 2595 * log10(1 + f/700)
 *            f   = 700 * (10^(mel/2595) - 1)
 *
 * Atlas FFT parameters: ALARM_FFT_SIZE=2048, sampleRate=44100
 *   -> numBins = 1024, binWidth ~= 21.53 Hz/bin
 */

import { ALARM_FFT_SIZE, NUM_MEL_BANDS } from '../theme';

// ---------------------------------------------------------------------------
// Private constants (internal to this module)
// ---------------------------------------------------------------------------

/** Audio sample rate must match the AudioContext used by the alarm detector. */
const SAMPLE_RATE = 44100;

/** Number of usable FFT magnitude bins (Nyquist = half of fftSize). */
const NUM_BINS = ALARM_FFT_SIZE / 2; // 1024

/** Frequency resolution per bin (Hz). */
const BIN_HZ = SAMPLE_RATE / ALARM_FFT_SIZE; // ~=21.533 Hz

/** Lower frequency boundary for the filter bank - excludes sub-vocal rumble. */
const MEL_LOW_HZ = 80;

/** Upper frequency boundary - covers the third formant region for most voices. */
const MEL_HIGH_HZ = 3500;

// ---------------------------------------------------------------------------
// Internal sparse filter bank type
// ---------------------------------------------------------------------------

interface FilterEntry {
  /** FFT bin index (0 to NUM_BINS - 1). */
  bin: number;
  /** Triangular filter weight at this bin (0 < weight <= 1). */
  weight: number;
}

/** Sparse Mel filter bank: one array of {bin, weight} entries per band. */
type FilterBank = FilterEntry[][];

// ---------------------------------------------------------------------------
// Mel scale conversion helpers
// ---------------------------------------------------------------------------

function hzToMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel: number): number {
  return 700 * (Math.pow(10, mel / 2595) - 1);
}

// ---------------------------------------------------------------------------
// Filter bank construction (called once at module load)
// ---------------------------------------------------------------------------

/**
 * Build a sparse triangular Mel filter bank.
 *
 * Creates `numBands + 2` Mel-uniformly-spaced frequency points between
 * `lowHz` and `highHz`, converts them to FFT bin indices, then constructs
 * a rising + falling triangular response for each of the `numBands` filters.
 *
 * Only entries with weight > 0 are stored (sparse representation).
 *
 * @param numBands  Number of triangular filters (= number of output features).
 * @param lowHz     Left edge of the lowest filter (Hz).
 * @param highHz    Right edge of the highest filter (Hz).
 * @returns         Sparse filter bank: `numBands` arrays of {bin, weight}.
 */
function buildMelFilterBank(
  numBands: number,
  lowHz: number,
  highHz: number,
): FilterBank {
  const lowMel = hzToMel(lowHz);
  const highMel = hzToMel(highHz);

  // (numBands + 2) points equally spaced in Mel: edges + numBands centres.
  const melPoints: number[] = [];
  for (let i = 0; i <= numBands + 1; i++) {
    melPoints.push(lowMel + (i / (numBands + 1)) * (highMel - lowMel));
  }

  // Convert Mel points -> Hz -> nearest FFT bin index (floor).
  const binPoints = melPoints.map((m) => Math.floor(melToHz(m) / BIN_HZ));

  const bank: FilterBank = [];

  for (let m = 1; m <= numBands; m++) {
    const leftBin   = binPoints[m - 1];
    const centerBin = binPoints[m];
    const rightBin  = binPoints[m + 1];

    const entries: FilterEntry[] = [];

    // Rising slope: leftBin (exclusive) -> centerBin (exclusive)
    const riseWidth = Math.max(centerBin - leftBin, 1);
    for (let k = leftBin + 1; k < centerBin; k++) {
      if (k < 0 || k >= NUM_BINS) continue;
      const weight = (k - leftBin) / riseWidth;
      entries.push({ bin: k, weight });
    }

    // Peak: centerBin at weight 1.0
    if (centerBin >= 0 && centerBin < NUM_BINS) {
      entries.push({ bin: centerBin, weight: 1.0 });
    }

    // Falling slope: centerBin (exclusive) -> rightBin (exclusive)
    const fallWidth = Math.max(rightBin - centerBin, 1);
    for (let k = centerBin + 1; k < rightBin; k++) {
      if (k < 0 || k >= NUM_BINS) continue;
      const weight = (rightBin - k) / fallWidth;
      entries.push({ bin: k, weight });
    }

    bank.push(entries);
  }

  return bank;
}

// ---------------------------------------------------------------------------
// Module-level pre-computed filter bank
// ---------------------------------------------------------------------------

/**
 * Triangular Mel filter bank pre-computed at import time.
 *
 * Parameters match the Atlas alarm detector's FFT configuration:
 *   ALARM_FFT_SIZE=2048, sampleRate=44100, NUM_MEL_BANDS=5
 *   -> bands covering ~80 Hz - 3500 Hz with logarithmic spacing.
 *
 * Approximate band centre frequencies:
 *   Band 0: ~330 Hz  (fundamental + lower harmonics, male F0 region)
 *   Band 1: ~670 Hz  (upper male F0 / first formant boundary)
 *   Band 2: ~1110 Hz (first formant for many vowels)
 *   Band 3: ~1700 Hz (second formant, vowel identity)
 *   Band 4: ~2470 Hz (third formant / voice quality / fricative energy)
 *
 * Exported so callers can pass it explicitly
 */
export const MEL_FILTER_BANK: FilterBank = buildMelFilterBank(
  NUM_MEL_BANDS,
  MEL_LOW_HZ,
  MEL_HIGH_HZ,
);

// ---------------------------------------------------------------------------
// Feature extraction - hot path (called every 100 ms)
// ---------------------------------------------------------------------------

/**
 * Compute log Mel-band energies from a byte-frequency FFT magnitude buffer.
 *
 * Each output element is:
 *   `ln( sum_k( buffer[k]^2 * H_m(k) ) + eps )`
 *
 * where H_m(k) is the triangular filter weight for band m at bin k, and
 * eps = 1e-8 prevents log(0) on silent frames.
 *
 * **Performance:** With NUM_MEL_BANDS=5 and the 80-3500 Hz range over
 * 1024 bins, the sparse bank contains ~375 non-zero entries total, so
 * this function performs ~375 FMA operations -- well under the 100 ms
 * poll budget.
 *
 * @param buffer      `Uint8Array` from `analyser.getByteFrequencyData()`.
 *                    Values are 0-255, proportional to spectral magnitude.
 * @param filterBank  Pre-computed filter bank (defaults to module-level
 *                    `MEL_FILTER_BANK`; pass a custom bank for testing).
 * @returns           `Float32Array` of `numBands` log-energy values.
 *                    Typical range on voiced speech: roughly 1.0 - 9.0.
 *                    Silent / unvoiced bands approach ln(1e-8) ~= -18.4.
 */
export function computeMelBandEnergies(
  buffer: Uint8Array,
  filterBank: FilterBank = MEL_FILTER_BANK,
): Float32Array {
  const numBands = filterBank.length;
  const result = new Float32Array(numBands);

  for (let m = 0; m < numBands; m++) {
    let bandEnergy = 0;
    const entries = filterBank[m];
    for (let i = 0; i < entries.length; i++) {
      const v = buffer[entries[i].bin];
      bandEnergy += v * v * entries[i].weight; // power = amplitude^2
    }
    result[m] = Math.log(bandEnergy + 1e-8);
  }

  return result;
}
