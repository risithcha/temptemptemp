/**
 * Atlas Mobile - Online (2+N)D Speaker Clustering
 *
 * Assigns speech segments to speaker profiles using a range-normalized
 * (2+N)D Euclidean distance over fundamental frequency (F0), spectral
 * centroid (timbre), and N log Mel-band energies (vocal tract shape).
 * The Mel dimensions separate voices that share similar F0 and room
 * resonance but differ in formant structure or phonetic quality.
 *
 * All features are normalized before distance calculation so no single
 * dimension dominates:
 *   - Pitch:       75-400 Hz    (range 325 Hz)
 *   - Centroid:    300-3000 Hz  (range 2700 Hz)
 *   - Mel band k:  [0, MEL_FEATURE_RANGE]  (log-energy, normalized)
 */

import {
  PITCH_MIN,
  PITCH_MAX,
  CENTROID_MIN,
  CENTROID_MAX,
  VOICE_PROFILE_TOLERANCE,
  NUM_MEL_BANDS,
  MEL_FEATURE_RANGE,
  MEL_FEATURE_WEIGHT,
  SPEAKER_STICKY_FACTOR,
} from '../theme';

const SPEAKER_LABELS = ['Speaker A', 'Speaker B', 'Speaker C', 'Speaker D'];
const MAX_SPEAKERS = SPEAKER_LABELS.length;

const PITCH_RANGE = PITCH_MAX - PITCH_MIN;
const CENTROID_RANGE = CENTROID_MAX - CENTROID_MIN;

export interface SpeakerProfile {
  label: string;
  avgPitch: number;
  avgCentroid: number;
  /** Running per-band log Mel-energy averages for (2+N)D clustering. */
  avgFeatures: number[];
  sampleCount: number;
}

export interface AssignResult {
  label: string;
  profiles: SpeakerProfile[];
}

function normalizedDistance(
  pitchA: number,
  centroidA: number,
  featuresA: number[],
  pitchB: number,
  centroidB: number,
  featuresB: number[],
): number {
  const dp = (pitchA - pitchB) / PITCH_RANGE;
  const dc = (centroidA - centroidB) / CENTROID_RANGE;
  let melSumSq = 0;
  for (let k = 0; k < NUM_MEL_BANDS; k++) {
    const df = ((featuresA[k] ?? 0) - (featuresB[k] ?? 0)) / MEL_FEATURE_RANGE;
    melSumSq += df * df;
  }
  return Math.sqrt(dp * dp + dc * dc + MEL_FEATURE_WEIGHT * melSumSq);
}

/**
 * Assign a speech segment (identified by its median pitch, median centroid,
 * and median log Mel-band energies) to the closest existing speaker profile,
 * or create a new one if no match falls within the tolerance.
 *
 * @param lastActiveSpeakerLabel  Label of the speaker who spoke most recently.
 *   When provided, that profile's raw distance is multiplied by
 *   `SPEAKER_STICKY_FACTOR` (< 1) before comparison.  This implements
 *   hysteresis: borderline segments prefer to merge with the ongoing speaker
 *   rather than creating a spurious new profile across short pauses.
 */
export function assignSpeaker(
  medianPitch: number,
  medianCentroid: number,
  medianFeatures: number[],
  profiles: SpeakerProfile[],
  tolerance: number = VOICE_PROFILE_TOLERANCE,
  lastActiveSpeakerLabel: string | null = null,
): AssignResult {
  if (profiles.length === 0) {
    const newProfile: SpeakerProfile = {
      label: SPEAKER_LABELS[0],
      avgPitch: medianPitch,
      avgCentroid: medianCentroid,
      avgFeatures: [...medianFeatures],
      sampleCount: 1,
    };
    return { label: newProfile.label, profiles: [newProfile] };
  }

  // Compute raw distance for profile 0 and apply sticky discount if applicable.
  const rawDist0 = normalizedDistance(
    medianPitch, medianCentroid, medianFeatures,
    profiles[0].avgPitch, profiles[0].avgCentroid, profiles[0].avgFeatures,
  );
  let bestIdx = 0;
  let bestDist =
    lastActiveSpeakerLabel !== null && profiles[0].label === lastActiveSpeakerLabel
      ? rawDist0 * SPEAKER_STICKY_FACTOR
      : rawDist0;

  for (let i = 1; i < profiles.length; i++) {
    const rawDist = normalizedDistance(
      medianPitch, medianCentroid, medianFeatures,
      profiles[i].avgPitch, profiles[i].avgCentroid, profiles[i].avgFeatures,
    );
    // Apply the sticky discount to the last-active speaker so borderline
    // segments merge with it rather than spawning a new profile.
    const dist =
      lastActiveSpeakerLabel !== null && profiles[i].label === lastActiveSpeakerLabel
        ? rawDist * SPEAKER_STICKY_FACTOR
        : rawDist;
    if (dist < bestDist) {
      bestDist = dist;
      bestIdx = i;
    }
  }

  if (bestDist <= tolerance) {
    const p = profiles[bestIdx];
    p.avgPitch =
      (p.avgPitch * p.sampleCount + medianPitch) / (p.sampleCount + 1);
    p.avgCentroid =
      (p.avgCentroid * p.sampleCount + medianCentroid) / (p.sampleCount + 1);
    for (let k = 0; k < NUM_MEL_BANDS; k++) {
      p.avgFeatures[k] =
        (p.avgFeatures[k] * p.sampleCount + (medianFeatures[k] ?? 0)) /
        (p.sampleCount + 1);
    }
    p.sampleCount += 1;
    return { label: p.label, profiles };
  }

  if (profiles.length < MAX_SPEAKERS) {
    const newProfile: SpeakerProfile = {
      label: SPEAKER_LABELS[profiles.length],
      avgPitch: medianPitch,
      avgCentroid: medianCentroid,
      avgFeatures: [...medianFeatures],
      sampleCount: 1,
    };
    profiles.push(newProfile);
    return { label: newProfile.label, profiles };
  }

  // Cap reached - assign to the closest anyway.
  const p = profiles[bestIdx];
  p.avgPitch =
    (p.avgPitch * p.sampleCount + medianPitch) / (p.sampleCount + 1);
  p.avgCentroid =
    (p.avgCentroid * p.sampleCount + medianCentroid) / (p.sampleCount + 1);
  for (let k = 0; k < NUM_MEL_BANDS; k++) {
    p.avgFeatures[k] =
      (p.avgFeatures[k] * p.sampleCount + (medianFeatures[k] ?? 0)) /
      (p.sampleCount + 1);
  }
  p.sampleCount += 1;
  return { label: p.label, profiles };
}

/**
 * Compute the median of a numeric array.  Returns `null` for empty input.
 */
export function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Compute a weighted median.  Each value[i] is paired with weight[i]
 * (must be > 0).  Returns `null` when no valid pairs exist.
 *
 * Algorithm: sort by value, accumulate weights until >= half of total weight.
 * O(n log n) - negligible for the small sample counts we process (< 100).
 */
export function weightedMedian(
  values: number[],
  weights: number[],
): number | null {
  if (values.length === 0 || values.length !== weights.length) return null;

  const pairs: { v: number; w: number }[] = [];
  let totalWeight = 0;
  for (let i = 0; i < values.length; i++) {
    if (weights[i] > 0) {
      pairs.push({ v: values[i], w: weights[i] });
      totalWeight += weights[i];
    }
  }
  if (pairs.length === 0) return null;

  pairs.sort((a, b) => a.v - b.v);

  const halfWeight = totalWeight / 2;
  let cumulative = 0;
  for (const { v, w } of pairs) {
    cumulative += w;
    if (cumulative >= halfWeight) return v;
  }
  return pairs[pairs.length - 1].v;
}
