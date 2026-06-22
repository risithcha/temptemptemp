/**
 * Atlas Mobile - Real-time FFT Alarm Detector
 *
 * Uses react-native-audio-api (Web Audio API compatible) to capture
 * microphone input, run native FFT via AnalyserNode, and detect high-pitched
 * emergency alarms (fire alarms, smoke detectors, sirens) in real time.
 *
 * Detection uses getByteFrequencyData (0-255 normalized) with a
 * peak-based + ratio hybrid approach.  The audio graph inserts a
 * GainNode(0) before destination so mic audio is analysed but never
 * played through the speakers.
 *
 * IMPORTANT: On Android the alarm detector MUST start BEFORE
 * expo-speech-recognition to grab the mic first.  Both can share it,
 * but the first opener gets priority.
 */
import { useState, useRef, useCallback, useEffect } from 'react';
import type { RefObject } from 'react';
import {
  AudioRecorder,
  AudioContext as RNAudioContext,
  AudioManager,
} from 'react-native-audio-api';

import { estimatePitch } from '../utils/pitch_utils';
import { computeMelBandEnergies, MEL_FILTER_BANK } from '../utils/mel_utils';
import {
  ALARM_FFT_SIZE,
  ALARM_CONSECUTIVE_FRAMES,
  ALARM_POLL_INTERVAL_MS,
  ALARM_AUTO_CLEAR_MS,
  MIN_VOICE_CONFIDENCE,
  CENTROID_MIN,
  PCM_TAP_SAMPLE_RATE,
  PCM_TAP_CHANNELS,
  PITCH_MIN,
  PITCH_MAX,
} from '../theme';

// ---------------------------------------------------------------------------
// Debug tag
// ---------------------------------------------------------------------------
const TAG = '[AlarmDetector]';

/** Raw PCM frame delivered by the recorder's data-callback tap. */
export type PcmListener = (samples: Float32Array, sampleRate: number) => void;

const ALARM_BANDS = {
  // High band – residential smoke chirps (~3 kHz) and phone-speaker smear.
  smoke_alarm: { low: 3000, high: 4500, label: 'SMOKE_ALARM', message: 'SMOKE ALARM DETECTED', minEnergyRatio: 0.08 },
  // Mid band – evacuation horns / T3-T4 beeps (dominate through phone speakers).
  fire_alarm:  { low: 900,  high: 2600, label: 'FIRE_ALARM',  message: 'FIRE ALARM DETECTED', minEnergyRatio: 0.14 },
  siren:       { low: 600,  high: 2200, label: 'SIREN', message: 'EMERGENCY SIREN DETECTED', minEnergyRatio: 0.16, requiresSweep: true },
} as const;

/** Hint when FFT cannot split overlapping chirp bands – Gemini must decide from audio pattern. */
export const EMERGENCY_CHIRP_HINT = 'EMERGENCY_CHIRP';

/** Min Hz swing in spectral centroid over recent polls to qualify as a siren sweep. */
const SIREN_CENTROID_SPAN_HZ = 500;
/** Vehicle sirens keep centroid elevated; phone smoke chirps stay below this. */
const SIREN_MIN_CENTROID_HZ = 1000;
/** Mid-band energy with low centroid → smoke chirp through phone speaker. */
const CHIRP_MAX_CENTROID_HZ = 1000;
const CENTROID_HISTORY_LEN = 20;
/** ~3 s of poll frames – used to distinguish grouped fire beeps vs sparse smoke chirps. */
const BURST_HISTORY_LEN = 30;
/** Fire T3/T4 groups produce many on/off burst transitions; a lone smoke chirp produces one. */
const SPARSE_CHIRP_MAX_FRAMES = 7;
/** Fire must exceed smoke by this factor to win when both bands hit on chirp-like audio. */
const FIRE_OVER_SMOKE_ENERGY = 1.12;
const FIRE_OVER_SMOKE_PEAK = 1.08;
/** Min ms between FFT anomaly events sent to Gemini. */
const ANOMALY_COOLDOWN_MS = 10_000;
/** Min local FFT confidence before emitting an anomaly. */
const MIN_ANOMALY_CONFIDENCE = 0.4;

type AlarmBandKey = keyof typeof ALARM_BANDS;

interface BandMetrics {
  key: AlarmBandKey;
  bandMax: number;
  energyRatio: number;
  isHit: boolean;
}

interface BandScanResult {
  bestBandKey: AlarmBandKey | null;
  bestPeak: number;
  bestRatio: number;
  /** True when any alarm band shows tonal energy (blocks speech suppression). */
  hasEmergencyActivity: boolean;
  rawMetrics: BandMetrics[];
}

function centroidSpanHz(history: number[]): number {
  if (history.length < 8) return 0;
  let min = history[0];
  let max = history[0];
  for (const c of history) {
    if (c < min) min = c;
    if (c > max) max = c;
  }
  return max - min;
}

function measureBand(
  buffer: Uint8Array,
  key: AlarmBandKey,
  totalEnergy: number,
  avgByte: number,
  peakThreshold: number,
): BandMetrics {
  const band = ALARM_BANDS[key];
  const lowBin = Math.floor(band.low / BIN_HZ);
  const highBin = Math.min(Math.ceil(band.high / BIN_HZ), FREQ_BIN_COUNT - 1);

  let bandEnergy = 0;
  let bandMax = 0;

  for (let i = lowBin; i <= highBin; i++) {
    const v = buffer[i];
    bandEnergy += v * v;
    if (v > bandMax) bandMax = v;
  }

  const energyRatio = totalEnergy > 0 ? bandEnergy / totalEnergy : 0;
  const peakRatio = bandMax / Math.max(avgByte, 1);
  const minEnergyRatio = band.minEnergyRatio;

  const peakAboveThreshold = bandMax >= peakThreshold;
  const peakAboveAvg = bandMax >= avgByte * PEAK_TO_AVG_RATIO;
  const ratioAboveThreshold = energyRatio >= minEnergyRatio;
  const isAgcHit =
    bandMax >= 35 && bandMax >= avgByte * 3.5 && energyRatio >= minEnergyRatio * 0.65;

  const isHit =
    (peakAboveThreshold && peakAboveAvg && ratioAboveThreshold) || isAgcHit;

  if (bandMax > 40) {
    console.log(
      `${TAG} Band "${key}":` +
      ` bandMax=${bandMax} (need >=${peakThreshold})` +
      ` peakRatio=${peakRatio.toFixed(2)} (need >=${PEAK_TO_AVG_RATIO})` +
      ` energyRatio=${energyRatio.toFixed(3)} (need >=${minEnergyRatio})` +
      ` hit=${isHit}`,
    );
  }

  return { key, bandMax, energyRatio, isHit };
}

function centroidMaxHz(history: number[]): number {
  if (history.length === 0) return 0;
  return Math.max(...history);
}

function centroidMedianHz(history: number[]): number {
  if (history.length === 0) return 0;
  const sorted = [...history].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** Prefer vehicle siren when sweep + siren-band energy dominates; chirps → smoke (over fire). */
function pickActiveBand(
  metrics: BandMetrics[],
  centroidMedianHz: number,
  centroidMaxHz: number,
  hasSirenSweep: boolean,
  burstGroups: number,
  recentBurstCount: number,
): BandMetrics | null {
  const raw = Object.fromEntries(metrics.map((m) => [m.key, m])) as Record<
    AlarmBandKey,
    BandMetrics
  >;
  const byKey = Object.fromEntries(metrics.map((m) => [m.key, { ...m }])) as Record<
    AlarmBandKey,
    BandMetrics
  >;

  const smoke = byKey.smoke_alarm;
  const fire = byKey.fire_alarm;
  const siren = byKey.siren;
  const rawSiren = raw.siren;
  const rawSmoke = raw.smoke_alarm;
  const rawFire = raw.fire_alarm;

  const elevatedCentroid =
    centroidMedianHz >= SIREN_MIN_CENTROID_HZ || centroidMaxHz >= 800;

  const sirenBandDominates =
    rawSiren.isHit &&
    hasSirenSweep &&
    rawSiren.bandMax >= 80 &&
    (rawSiren.bandMax > rawSmoke.bandMax * 1.15 ||
      rawSiren.energyRatio > rawSmoke.energyRatio * 1.8);

  if (sirenBandDominates && elevatedCentroid) {
    return rawSiren;
  }

  // Strong siren peaks in the siren band even when centroid history is stale (e.g. after smoke).
  if (
    rawSiren.isHit &&
    hasSirenSweep &&
    rawSiren.bandMax >= 100 &&
    rawSiren.bandMax > rawSmoke.bandMax * 1.25 &&
    rawSiren.energyRatio >= 0.25
  ) {
    return rawSiren;
  }

  const chirpLike = centroidMedianHz < CHIRP_MAX_CENTROID_HZ;

  // Phone-playback chirp: mid-band energy + low median centroid → smoke, not siren.
  if (siren?.isHit && !sirenBandDominates) {
    if (!hasSirenSweep) {
      siren.isHit = false;
    } else if (chirpLike) {
      const looksLikePhoneChirp = siren.energyRatio >= 0.12;
      siren.isHit = false;
      if (looksLikePhoneChirp && rawSiren.bandMax <= rawSmoke.bandMax * 1.1) {
        smoke.isHit = true;
        smoke.bandMax = Math.max(smoke.bandMax, siren.bandMax);
        smoke.energyRatio = Math.max(smoke.energyRatio, siren.energyRatio * 0.45);
      }
    }
  }

  const vehicleSiren =
    siren?.isHit &&
    hasSirenSweep &&
    (centroidMedianHz >= SIREN_MIN_CENTROID_HZ || centroidMaxHz >= 800) &&
    siren.energyRatio >= ALARM_BANDS.siren.minEnergyRatio;

  if (vehicleSiren) {
    return siren;
  }

  // Periodic chirps: smoke = sparse single chirps; fire = grouped T3/T4 beeps (often louder).
  if (chirpLike) {
    const sparseChirp = recentBurstCount <= SPARSE_CHIRP_MAX_FRAMES;
    if (burstGroups >= 2 && rawFire.isHit && rawSiren.bandMax < 80) return rawFire;
    if (rawFire.isHit && rawSmoke.isHit) {
      const fireClearlyStronger =
        rawFire.energyRatio > rawSmoke.energyRatio * FIRE_OVER_SMOKE_ENERGY ||
        rawFire.bandMax > rawSmoke.bandMax * FIRE_OVER_SMOKE_PEAK;
      if (fireClearlyStronger) return rawFire;
      // Mid-band (fire) energy without high-band (smoke) → evacuation horn through phone speaker.
      if (rawFire.isHit && !rawSmoke.isHit) return rawFire;
      if (rawSmoke.isHit && !rawFire.isHit) return smoke;
      return sparseChirp ? smoke : rawFire;
    }
    if (rawFire.isHit) return rawFire;
    if (smoke?.isHit) return smoke;
  }

  if (smoke?.isHit) return smoke;
  if (fire?.isHit) return fire;
  if (siren?.isHit) return siren;

  return null;
}

function countBurstGroups(history: boolean[]): number {
  if (history.length === 0) return 0;
  let groups = 0;
  let prev = false;
  for (const active of history) {
    if (active && !prev) groups += 1;
    prev = active;
  }
  return groups;
}

/** Override consensus label using raw band metrics and recent burst pattern. */
function deriveGeminiHint(
  consensusBandKey: AlarmBandKey,
  rawMetrics: BandMetrics[],
  hasSirenSweep: boolean,
  burstHistory: boolean[],
): string {
  const siren = rawMetrics.find((m) => m.key === 'siren');
  const smoke = rawMetrics.find((m) => m.key === 'smoke_alarm');
  const fire = rawMetrics.find((m) => m.key === 'fire_alarm');
  if (!siren || !smoke) return ALARM_BANDS[consensusBandKey].label;

  const sirenDominant =
    hasSirenSweep &&
    siren.isHit &&
    siren.bandMax >= 80 &&
    (siren.bandMax > smoke.bandMax * 1.15 || siren.energyRatio > smoke.energyRatio * 1.8);

  if (sirenDominant) return ALARM_BANDS.siren.label;

  const recentBurstCount = burstHistory.filter(Boolean).length;
  const burstGroups = countBurstGroups(burstHistory);
  const sparseChirp = recentBurstCount <= SPARSE_CHIRP_MAX_FRAMES;
  const sirenWeak = !siren.isHit || siren.bandMax < 80 || siren.energyRatio < 0.12;

  const bothChirpBands =
    fire?.isHit &&
    smoke.isHit &&
    fire.energyRatio >= ALARM_BANDS.fire_alarm.minEnergyRatio &&
    sirenWeak;

  // Overlapping chirp bands on phone speakers – never mislead Gemini with SMOKE-only hint.
  if (bothChirpBands) {
    if (sparseChirp && burstGroups <= 1) return ALARM_BANDS.smoke_alarm.label;
    if (burstGroups >= 2) return ALARM_BANDS.fire_alarm.label;
    return EMERGENCY_CHIRP_HINT;
  }

  if (sparseChirp && burstGroups <= 1 && smoke.isHit && !fire?.isHit) {
    return ALARM_BANDS.smoke_alarm.label;
  }

  if (burstGroups >= 2 && fire?.isHit && sirenWeak) return ALARM_BANDS.fire_alarm.label;

  if (fire?.isHit && !smoke?.isHit && sirenWeak) return ALARM_BANDS.fire_alarm.label;

  if (fire?.isHit && fire.energyRatio >= ALARM_BANDS.fire_alarm.minEnergyRatio) {
    const fireStronger =
      !smoke?.isHit ||
      fire.energyRatio > smoke.energyRatio * FIRE_OVER_SMOKE_ENERGY ||
      fire.bandMax > smoke.bandMax * FIRE_OVER_SMOKE_PEAK;
    if (fireStronger) return ALARM_BANDS.fire_alarm.label;
  }

  if (consensusBandKey === 'fire_alarm') return ALARM_BANDS.fire_alarm.label;
  return ALARM_BANDS[consensusBandKey].label;
}

function scanAlarmBands(
  buffer: Uint8Array,
  totalEnergy: number,
  avgByte: number,
  peakThreshold: number,
  hasSirenSweep: boolean,
  centroidMedianHz: number,
  centroidMaxHz: number,
  burstGroups: number,
  recentBurstCount: number,
): BandScanResult {
  const metrics: BandMetrics[] = (
    Object.keys(ALARM_BANDS) as AlarmBandKey[]
  ).map((key) => measureBand(buffer, key, totalEnergy, avgByte, peakThreshold));

  const active = pickActiveBand(
    metrics,
    centroidMedianHz,
    centroidMaxHz,
    hasSirenSweep,
    burstGroups,
    recentBurstCount,
  );
  const hasEmergencyActivity = active != null;

  return {
    bestBandKey: active?.key ?? null,
    bestPeak: active?.bandMax ?? 0,
    bestRatio: active?.energyRatio ?? 0,
    hasEmergencyActivity,
    rawMetrics: metrics,
  };
}

/** Allow alarm hits through speech suppression when the tone is unmistakable. */
function isStrongEmergency(
  bandKey: AlarmBandKey | null,
  bestPeak: number,
  bestRatio: number,
  peakThreshold: number,
  hasSirenSweep: boolean,
): boolean {
  if (!bandKey) return false;
  const band = ALARM_BANDS[bandKey];
  if (bandKey === 'siren') {
    return hasSirenSweep && bestPeak >= peakThreshold && bestRatio >= band.minEnergyRatio;
  }
  return bestPeak >= peakThreshold && bestRatio >= band.minEnergyRatio;
}

const SAMPLE_RATE = 44100;
const BIN_HZ = SAMPLE_RATE / ALARM_FFT_SIZE;
const FREQ_BIN_COUNT = ALARM_FFT_SIZE / 2;

// Detection thresholds (byte values 0-255)
const PEAK_THRESHOLD = 80;
const PEAK_TO_AVG_RATIO = 2.5;

export interface AlarmAlert {
  type: string;
  message: string;
  confidence: number;
}

export interface AlarmAnomaly {
  bandLabel: string;
  confidence: number;
  pcmSnapshot: Float32Array;
  sampleRate: number;
  timestamp: number;
}

export interface PitchSample {
  hz: number;
  centroid: number;
  /** Harmonicity confidence in [0, 1]. Higher = more periodic (voice-like). */
  confidence: number;
  /** Log Mel-band energies extracted from the FFT buffer at sample time. */
  features: number[];
  time: number;
}

export interface AlarmDetectorState {
  isMonitoring: boolean;
  alert: AlarmAlert | null;
  anomaly: AlarmAnomaly | null;
  startMonitoring: () => Promise<void>;
  stopMonitoring: () => void;
  dismissAlert: () => void;
  /** Clears FFT anomaly debounce so a new snapshot can fire after Gemini unavailable. */
  releaseAnomalyDebounce: () => void;
  /** Rolling buffer of recent vocal pitch estimates for speaker diarization. */
  pitchHistoryRef: RefObject<PitchSample[]>;
}

export interface AlarmDetectorOptions {
  /** FFT peak threshold override (default: 80). Lower = more sensitive. */
  peakThreshold?: number;
  /**
   * Raw PCM tap.  When provided, the recorder's `onAudioReady` data callback is
   * registered and every captured mono frame (Float32 `[-1,1]`) is forwarded
   * here along with the actually-negotiated sample rate.  This makes the alarm
   * detector the single mic owner and feeds Deepgram streaming without a second
   * recorder.  The FFT/alarm graph is unaffected.
   */
  onPcm?: PcmListener;
}

export function useAlarmDetector(
  options: AlarmDetectorOptions = {},
): AlarmDetectorState {
  const peakThresholdRef = useRef(options.peakThreshold ?? PEAK_THRESHOLD);
  peakThresholdRef.current = options.peakThreshold ?? PEAK_THRESHOLD;

  // Keep the latest PCM listener without re-subscribing the recorder callback.
  const onPcmRef = useRef<PcmListener | undefined>(options.onPcm);
  onPcmRef.current = options.onPcm;

  const [isMonitoring, setIsMonitoring] = useState(false);
  const isMonitoringRef = useRef(false);
  isMonitoringRef.current = isMonitoring;
  const [alert, setAlert] = useState<AlarmAlert | null>(null);
  const [anomaly, setAnomaly] = useState<AlarmAnomaly | null>(null);

  const audioContextRef = useRef<InstanceType<typeof RNAudioContext> | null>(null);
  const recorderRef = useRef<InstanceType<typeof AudioRecorder> | null>(null);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const consecutiveHitsRef = useRef(0);
  const lastAlertTimeRef = useRef(0);
  const autoClearTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const analyserRef = useRef<ReturnType<InstanceType<typeof RNAudioContext>['createAnalyser']> | null>(null);
  const pollCountRef = useRef(0);
  const byteBufferRef = useRef<Uint8Array | null>(null);
  const timeDomainBufferRef = useRef<Float32Array | null>(null);
  const pitchHistoryRef = useRef<PitchSample[]>([]);
  const pcmFrameCountRef = useRef(0);
  const hitHistoryRef = useRef<string[]>([]);
  const lastActiveBandKeyRef = useRef<AlarmBandKey | null>(null);
  const centroidHistoryRef = useRef<number[]>([]);
  const emergencyBurstHistoryRef = useRef<boolean[]>([]);
  
  // 3-second rolling PCM buffer
  const rollingPcmBufferRef = useRef<Float32Array | null>(null);
  const rollingPcmIndexRef = useRef(0);

  const clearTimers = useCallback(() => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
    if (autoClearTimerRef.current) {
      clearTimeout(autoClearTimerRef.current);
      autoClearTimerRef.current = null;
    }
  }, []);

  const teardownAudio = useCallback(() => {
    console.log(`${TAG} teardownAudio() called`);
    clearTimers();
    if (recorderRef.current) {
      try { recorderRef.current.clearOnAudioReady(); } catch { /* no tap registered */ }
      try { recorderRef.current.stop(); } catch { /* already stopped */ }
      recorderRef.current = null;
    }
    if (audioContextRef.current) {
      try { audioContextRef.current.close(); } catch { /* ok */ }
      audioContextRef.current = null;
    }
    analyserRef.current = null;
    byteBufferRef.current = null;
    timeDomainBufferRef.current = null;
    pitchHistoryRef.current = [];
    consecutiveHitsRef.current = 0;
    hitHistoryRef.current = [];
    centroidHistoryRef.current = [];
    emergencyBurstHistoryRef.current = [];
    rollingPcmBufferRef.current = null;
    rollingPcmIndexRef.current = 0;
    pollCountRef.current = 0;
    pcmFrameCountRef.current = 0;
    console.log(`${TAG} teardownAudio() complete`);
  }, [clearTimers]);

  const dismissAlert = useCallback(() => {
    console.log(`${TAG} dismissAlert() called`);
    setAlert(null);
    setAnomaly(null);
    consecutiveHitsRef.current = 0;
    hitHistoryRef.current = [];
    centroidHistoryRef.current = [];
    lastActiveBandKeyRef.current = null;
    emergencyBurstHistoryRef.current = [];
    if (autoClearTimerRef.current) {
      clearTimeout(autoClearTimerRef.current);
      autoClearTimerRef.current = null;
    }
  }, []);

  const analyzeSpectrum = useCallback(() => {
    const analyser = analyserRef.current;
    if (!analyser) {
      console.warn(`${TAG} analyzeSpectrum: analyser is null – interval should be cleared`);
      return;
    }

    const buffer = byteBufferRef.current;
    if (!buffer) {
      console.warn(`${TAG} analyzeSpectrum: byteBuffer is null`);
      return;
    }

    pollCountRef.current += 1;

    analyser.getByteFrequencyData(buffer);

    // Compute overall stats + spectral centroid for voice profiling
    let totalEnergy = 0;
    let maxByte = 0;
    let maxByteIdx = 0;
    let weightedFreqSum = 0;
    let magnitudeSum = 0;

    for (let i = 0; i < FREQ_BIN_COUNT; i++) {
      const v = buffer[i];
      totalEnergy += v * v;
      if (v > maxByte) {
        maxByte = v;
        maxByteIdx = i;
      }
      weightedFreqSum += (i * BIN_HZ) * v;
      magnitudeSum += v;
    }

    const avgByte = Math.sqrt(totalEnergy / FREQ_BIN_COUNT);
    const spectralCentroid = magnitudeSum > 0 ? weightedFreqSum / magnitudeSum : 0;

    centroidHistoryRef.current.push(spectralCentroid);
    if (centroidHistoryRef.current.length > CENTROID_HISTORY_LEN) {
      centroidHistoryRef.current.shift();
    }
    const hasSirenSweep = centroidSpanHz(centroidHistoryRef.current) >= SIREN_CENTROID_SPAN_HZ;

    // Log every 50 polls (~5 seconds) so we can verify FFT is receiving audio
    if (pollCountRef.current % 50 === 0) {
      console.log(
        `${TAG} FFT poll #${pollCountRef.current}` +
        ` | maxByte=${maxByte}` +
        ` | avgByte=${avgByte.toFixed(1)}` +
        ` | centroid=${spectralCentroid.toFixed(0)} Hz` +
        ` | consecutiveHits=${consecutiveHitsRef.current}` +
        ` | pcmFrames=${pcmFrameCountRef.current}`,
      );
    }

    if (maxByte === 0) {
      if (pollCountRef.current % 50 === 0) {
        console.log(`${TAG} FFT poll #${pollCountRef.current}: maxByte=0 (silence / no audio signal)`);
      }
      consecutiveHitsRef.current = 0;
      return;
    }

    // --- Pitch + centroid estimation for speaker diarization (runs first) ---
    let voicedSpeechDetected = false;
    const tdBuffer = timeDomainBufferRef.current;
    if (tdBuffer) {
      analyser.getFloatTimeDomainData(tdBuffer);
      const pitchResult = estimatePitch(tdBuffer, SAMPLE_RATE);
      if (pitchResult !== null && pitchResult.confidence >= MIN_VOICE_CONFIDENCE) {
        const speechLikePitch =
          pitchResult.hz >= PITCH_MIN &&
          pitchResult.hz <= PITCH_MAX &&
          spectralCentroid <= 2000;
        if (speechLikePitch) {
          voicedSpeechDetected = true;
        }
        if (spectralCentroid >= CENTROID_MIN && speechLikePitch) {
          const now = Date.now();
          const melFeatures = computeMelBandEnergies(buffer, MEL_FILTER_BANK);
          pitchHistoryRef.current.push({
            hz: pitchResult.hz,
            centroid: spectralCentroid,
            confidence: pitchResult.confidence,
            features: Array.from(melFeatures),
            time: now,
          });
          const cutoff = now - 10_000;
          while (
            pitchHistoryRef.current.length > 0 &&
            pitchHistoryRef.current[0].time < cutoff
          ) {
            pitchHistoryRef.current.shift();
          }
        }
      }
    }

    // --- Alarm band detection (suppressed during voiced speech unless tone is strong) ---
    const centroidMed = centroidMedianHz(centroidHistoryRef.current);
    const centroidMax = centroidMaxHz(centroidHistoryRef.current);
    const burstGroupsSoFar = countBurstGroups(emergencyBurstHistoryRef.current);
    const recentBurstCountSoFar = emergencyBurstHistoryRef.current.filter(Boolean).length;
    const { bestBandKey, bestPeak, bestRatio, rawMetrics } = scanAlarmBands(
      buffer,
      totalEnergy,
      avgByte,
      peakThresholdRef.current,
      hasSirenSweep,
      centroidMed,
      centroidMax,
      burstGroupsSoFar,
      recentBurstCountSoFar,
    );

    let activeBandKey = bestBandKey;
    if (
      voicedSpeechDetected &&
      !isStrongEmergency(
        bestBandKey,
        bestPeak,
        bestRatio,
        peakThresholdRef.current,
        hasSirenSweep,
      )
    ) {
      activeBandKey = null;
    }

    if (activeBandKey) {
      if (
        lastActiveBandKeyRef.current != null &&
        lastActiveBandKeyRef.current !== activeBandKey
      ) {
        consecutiveHitsRef.current = 0;
        hitHistoryRef.current = [];
      }
      lastActiveBandKeyRef.current = activeBandKey;

      hitHistoryRef.current.push(activeBandKey);
      if (hitHistoryRef.current.length > 20) {
        hitHistoryRef.current.shift();
      }
      consecutiveHitsRef.current += 1;
      console.log(
        `${TAG} ✅ BAND HIT "${activeBandKey}" – consecutiveHits=${consecutiveHitsRef.current}/${ALARM_CONSECUTIVE_FRAMES}` +
        ` bestPeak=${bestPeak} bestRatio=${bestRatio.toFixed(3)}`,
      );
    } else {
      consecutiveHitsRef.current = Math.max(0, consecutiveHitsRef.current - 1);
      if (consecutiveHitsRef.current === 0) {
        hitHistoryRef.current = [];
        lastActiveBandKeyRef.current = null;
      }
    }

    emergencyBurstHistoryRef.current.push(activeBandKey != null);
    if (emergencyBurstHistoryRef.current.length > BURST_HISTORY_LEN) {
      emergencyBurstHistoryRef.current.shift();
    }

    if (consecutiveHitsRef.current >= ALARM_CONSECUTIVE_FRAMES && activeBandKey) {
      const now = Date.now();
      if (now - lastAlertTimeRef.current > ANOMALY_COOLDOWN_MS) {
        lastAlertTimeRef.current = now;

        // Majority vote over the recent hits to stabilize the UI
        // Sweeping alarms cross multiple bands, causing frame-by-frame fluctuation.
        const counts: Record<string, number> = {};
        let consensusBandKey = activeBandKey;
        let maxCount = 0;
        for (const k of hitHistoryRef.current) {
          counts[k] = (counts[k] || 0) + 1;
          if (counts[k] > maxCount) {
            maxCount = counts[k];
            consensusBandKey = k as AlarmBandKey;
          }
        }

        const band = ALARM_BANDS[consensusBandKey];
        const burstHistory = [...emergencyBurstHistoryRef.current];
        const recentBurstCount = burstHistory.filter(Boolean).length;
        const burstGroups = countBurstGroups(burstHistory);
        const geminiHint = deriveGeminiHint(
          consensusBandKey,
          rawMetrics,
          hasSirenSweep,
          burstHistory,
        );
        const confidence = Math.min(
          1,
          (bestPeak / 255) * (bestRatio / band.minEnergyRatio) * 0.42,
        );

        if (confidence < MIN_ANOMALY_CONFIDENCE) {
          console.log(
            `${TAG} Anomaly suppressed – local confidence ${confidence.toFixed(3)} below ${MIN_ANOMALY_CONFIDENCE}`,
          );
          return;
        }

        console.log(
          `${TAG} 🚨 ANOMALY DETECTED (FFT hint=${geminiHint}, consensus=${maxCount}/${hitHistoryRef.current.length})` +
          ` confidence=${confidence.toFixed(3)} sweep=${hasSirenSweep} bursts=${recentBurstCount}/${BURST_HISTORY_LEN}` +
          ` burstGroups=${burstGroups} centroidMed=${centroidMed.toFixed(0)}Hz – awaiting Gemini validation`,
        );

        // Capture PCM snapshot (3 seconds)
        let snapshot = new Float32Array(0);
        if (rollingPcmBufferRef.current) {
          const buf = rollingPcmBufferRef.current;
          snapshot = new Float32Array(buf.length);
          const idx = rollingPcmIndexRef.current;
          let dst = 0;
          for (let i = idx; i < buf.length; i++) snapshot[dst++] = buf[i];
          for (let i = 0; i < idx; i++) snapshot[dst++] = buf[i];
        }

        setAnomaly({
          bandLabel: geminiHint,
          confidence,
          pcmSnapshot: snapshot,
          sampleRate: PCM_TAP_SAMPLE_RATE,
          timestamp: now,
        });

        if (autoClearTimerRef.current) clearTimeout(autoClearTimerRef.current);
        autoClearTimerRef.current = setTimeout(() => {
          setAlert(null);
          setAnomaly(null);
          consecutiveHitsRef.current = 0;
        }, ALARM_AUTO_CLEAR_MS);
      } else {
        console.log(`${TAG} Alarm debounced – last alert was ${now - lastAlertTimeRef.current}ms ago`);
      }
    }
  }, []);

  const startMonitoring = useCallback(async () => {
    console.log(`${TAG} startMonitoring() called | isMonitoring=${isMonitoringRef.current}`);

    if (isMonitoringRef.current) {
      console.warn(`${TAG} startMonitoring() skipped – already monitoring`);
      return;
    }

    // Step 1: Request permissions
    console.log(`${TAG} [1/7] Requesting recording permissions…`);
    let permissions: string;
    try {
      permissions = await AudioManager.requestRecordingPermissions();
    } catch (e) {
      console.error(`${TAG} [1/7] requestRecordingPermissions threw:`, e);
      return;
    }
    console.log(`${TAG} [1/7] Permission result: "${permissions}"`);
    if (permissions !== 'Granted') {
      console.error(`${TAG} [1/7] Permission denied – aborting startMonitoring`);
      return;
    }

    // Step 2: Activate audio session
    console.log(`${TAG} [2/7] Setting audio session activity…`);
    let sessionOk: boolean;
    try {
      sessionOk = await AudioManager.setAudioSessionActivity(true);
    } catch (e) {
      console.error(`${TAG} [2/7] setAudioSessionActivity threw:`, e);
      return;
    }
    console.log(`${TAG} [2/7] Audio session activity: ${sessionOk}`);
    if (!sessionOk) {
      console.error(`${TAG} [2/7] Audio session activation failed – aborting`);
      return;
    }

    // Step 3: Create audio graph
    console.log(`${TAG} [3/7] Creating AudioContext + AudioRecorder + AnalyserNode (sampleRate=${SAMPLE_RATE})…`);
    let ctx: InstanceType<typeof RNAudioContext>;
    let recorder: InstanceType<typeof AudioRecorder>;
    try {
      ctx = new RNAudioContext({ sampleRate: SAMPLE_RATE });
      recorder = new AudioRecorder();
    } catch (e) {
      console.error(`${TAG} [3/7] Failed to create AudioContext or AudioRecorder:`, e);
      return;
    }

    let analyser: ReturnType<InstanceType<typeof RNAudioContext>['createAnalyser']>;
    try {
      analyser = ctx.createAnalyser();
      analyser.fftSize = ALARM_FFT_SIZE;
      analyser.smoothingTimeConstant = 0.3;

      const adapter = ctx.createRecorderAdapter();
      const silentGain = ctx.createGain();
      silentGain.gain.value = 0;

      recorder.connect(adapter);
      adapter.connect(analyser);
      analyser.connect(silentGain);
      silentGain.connect(ctx.destination);
    } catch (e) {
      console.error(`${TAG} [3/7] Failed to build audio graph:`, e);
      try { ctx.close(); } catch { /* ignore */ }
      return;
    }
    console.log(`${TAG} [3/7] Audio graph built. fftSize=${analyser.fftSize} frequencyBinCount=${analyser.frequencyBinCount}`);

    // Step 4: Resume context if suspended
    console.log(`${TAG} [4/7] AudioContext state: "${ctx.state}"`);
    if (ctx.state === 'suspended') {
      console.log(`${TAG} [4/7] Resuming suspended AudioContext…`);
      try {
        await ctx.resume();
        console.log(`${TAG} [4/7] AudioContext resumed. State: "${ctx.state}"`);
      } catch (e) {
        console.error(`${TAG} [4/7] ctx.resume() threw:`, e);
        try { ctx.close(); } catch { /* ignore */ }
        return;
      }
    }

    // Step 5: Register PCM tap (ALWAYS ENABLED for rolling buffer)
    const hasPcmListener = typeof onPcmRef.current === 'function';
    console.log(`${TAG} [5/7] PCM tap: ENABLED (hasPcmListener=${hasPcmListener}) sampleRate=${PCM_TAP_SAMPLE_RATE} channels=${PCM_TAP_CHANNELS}`);

    // Allocate 3-second rolling buffer
    rollingPcmBufferRef.current = new Float32Array(PCM_TAP_SAMPLE_RATE * 3);
    rollingPcmIndexRef.current = 0;

    try {
      recorder.onAudioReady(
        {
          sampleRate: PCM_TAP_SAMPLE_RATE,
          bufferLength: Math.round(PCM_TAP_SAMPLE_RATE * 0.1), // ~100 ms chunks
          channelCount: PCM_TAP_CHANNELS,
        },
        (event) => {
          pcmFrameCountRef.current += 1;
          // Log the first few PCM frames so we can confirm audio is flowing
          if (pcmFrameCountRef.current <= 5) {
            console.log(
              `${TAG} PCM frame #${pcmFrameCountRef.current}:` +
              ` numFrames=${event.numFrames}` +
              ` sampleRate=${event.buffer?.sampleRate}`,
            );
          }
          if (pcmFrameCountRef.current === 100) {
            console.log(`${TAG} PCM tap: 100 frames received – tap is healthy`);
          }

          const listener = onPcmRef.current;
          const frames = event.numFrames;
          if (frames <= 0) return;
          // copyFromChannel (NOT getChannelData) avoids the known native crash.
          const samples = new Float32Array(frames);
          try {
            event.buffer.copyFromChannel(samples, 0, 0);
          } catch (e) {
            console.warn(`${TAG} PCM copyFromChannel failed:`, e);
            return;
          }

          // Write to rolling buffer
          if (rollingPcmBufferRef.current) {
            const buf = rollingPcmBufferRef.current;
            for (let i = 0; i < frames; i++) {
              buf[rollingPcmIndexRef.current] = samples[i];
              rollingPcmIndexRef.current = (rollingPcmIndexRef.current + 1) % buf.length;
            }
          }

          if (listener) listener(samples, event.buffer.sampleRate);
        },
      );
      console.log(`${TAG} [5/7] onAudioReady registered successfully`);
    } catch (e) {
      console.warn(`${TAG} [5/7] onAudioReady threw (PCM tap unavailable – alarm FFT still works):`, e);
    }

    // Step 6: Start recorder
    console.log(`${TAG} [6/7] Starting recorder…`);
    let result: { status: string };
    try {
      result = recorder.start();
    } catch (e) {
      console.error(`${TAG} [6/7] recorder.start() threw:`, e);
      try { recorder.clearOnAudioReady(); } catch { /* ignore */ }
      try { ctx.close(); } catch { /* ignore */ }
      return;
    }
    console.log(`${TAG} [6/7] recorder.start() result: status="${result.status}"`);

    if (result.status === 'error') {
      console.error(`${TAG} [6/7] recorder.start() returned error – aborting`);
      try { recorder.clearOnAudioReady(); } catch { /* ignore */ }
      ctx.close();
      return;
    }

    // Step 7: Start polling
    audioContextRef.current = ctx;
    recorderRef.current = recorder;
    analyserRef.current = analyser;
    byteBufferRef.current = new Uint8Array(analyser.frequencyBinCount);
    timeDomainBufferRef.current = new Float32Array(analyser.fftSize);
    pitchHistoryRef.current = [];
    pollCountRef.current = 0;
    pcmFrameCountRef.current = 0;

    pollTimerRef.current = setInterval(analyzeSpectrum, ALARM_POLL_INTERVAL_MS);
    setIsMonitoring(true);

    console.log(
      `${TAG} [7/7] ✅ Monitoring STARTED` +
      ` | pollInterval=${ALARM_POLL_INTERVAL_MS}ms` +
      ` | peakThreshold=${peakThresholdRef.current}` +
      ` | consecutiveFramesNeeded=${ALARM_CONSECUTIVE_FRAMES}` +
      ` | autoClearMs=${ALARM_AUTO_CLEAR_MS}`,
    );
  }, [analyzeSpectrum]);

  const stopMonitoring = useCallback(() => {
    if (!isMonitoringRef.current && audioContextRef.current == null) {
      return;
    }
    console.log(`${TAG} stopMonitoring() called | isMonitoring=${isMonitoringRef.current}`);
    teardownAudio();
    setIsMonitoring(false);
    console.log(`${TAG} stopMonitoring() complete`);
  }, [teardownAudio]);

  useEffect(() => {
    console.log(`${TAG} mounted`);
    return () => {
      console.log(`${TAG} unmounting – tearing down audio`);
      teardownAudio();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const releaseAnomalyDebounce = useCallback(() => {
    lastAlertTimeRef.current = 0;
  }, []);

  return {
    isMonitoring,
    alert,
    anomaly,
    startMonitoring,
    stopMonitoring,
    dismissAlert,
    releaseAnomalyDebounce,
    pitchHistoryRef,
  };
}
