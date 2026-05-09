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
} from '../theme';

const ALARM_BANDS = {
  smoke_alarm: { low: 2800, high: 4500, label: 'SMOKE_ALARM', message: 'SMOKE ALARM DETECTED' },
  fire_alarm:  { low: 2400, high: 4000, label: 'FIRE_ALARM',  message: 'FIRE ALARM DETECTED' },
  siren_high:  { low: 1800, high: 3500, label: 'SIREN',       message: 'EMERGENCY SIREN DETECTED' },
} as const;

const SAMPLE_RATE = 44100;
const BIN_HZ = SAMPLE_RATE / ALARM_FFT_SIZE;
const FREQ_BIN_COUNT = ALARM_FFT_SIZE / 2;

// Detection thresholds (byte values 0-255)
const PEAK_THRESHOLD = 80;
const PEAK_TO_AVG_RATIO = 2.5;
const BAND_ENERGY_RATIO_THRESHOLD = 0.20;

export interface AlarmAlert {
  type: string;
  message: string;
  confidence: number;
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
  startMonitoring: () => Promise<void>;
  stopMonitoring: () => void;
  dismissAlert: () => void;
  /** Rolling buffer of recent vocal pitch estimates for speaker diarization. */
  pitchHistoryRef: RefObject<PitchSample[]>;
}

export interface AlarmDetectorOptions {
  /** FFT peak threshold override (default: 80). Lower = more sensitive. */
  peakThreshold?: number;
}

export function useAlarmDetector(
  options: AlarmDetectorOptions = {},
): AlarmDetectorState {
  const peakThresholdRef = useRef(options.peakThreshold ?? PEAK_THRESHOLD);
  peakThresholdRef.current = options.peakThreshold ?? PEAK_THRESHOLD;

  const [isMonitoring, setIsMonitoring] = useState(false);
  const [alert, setAlert] = useState<AlarmAlert | null>(null);

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
    clearTimers();
    if (recorderRef.current) {
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
    pollCountRef.current = 0;
  }, [clearTimers]);

  const dismissAlert = useCallback(() => {
    setAlert(null);
    consecutiveHitsRef.current = 0;
    if (autoClearTimerRef.current) {
      clearTimeout(autoClearTimerRef.current);
      autoClearTimerRef.current = null;
    }
  }, []);

  const analyzeSpectrum = useCallback(() => {
    const analyser = analyserRef.current;
    if (!analyser) return;

    const buffer = byteBufferRef.current;
    if (!buffer) return;

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

    if (maxByte === 0) {
      consecutiveHitsRef.current = 0;
      return;
    }

    // --- Pitch + centroid estimation for speaker diarization ---
    // Runs BEFORE alarm checks so we can suppress false alarms during speech.
    let voicedSpeechDetected = false;
    const tdBuffer = timeDomainBufferRef.current;
    if (tdBuffer) {
      analyser.getFloatTimeDomainData(tdBuffer);
      const pitchResult = estimatePitch(tdBuffer, SAMPLE_RATE);
      if (pitchResult !== null && pitchResult.confidence >= MIN_VOICE_CONFIDENCE) {
        voicedSpeechDetected = true;
        // Only record for diarization if the centroid is physically
        // plausible for speech.  Readings below CENTROID_MIN (e.g. 60 Hz)
        // are room-rumble / movement artifacts that would poison the
        // weighted median and create false speaker profiles.
        if (spectralCentroid >= CENTROID_MIN) {
          const now = Date.now();
          const melFeatures = computeMelBandEnergies(buffer, MEL_FILTER_BANK);
          pitchHistoryRef.current.push({
            hz: pitchResult.hz,
            centroid: spectralCentroid,
            confidence: pitchResult.confidence,
            features: Array.from(melFeatures),
            time: now,
          });
          // Evict samples older than 10 seconds to bound memory.
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

    // --- Alarm band detection (suppressed during voiced speech) ---
    let bestBandKey: string | null = null;
    let bestPeak = 0;
    let bestRatio = 0;

    if (!voicedSpeechDetected) {
      for (const [key, band] of Object.entries(ALARM_BANDS)) {
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

        const peakAboveThreshold = bandMax >= peakThresholdRef.current;
        const peakAboveAvg = bandMax >= avgByte * PEAK_TO_AVG_RATIO;
        const ratioAboveThreshold = energyRatio >= BAND_ENERGY_RATIO_THRESHOLD;

        const isHit = peakAboveThreshold && peakAboveAvg && ratioAboveThreshold;

        if (isHit && bandMax > bestPeak) {
          bestPeak = bandMax;
          bestRatio = energyRatio;
          bestBandKey = key;
        }
      }
    }

    if (bestBandKey) {
      consecutiveHitsRef.current += 1;
    } else {
      consecutiveHitsRef.current = Math.max(0, consecutiveHitsRef.current - 1);
    }

    if (consecutiveHitsRef.current >= ALARM_CONSECUTIVE_FRAMES && bestBandKey) {
      const now = Date.now();
      if (now - lastAlertTimeRef.current > 2000) {
        lastAlertTimeRef.current = now;

        const band = ALARM_BANDS[bestBandKey as keyof typeof ALARM_BANDS];
        const confidence = Math.min(1, (bestPeak / 255) * (bestRatio / BAND_ENERGY_RATIO_THRESHOLD) * 0.5);

        setAlert({
          type: band.label,
          message: band.message,
          confidence,
        });

        if (autoClearTimerRef.current) clearTimeout(autoClearTimerRef.current);
        autoClearTimerRef.current = setTimeout(() => {
          setAlert(null);
          consecutiveHitsRef.current = 0;
        }, ALARM_AUTO_CLEAR_MS);
      }
    }
  }, []);

  const startMonitoring = useCallback(async () => {
    if (isMonitoring) {
      return;
    }

    const permissions = await AudioManager.requestRecordingPermissions();
    if (permissions !== 'Granted') {
      return;
    }

    const sessionOk = await AudioManager.setAudioSessionActivity(true);
    if (!sessionOk) {
      return;
    }

    const ctx = new RNAudioContext({ sampleRate: SAMPLE_RATE });
    const recorder = new AudioRecorder();
    const analyser = ctx.createAnalyser();
    analyser.fftSize = ALARM_FFT_SIZE;
    analyser.smoothingTimeConstant = 0.3;

    const adapter = ctx.createRecorderAdapter();
    const silentGain = ctx.createGain();
    silentGain.gain.value = 0;

    recorder.connect(adapter);
    adapter.connect(analyser);
    analyser.connect(silentGain);
    silentGain.connect(ctx.destination);

    if (ctx.state === 'suspended') {
      await ctx.resume();
    }

    const result = recorder.start();
    if (result.status === 'error') {
      ctx.close();
      return;
    }

    audioContextRef.current = ctx;
    recorderRef.current = recorder;
    analyserRef.current = analyser;
    byteBufferRef.current = new Uint8Array(analyser.frequencyBinCount);
    timeDomainBufferRef.current = new Float32Array(analyser.fftSize);
    pitchHistoryRef.current = [];
    pollCountRef.current = 0;

    pollTimerRef.current = setInterval(analyzeSpectrum, ALARM_POLL_INTERVAL_MS);
    setIsMonitoring(true);
  }, [isMonitoring, analyzeSpectrum]);

  const stopMonitoring = useCallback(() => {
    teardownAudio();
    setIsMonitoring(false);
  }, [teardownAudio]);

  useEffect(() => {
    return () => {
      teardownAudio();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return {
    isMonitoring,
    alert,
    startMonitoring,
    stopMonitoring,
    dismissAlert,
    pitchHistoryRef,
  };
}
