/**
 * Atlas – Hearing Assist Mode (Voice Processing Layer)
 *
 * A reusable React hook that wraps `expo-speech-recognition` into a clean,
 * plug-and-play abstraction for continuous live captioning.
 *
 * Some things we added on top of the raw module:
 *  1. On-device-first recognition with automatic network fallback.
 *  2. Auto-restart on silence / timeout for seamless continuous captioning.
 *  3. iOS audio session save / restore to avoid conflicts with
 *     react-native-vision-camera.
 *  4. Graceful permission handling — denied permissions never crash the app.
 */
import { useState, useCallback, useRef, useEffect } from 'react';
import type { RefObject } from 'react';
import { Platform } from 'react-native';
import { playHapticPattern } from '../utils/haptic_patterns';
import {
  assignSpeaker,
  median,
  weightedMedian,
  type SpeakerProfile,
} from '../utils/speaker_cluster';
import type { PitchSample } from './useAlarmDetector';
import { MIN_DIARIZATION_SAMPLES, NUM_MEL_BANDS, PROFILE_STALE_MS } from '../theme';
import {
  ExpoSpeechRecognitionModule,
  useSpeechRecognitionEvent,
  type ExpoSpeechRecognitionErrorCode,
} from 'expo-speech-recognition';

// Types

/** A single speaker-tagged segment of transcript. */
export interface TranscriptSegment {
  /** Speaker label (e.g. "Speaker 1", "Speaker 2"). */
  speaker: string;
  /** The text spoken in this segment. */
  text: string;
}

/** Options accepted by the hook consumer. */
export interface UseSpeechRecognitionOptions {
  /** BCP-47 language tag. @default "en-US" */
  lang?: string;
  /**
   * When true, the hook will automatically restart recognition after a
   * silence / timeout event so captioning feels continuous.
   * @default true
   */
  continuous?: boolean;
  /**
   * Delay (ms) before auto-restarting after an unexpected stop.
   * Prevents tight restart loops.
   * @default 300
   */
  restartDelayMs?: number;
  /**
   * Silence duration (ms) between final segments that triggers a new
   * speaker turn.  Set to 0 to disable speaker differentiation.
   * @default 1800
   */
  speakerPauseMs?: number;
  /**
   * When true, fires a gentle haptic tap on speaker change.
   * @default true
   */
  hapticPatternsEnabled?: boolean;
  /**
   * Rolling pitch history from the alarm detector's AnalyserNode.
   * When provided, speaker assignment uses pitch-based clustering
   * instead of pause-based numbering.
   */
  pitchHistoryRef?: RefObject<PitchSample[]>;
}

/** The public surface returned by the hook. */
export interface SpeechRecognitionState {
  /** Live transcript: accumulated final segments + current interim text. */
  text: string;
  /** Speaker-segmented transcript for rich rendering. */
  segments: TranscriptSegment[];
  /** The current interim (partial) text being recognized. */
  interimText: string;
  /** Whether the recogniser is actively listening. */
  isListening: boolean;
  /** Last error code, or `null` when everything is fine. */
  error: string | null;
  /** Whether speech recognition is available on this device at all. */
  isAvailable: boolean;
  /** Request permissions & begin listening. */
  startListening: () => Promise<void>;
  /** Gracefully stop listening (processes a final result). */
  stopListening: () => Promise<void>;
  /** Clear the accumulated transcript. */
  resetTranscript: () => void;
}

// Error codes that should trigger an automatic restart rather than being
// surfaced to the consumer as hard errors.
const RESTARTABLE_ERRORS: Set<ExpoSpeechRecognitionErrorCode> = new Set([
  'no-speech',
  'speech-timeout',
  'network', // transient – worth retrying
  'client', // generic Android client error
]);

// Hook implementation

export function useSpeechRecognition(
  options: UseSpeechRecognitionOptions = {},
): SpeechRecognitionState {
  const {
    lang = 'en-US',
    continuous = true,
    restartDelayMs = 300,
    speakerPauseMs = 1800,
    hapticPatternsEnabled = true,
    pitchHistoryRef,
  } = options;

  // State
  const [text, setText] = useState('');
  const [segments, setSegments] = useState<TranscriptSegment[]>([]);
  const [interimText, setInterimText] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Refs (mutable across renders, no re-render cost)

  /** Whether the user intends recognition to be active. */
  const shouldBeListening = useRef(false);

  /** Accumulated finalised transcript segments. */
  const finalTranscript = useRef('');

  /** Timer handle for the auto-restart debounce. */
  const restartTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  /** Speaker differentiation state. */
  const speakerSegmentsRef = useRef<TranscriptSegment[]>([]);
  const lastFinalTimestampRef = useRef<number>(0);
  const currentSpeakerIndexRef = useRef(1);

  /** Pitch-based diarization state. */
  const speakerProfilesRef = useRef<SpeakerProfile[]>([]);
  const segmentStartTimeRef = useRef<number>(0);
  /**
   * Label of the most recently confirmed speaker.  Passed to `assignSpeaker`
   * on every diarization call so the sticky-hysteresis bias can prefer
   * merging with the ongoing speaker over creating a new profile.
   */
  const lastActiveSpeakerRef = useRef<string | null>(null);
  /** Timer that purges stale speaker profiles after PROFILE_STALE_MS of silence. */
  const profileStaleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  /** Whether we already fell back to network-based recognition. */
  const fellBackToNetwork = useRef(false);

  /**
   * (Re)starts the stale-profile countdown.  Called after every final
   * result so that profiles are only purged when there has been a full
   * PROFILE_STALE_MS of silence; not during active conversation.
   */
  const resetProfileStaleTimer = useCallback(() => {
    if (profileStaleTimerRef.current) clearTimeout(profileStaleTimerRef.current);
    profileStaleTimerRef.current = setTimeout(() => {
      speakerProfilesRef.current = [];
      lastActiveSpeakerRef.current = null;
    }, PROFILE_STALE_MS);
  }, []);

  /** Track haptic setting for use in event callbacks. */
  const hapticEnabledRef = useRef(hapticPatternsEnabled);
  hapticEnabledRef.current = hapticPatternsEnabled;

  /** Saved iOS audio session so we can restore it after recognition. */
  const savedIOSAudioSession = useRef<{
    category: string;
    categoryOptions: string[];
    mode: string;
  } | null>(null);

  // Derived
  const isAvailable = ExpoSpeechRecognitionModule.isRecognitionAvailable();

  // Helpers
  /** Cancel any pending restart timer. */
  const clearRestart = useCallback(() => {
    if (restartTimer.current) {
      clearTimeout(restartTimer.current);
      restartTimer.current = null;
    }
  }, []);

  /**
   * Save the current iOS audio session state so we can restore it when
   * speech recognition ends. This prevents conflicts with the camera.
   */
  const saveIOSAudioSession = useCallback(() => {
    if (Platform.OS !== 'ios') return;
    try {
      const session =
        ExpoSpeechRecognitionModule.getAudioSessionCategoryAndOptionsIOS();
      savedIOSAudioSession.current = session;
    } catch {
      // Non-critical – worst case we don't restore.
      savedIOSAudioSession.current = null;
    }
  }, []);

  /**
   * Restore the iOS audio session to whatever it was before we started
   * speech recognition.
   */
  const restoreIOSAudioSession = useCallback(() => {
    if (Platform.OS !== 'ios' || !savedIOSAudioSession.current) return;
    try {
      ExpoSpeechRecognitionModule.setCategoryIOS(
        savedIOSAudioSession.current as Parameters<
          typeof ExpoSpeechRecognitionModule.setCategoryIOS
        >[0],
      );
    } catch {
      // Best-effort restore.
    }
    savedIOSAudioSession.current = null;
  }, []);

  /** Internal: actually kick off the native recogniser. */
  const beginRecognition = useCallback(() => {
    // Determine whether to use on-device recognition.
    const supportsOnDevice =
      ExpoSpeechRecognitionModule.supportsOnDeviceRecognition();
    const useOnDevice = supportsOnDevice && !fellBackToNetwork.current;

    ExpoSpeechRecognitionModule.start({
      lang,
      interimResults: true,
      continuous,
      requiresOnDeviceRecognition: useOnDevice,
      addsPunctuation: true,
      // Extend Android's silence window so it doesn't cut off recognition
      // early in a quiet room.
      androidIntentOptions: {
        EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS: 10000,
        EXTRA_MASK_OFFENSIVE_WORDS: false,
      },
      // Tell iOS to use a reasonable audio session configuration that coexists
      // with camera audio.
      iosCategory: {
        category: 'playAndRecord',
        categoryOptions: ['defaultToSpeaker', 'allowBluetooth'],
        mode: 'measurement',
      },
      iosTaskHint: 'dictation',
    });
  }, [lang, continuous]);

  // Public methods

  const startListening = useCallback(async () => {
    if (!isAvailable) {
      setError('service-not-allowed');
      return;
    }

    // Request permissions – gracefully bail if denied.
    try {
      const result =
        await ExpoSpeechRecognitionModule.requestPermissionsAsync();
      if (!result.granted) {
        setError('not-allowed');
        return;
      }
    } catch {
      setError('not-allowed');
      return;
    }

    // Clear previous state.
    setError(null);
    fellBackToNetwork.current = false;
    shouldBeListening.current = true;

    // Save the iOS audio session before we mutate it.
    saveIOSAudioSession();

    beginRecognition();
  }, [isAvailable, saveIOSAudioSession, beginRecognition]);

  const stopListening = useCallback(async () => {
    shouldBeListening.current = false;
    clearRestart();

    try {
      ExpoSpeechRecognitionModule.stop();
    } catch {
      // Already stopped or destroyed – safe to ignore.
    }
  }, [clearRestart]);

  const resetTranscript = useCallback(() => {
    finalTranscript.current = '';
    speakerSegmentsRef.current = [];
    currentSpeakerIndexRef.current = 1;
    lastFinalTimestampRef.current = 0;
    speakerProfilesRef.current = [];
    segmentStartTimeRef.current = 0;
    lastActiveSpeakerRef.current = null;
    if (profileStaleTimerRef.current) {
      clearTimeout(profileStaleTimerRef.current);
      profileStaleTimerRef.current = null;
    }
    setText('');
    setSegments([]);
    setInterimText('');
  }, []);

  // Event listeners

  useSpeechRecognitionEvent('start', () => {
    setIsListening(true);
    setError(null);
  });

  useSpeechRecognitionEvent('end', () => {
    setIsListening(false);
    restoreIOSAudioSession();

    if (shouldBeListening.current && continuous) {
      clearRestart();
      restartTimer.current = setTimeout(() => {
        if (shouldBeListening.current) {
          saveIOSAudioSession();
          beginRecognition();
        }
      }, restartDelayMs);
    }
  });

  useSpeechRecognitionEvent('result', (event) => {
    const bestResult = event.results[0];
    if (!bestResult) return;

    if (event.isFinal) {
      const segment = bestResult.transcript.trim();
      if (!segment) return;

      const now = Date.now();
      const elapsed = lastFinalTimestampRef.current > 0
        ? now - lastFinalTimestampRef.current
        : 0;
      lastFinalTimestampRef.current = now;

      // --- Speaker assignment: pitch-based (preferred) or smart fallback ---
      let speakerLabel: string;
      const pitchHistory = pitchHistoryRef?.current;
      const segStart = segmentStartTimeRef.current;

      let segmentPitches: number[] = [];
      let segmentCentroids: number[] = [];
      let segmentConfidences: number[] = [];
      let segmentFeatures: number[][] = [];
      if (pitchHistory && segStart > 0) {
        const segmentSamples = pitchHistory
          .filter((s) => s.time >= segStart && s.time <= now);
        segmentPitches = segmentSamples.map((s) => s.hz);
        segmentCentroids = segmentSamples.map((s) => s.centroid);
        segmentConfidences = segmentSamples.map((s) => s.confidence);
        segmentFeatures = segmentSamples.map((s) => s.features ?? []);
      }

      // Use confidence-weighted medians so high-confidence frames have
      // more influence on the final value than borderline ones.
      const wPitch = weightedMedian(segmentPitches, segmentConfidences);
      const wCentroid = weightedMedian(segmentCentroids, segmentConfidences);
      // Compute per-band weighted median to obtain the 5 Mel-energy features.
      const wFeatures: number[] = Array.from({ length: NUM_MEL_BANDS }, (_, k) => {
        const bandValues = segmentFeatures.map((f) => f[k] ?? 0);
        return weightedMedian(bandValues, segmentConfidences) ?? 0;
      });

      if (wPitch !== null && wCentroid !== null && segmentPitches.length >= MIN_DIARIZATION_SAMPLES) {
        // Enough high-confidence samples = trust the (2+5)D Mel clustering.
        const result = assignSpeaker(
          wPitch,
          wCentroid,
          wFeatures,
          speakerProfilesRef.current,
          undefined,
          lastActiveSpeakerRef.current,
        );
        speakerProfilesRef.current = result.profiles;
        speakerLabel = result.label;
        lastActiveSpeakerRef.current = result.label;
      } else if (wPitch !== null && segmentPitches.length < MIN_DIARIZATION_SAMPLES) {
        // Too few samples to trust the weighted median = reuse last speaker.
        const segs = speakerSegmentsRef.current;
        const lastSeg = segs.length > 0 ? segs[segs.length - 1] : null;
        speakerLabel = lastSeg?.speaker ?? 'Speaker A';
      } else if (pitchHistoryRef) {
        // Insufficient or zero pitch data = reuse the last speaker.
        const segs = speakerSegmentsRef.current;
        const lastSeg = segs.length > 0 ? segs[segs.length - 1] : null;
        speakerLabel = lastSeg?.speaker ?? 'Speaker A';
      } else {
        // Pitch system entirely unavailable = pure pause-based fallback.
        speakerLabel = resolveSpeakerByPause(elapsed);
      }

      // Reset segment start for the next speech chunk.
      segmentStartTimeRef.current = 0;
      // Restart the stale-profile countdown. Profiles should only expire
      // after a genuine long pause, not between normal sentences.
      resetProfileStaleTimer();

      // Detect speaker change for haptics.
      const segs = speakerSegmentsRef.current;
      const lastSeg = segs.length > 0 ? segs[segs.length - 1] : null;
      const speakerChanged = lastSeg != null && lastSeg.speaker !== speakerLabel;

      if (speakerChanged) {
        if (hapticEnabledRef.current) {
          playHapticPattern('newSpeaker');
        }
      }

      if (lastSeg && lastSeg.speaker === speakerLabel) {
        lastSeg.text = lastSeg.text ? `${lastSeg.text} ${segment}` : segment;
      } else {
        segs.push({ speaker: speakerLabel, text: segment });
      }

      finalTranscript.current = finalTranscript.current
        ? `${finalTranscript.current} ${segment}`
        : segment;

      setText(finalTranscript.current);
      setSegments([...segs]);
      setInterimText('');
    } else {
      // Mark the start of this speech segment for pitch sampling.
      if (segmentStartTimeRef.current === 0) {
        segmentStartTimeRef.current = Date.now();
      }

      const interim = bestResult.transcript.trim();
      setInterimText(interim);
      const combined = finalTranscript.current
        ? `${finalTranscript.current} ${interim}`
        : interim;
      setText(combined);
    }
  });

  /**
   * Fallback: assign speaker based on silence duration between segments.
   * Used only when the pitch system is entirely unavailable.
   */
  function resolveSpeakerByPause(elapsedMs: number): string {
    const shouldChange =
      speakerPauseMs > 0 &&
      elapsedMs >= speakerPauseMs &&
      speakerSegmentsRef.current.length > 0;

    if (shouldChange) {
      currentSpeakerIndexRef.current += 1;
    }
    return `Speaker ${currentSpeakerIndexRef.current}`;
  }

  useSpeechRecognitionEvent('error', (event) => {
    const code = event.error;

    if (
      (code === 'service-not-allowed' || code === 'language-not-supported') &&
      !fellBackToNetwork.current &&
      shouldBeListening.current
    ) {
      fellBackToNetwork.current = true;
      clearRestart();
      restartTimer.current = setTimeout(() => {
        if (shouldBeListening.current) {
          saveIOSAudioSession();
          beginRecognition();
        }
      }, restartDelayMs);
      return;
    }

    if (RESTARTABLE_ERRORS.has(code as ExpoSpeechRecognitionErrorCode)) {
      return;
    }

    setError(code);

    if (code === 'not-allowed') {
      shouldBeListening.current = false;
      clearRestart();
    }
  });

  // Cleanup on unmount

  useEffect(() => {
    return () => {
      shouldBeListening.current = false;
      clearRestart();
      if (profileStaleTimerRef.current) {
        clearTimeout(profileStaleTimerRef.current);
        profileStaleTimerRef.current = null;
      }
      try {
        ExpoSpeechRecognitionModule.abort();
      } catch {
        // Component unmounting - swallow.
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Return the public API

  return {
    text,
    segments,
    interimText,
    isListening,
    error,
    isAvailable,
    startListening,
    stopListening,
    resetTranscript,
  };
}
