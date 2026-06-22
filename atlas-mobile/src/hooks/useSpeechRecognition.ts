/**
 * Atlas – Hearing Assist Mode (Voice Processing Layer)
 *
 * Live captioning + speaker diarization. Two paths behind one stable interface:
 *
 *  1. PRIMARY — Deepgram Nova-3 streaming for transcription.
 *     Raw mono PCM is pushed in via {@link SpeechRecognitionState.pushPcm}
 *     (fed by `useAlarmDetector`'s `onPcm` tap, so there is a single mic owner).
 *
 *  2. Speaker labels — pitch-based (2+N)D clustering when `pitchHistoryRef`
 *     is wired from the alarm detector (preferred). Deepgram neural diarization
 *     is used only when pitch history is unavailable.
 *
 *  3. FALLBACK — `expo-speech-recognition` (on-device), with pitch clustering
 *     or pause-based speaker turns when Deepgram is unavailable.
 *
 * The public surface is unchanged from the previous (MFCC-clustering) version,
 * plus one additive `pushPcm` method that screens use to bridge the mic tap.
 */
import { useState, useCallback, useRef, useEffect } from 'react';
import type { RefObject } from 'react';
import { Platform } from 'react-native';
import Constants from 'expo-constants';

import { playHapticPattern } from '../utils/haptic_patterns';
import {
  assignSpeaker,
  weightedMedian,
  type SpeakerProfile,
} from '../utils/speaker_cluster';
import type { PitchSample } from './useAlarmDetector';
import { MIN_DIARIZATION_SAMPLES, NUM_MEL_BANDS, PROFILE_STALE_MS } from '../theme';
import {
  DeepgramTransport,
  wordsToSegments,
  mergeSegments,
  extractAlternative,
  type DiarizedSegment,
  type DeepgramResultMessage,
} from './deepgram';
import {
  ExpoSpeechRecognitionModule,
  useSpeechRecognitionEvent,
  type ExpoSpeechRecognitionErrorCode,
} from 'expo-speech-recognition';

const TAG = '[SpeechRecognition]';

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
   * When true, the fallback recogniser auto-restarts after a silence / timeout
   * event so captioning feels continuous. (Deepgram streaming is inherently
   * continuous.) @default true
   */
  continuous?: boolean;
  /** Delay (ms) before auto-restarting the fallback recogniser. @default 300 */
  restartDelayMs?: number;
  /**
   * Silence (ms) between fallback final segments that triggers a new speaker
   * turn. Ignored by the Deepgram path (neural diarization). @default 1800
   */
  speakerPauseMs?: number;
  /** Fire a gentle haptic tap on speaker change. @default true */
  hapticPatternsEnabled?: boolean;
  /**
   * Rolling pitch history from the alarm detector's AnalyserNode.
   * When provided, speaker assignment uses pitch-based clustering
   * (preferred over Deepgram/pause-based labels).
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
  /**
   * Feed one mono Float32 PCM frame (range [-1,1]) plus its sample rate.
   * Wire this to `useAlarmDetector({ onPcm })` so the alarm detector remains
   * the single mic owner. No-op while not listening / in fallback mode.
   */
  pushPcm: (samples: Float32Array, sampleRate: number) => void;
}

// Error codes that should trigger an automatic fallback restart rather than
// being surfaced to the consumer as hard errors.
const RESTARTABLE_ERRORS: Set<ExpoSpeechRecognitionErrorCode> = new Set([
  'no-speech',
  'speech-timeout',
  'network',
  'client',
]);

/** Read the Deepgram API key injected via app.config.ts `extra`. */
function getDeepgramKey(): string {
  const key = (Constants.expoConfig?.extra as { deepgramKey?: string } | undefined)
    ?.deepgramKey;
  return typeof key === 'string' ? key.trim() : '';
}

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

  // Config
  const deepgramKey = getDeepgramKey();
  const expoAvailable = ExpoSpeechRecognitionModule.isRecognitionAvailable();
  const isAvailable = deepgramKey.length > 0 || expoAvailable;

  // Log config on first render only (ref trick avoids re-log on re-renders)
  const loggedConfigRef = useRef(false);
  if (!loggedConfigRef.current) {
    loggedConfigRef.current = true;
    console.log(
      `${TAG} init` +
      ` | deepgramKey=${deepgramKey.length > 0 ? `SET (${deepgramKey.length} chars)` : 'NOT SET'}` +
      ` | expoAvailable=${expoAvailable}` +
      ` | isAvailable=${isAvailable}` +
      ` | platform=${Platform.OS}`,
    );
  }

  // Shared refs
  const shouldBeListening = useRef(false);
  /** True once we've switched to the expo-speech-recognition fallback. */
  const usingFallbackRef = useRef(false);
  /** Accumulated finalized segments (source of truth for `segments`/`text`). */
  const finalSegmentsRef = useRef<DiarizedSegment[]>([]);
  const lastSpeakerRef = useRef<string | null>(null);
  const hapticEnabledRef = useRef(hapticPatternsEnabled);
  hapticEnabledRef.current = hapticPatternsEnabled;

  // Pitch-based diarization refs
  const segmentStartTimeRef = useRef(0);
  const speakerProfilesRef = useRef<SpeakerProfile[]>([]);
  const lastActiveSpeakerRef = useRef<string | null>(null);
  const profileStaleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const resetProfileStaleTimer = useCallback(() => {
    if (profileStaleTimerRef.current) clearTimeout(profileStaleTimerRef.current);
    profileStaleTimerRef.current = setTimeout(() => {
      speakerProfilesRef.current = [];
      lastActiveSpeakerRef.current = null;
    }, PROFILE_STALE_MS);
  }, []);

  const resolveSpeakerByPause = useCallback(
    (elapsedMs: number): string => {
      const shouldChange =
        speakerPauseMs > 0 &&
        elapsedMs >= speakerPauseMs &&
        finalSegmentsRef.current.length > 0;
      if (shouldChange) currentSpeakerIndexRef.current += 1;
      return `Speaker ${currentSpeakerIndexRef.current}`;
    },
    [speakerPauseMs],
  );

  const resolveSpeakerFromPitch = useCallback(
    (segStart: number, segEnd: number, elapsedMs: number): string => {
      if (!pitchHistoryRef) {
        return resolveSpeakerByPause(elapsedMs);
      }

      const pitchHistory = pitchHistoryRef.current;
      const segmentSamples =
        segStart > 0
          ? pitchHistory.filter((s) => s.time >= segStart && s.time <= segEnd)
          : [];

      const segmentPitches = segmentSamples.map((s) => s.hz);
      const segmentCentroids = segmentSamples.map((s) => s.centroid);
      const segmentConfidences = segmentSamples.map((s) => s.confidence);
      const segmentFeatures = segmentSamples.map((s) => s.features ?? []);

      const wPitch = weightedMedian(segmentPitches, segmentConfidences);
      const wCentroid = weightedMedian(segmentCentroids, segmentConfidences);
      const wFeatures: number[] = Array.from({ length: NUM_MEL_BANDS }, (_, k) => {
        const bandValues = segmentFeatures.map((f) => f[k] ?? 0);
        return weightedMedian(bandValues, segmentConfidences) ?? 0;
      });

      if (
        wPitch !== null &&
        wCentroid !== null &&
        segmentPitches.length >= MIN_DIARIZATION_SAMPLES
      ) {
        const result = assignSpeaker(
          wPitch,
          wCentroid,
          wFeatures,
          speakerProfilesRef.current,
          undefined,
          lastActiveSpeakerRef.current,
        );
        speakerProfilesRef.current = result.profiles;
        lastActiveSpeakerRef.current = result.label;
        return result.label;
      }

      const lastSeg =
        finalSegmentsRef.current.length > 0
          ? finalSegmentsRef.current[finalSegmentsRef.current.length - 1]
          : null;
      return lastSeg?.speaker ?? 'Speaker A';
    },
    [pitchHistoryRef, resolveSpeakerByPause],
  );

  // Deepgram refs
  const transportRef = useRef<DeepgramTransport | null>(null);

  // Fallback (expo) refs
  const restartTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastFinalTimestampRef = useRef(0);
  const currentSpeakerIndexRef = useRef(1);
  const savedIOSAudioSession = useRef<{
    category: string;
    categoryOptions: string[];
    mode: string;
  } | null>(null);

  // ---- Shared transcript helpers --------------------------------------------

  /** Recompute `text` from finalized segments + the current interim text. */
  const publishText = useCallback((interim: string) => {
    const finalText = finalSegmentsRef.current.map((s) => s.text).join(' ').trim();
    const combined = interim ? `${finalText} ${interim}`.trim() : finalText;
    setText(combined);
  }, []);

  /** Commit a batch of finalized segments, merging + firing speaker haptics. */
  const commitSegments = useCallback((incoming: DiarizedSegment[]) => {
    if (incoming.length === 0) return;

    // Speaker-change haptic (compare incoming's first speaker to last known).
    const firstNew = incoming[0].speaker;
    if (
      lastSpeakerRef.current != null &&
      lastSpeakerRef.current !== firstNew &&
      hapticEnabledRef.current
    ) {
      playHapticPattern('newSpeaker');
    }

    finalSegmentsRef.current = mergeSegments(finalSegmentsRef.current, incoming);
    lastSpeakerRef.current =
      finalSegmentsRef.current[finalSegmentsRef.current.length - 1]?.speaker ??
      lastSpeakerRef.current;

    setSegments(finalSegmentsRef.current.map((s) => ({ ...s })));
  }, []);

  // ---- Deepgram path --------------------------------------------------------

  const handleDeepgramMessage = useCallback(
    (msg: DeepgramResultMessage) => {
      const { transcript, words } = extractAlternative(msg);

      if (msg.is_final) {
        const segment = transcript.trim();
        if (!segment) {
          setInterimText('');
          publishText('');
          return;
        }

        const now = Date.now();
        const elapsed =
          lastFinalTimestampRef.current > 0 ? now - lastFinalTimestampRef.current : 0;
        lastFinalTimestampRef.current = now;

        const segStart = segmentStartTimeRef.current || now - 2000;
        segmentStartTimeRef.current = 0;
        resetProfileStaleTimer();

        const speaker = pitchHistoryRef
          ? resolveSpeakerFromPitch(segStart, now, elapsed)
          : wordsToSegments(words, segment)[0]?.speaker ?? 'Speaker 1';

        commitSegments([{ speaker, text: segment }]);
        setInterimText('');
        publishText('');
      } else {
        if (segmentStartTimeRef.current === 0 && transcript.trim()) {
          segmentStartTimeRef.current = Date.now();
        }
        setInterimText(transcript);
        publishText(transcript);
      }
    },
    [
      commitSegments,
      publishText,
      pitchHistoryRef,
      resolveSpeakerFromPitch,
      resetProfileStaleTimer,
    ],
  );

  // ---- Fallback (expo-speech-recognition) -----------------------------------

  const clearRestart = useCallback(() => {
    if (restartTimer.current) {
      clearTimeout(restartTimer.current);
      restartTimer.current = null;
    }
  }, []);

  const saveIOSAudioSession = useCallback(() => {
    if (Platform.OS !== 'ios') return;
    try {
      savedIOSAudioSession.current =
        ExpoSpeechRecognitionModule.getAudioSessionCategoryAndOptionsIOS();
    } catch {
      savedIOSAudioSession.current = null;
    }
  }, []);

  const restoreIOSAudioSession = useCallback(() => {
    if (Platform.OS !== 'ios' || !savedIOSAudioSession.current) return;
    try {
      ExpoSpeechRecognitionModule.setCategoryIOS(
        savedIOSAudioSession.current as Parameters<
          typeof ExpoSpeechRecognitionModule.setCategoryIOS
        >[0],
      );
    } catch {
      /* best-effort */
    }
    savedIOSAudioSession.current = null;
  }, []);

  const beginExpoRecognition = useCallback(() => {
    const supportsOnDevice =
      ExpoSpeechRecognitionModule.supportsOnDeviceRecognition();
    ExpoSpeechRecognitionModule.start({
      lang,
      interimResults: true,
      continuous,
      requiresOnDeviceRecognition: supportsOnDevice,
      addsPunctuation: true,
      androidIntentOptions: {
        EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS: 10000,
        EXTRA_MASK_OFFENSIVE_WORDS: false,
      },
      iosCategory: {
        category: 'playAndRecord',
        categoryOptions: ['defaultToSpeaker', 'allowBluetooth'],
        mode: 'measurement',
      },
      iosTaskHint: 'dictation',
    });
  }, [lang, continuous]);

  /** Switch from the Deepgram path to the expo fallback (R6). */
  const switchToFallback = useCallback(() => {
    console.log(`${TAG} switchToFallback() called | usingFallback=${usingFallbackRef.current} shouldBeListening=${shouldBeListening.current} expoAvailable=${expoAvailable}`);
    if (usingFallbackRef.current || !shouldBeListening.current) return;
    if (!expoAvailable) {
      console.error(`${TAG} switchToFallback: expo-speech-recognition not available – setting error`);
      setError('service-not-allowed');
      return;
    }
    usingFallbackRef.current = true;
    console.log(`${TAG} switchToFallback: switching to expo-speech-recognition`);

    if (transportRef.current) {
      transportRef.current.close();
      transportRef.current = null;
    }

    saveIOSAudioSession();
    try {
      beginExpoRecognition();
      console.log(`${TAG} switchToFallback: beginExpoRecognition() called`);
    } catch (e) {
      console.error(`${TAG} switchToFallback: beginExpoRecognition() threw:`, e);
      setError('service-not-allowed');
    }
  }, [expoAvailable, saveIOSAudioSession, beginExpoRecognition]);

  // ---- PCM bridge -----------------------------------------------------------

  const pushPcmCallCountRef = useRef(0);
  const pushPcm = useCallback(
    (samples: Float32Array, sampleRate: number) => {
      pushPcmCallCountRef.current += 1;
      const count = pushPcmCallCountRef.current;
      // Log the first few frames and then every 100th
      if (count <= 3 || count % 100 === 0) {
        console.log(
          `${TAG} pushPcm #${count}` +
          ` | frames=${samples.length}` +
          ` | sampleRate=${sampleRate}` +
          ` | shouldBeListening=${shouldBeListening.current}` +
          ` | usingFallback=${usingFallbackRef.current}` +
          ` | hasTransport=${transportRef.current != null}`,
        );
      }
      if (!shouldBeListening.current || usingFallbackRef.current) return;
      transportRef.current?.pushPcm(samples, sampleRate);
    },
    [],
  );

  // ---- Public methods -------------------------------------------------------

  const startListening = useCallback(async () => {
    console.log(`${TAG} startListening() called | isAvailable=${isAvailable} deepgramKey=${deepgramKey.length > 0 ? 'SET' : 'EMPTY'}`);

    if (!isAvailable) {
      console.error(`${TAG} startListening: not available – setting service-not-allowed error`);
      setError('service-not-allowed');
      return;
    }

    // Request mic (+ speech) permission – gracefully bail if denied.
    console.log(`${TAG} startListening: requesting permissions…`);
    try {
      const result = await ExpoSpeechRecognitionModule.requestPermissionsAsync();
      console.log(`${TAG} startListening: permission granted=${result.granted}`);
      if (!result.granted) {
        console.error(`${TAG} startListening: mic permission denied`);
        setError('not-allowed');
        return;
      }
    } catch (e) {
      console.error(`${TAG} startListening: requestPermissionsAsync threw:`, e);
      setError('not-allowed');
      return;
    }

    setError(null);
    usingFallbackRef.current = false;
    shouldBeListening.current = true;
    setIsListening(true);
    pushPcmCallCountRef.current = 0;

    if (deepgramKey.length > 0) {
      // Deepgram primary. The socket opens lazily on the first pushed PCM frame.
      console.log(`${TAG} startListening: creating DeepgramTransport (socket opens on first PCM frame)`);
      transportRef.current = new DeepgramTransport(deepgramKey, {
        onMessage: handleDeepgramMessage,
        onOpen: () => console.log(`${TAG} Deepgram WebSocket OPENED`),
        onClose: (code, reason) => console.log(`${TAG} Deepgram WebSocket CLOSED code=${code} reason="${reason}"`) ,
        onError: (err) => {
          // Network / auth failure → fall back so the demo keeps working.
          console.error(`${TAG} Deepgram transport error:`, err?.message ?? err);
          switchToFallback();
        },
      });
      console.log(`${TAG} startListening: DeepgramTransport created – waiting for PCM frames via pushPcm`);
    } else {
      // No key configured → straight to fallback.
      console.log(`${TAG} startListening: no Deepgram key – going straight to expo-speech-recognition`);
      usingFallbackRef.current = true;
      saveIOSAudioSession();
      try {
        beginExpoRecognition();
        console.log(`${TAG} startListening: beginExpoRecognition() called`);
      } catch (e) {
        console.error(`${TAG} startListening: beginExpoRecognition() threw:`, e);
      }
    }
  }, [
    isAvailable,
    deepgramKey,
    handleDeepgramMessage,
    switchToFallback,
    saveIOSAudioSession,
    beginExpoRecognition,
  ]);

  const stopListening = useCallback(async () => {
    console.log(`${TAG} stopListening() called | usingFallback=${usingFallbackRef.current} hasTransport=${transportRef.current != null}`);
    shouldBeListening.current = false;
    usingFallbackRef.current = false;
    clearRestart();
    setIsListening(false);

    if (transportRef.current) {
      console.log(`${TAG} stopListening: closing DeepgramTransport`);
      transportRef.current.close();
      transportRef.current = null;
    }

    if (usingFallbackRef.current) {
      console.log(`${TAG} stopListening: stopping expo-speech-recognition`);
      try {
        ExpoSpeechRecognitionModule.stop();
      } catch (e) {
        console.warn(`${TAG} stopListening: ExpoSpeechRecognitionModule.stop() threw:`, e);
      }
    }
    setInterimText('');
    console.log(`${TAG} stopListening() complete`);
  }, [clearRestart]);

  const resetTranscript = useCallback(() => {
    finalSegmentsRef.current = [];
    lastSpeakerRef.current = null;
    lastFinalTimestampRef.current = 0;
    currentSpeakerIndexRef.current = 1;
    segmentStartTimeRef.current = 0;
    speakerProfilesRef.current = [];
    lastActiveSpeakerRef.current = null;
    if (profileStaleTimerRef.current) {
      clearTimeout(profileStaleTimerRef.current);
      profileStaleTimerRef.current = null;
    }
    setText('');
    setSegments([]);
    setInterimText('');
  }, []);

  // ---- expo event listeners (only act while in fallback mode) ---------------

  useSpeechRecognitionEvent('start', () => {
    console.log(`${TAG} expo event: start | usingFallback=${usingFallbackRef.current}`);
    if (!usingFallbackRef.current || !shouldBeListening.current) return;
    setIsListening(true);
    setError(null);
  });

  useSpeechRecognitionEvent('end', () => {
    console.log(`${TAG} expo event: end | usingFallback=${usingFallbackRef.current} shouldBeListening=${shouldBeListening.current} continuous=${continuous}`);
    if (!usingFallbackRef.current) return;
    restoreIOSAudioSession();
    if (shouldBeListening.current && continuous) {
      clearRestart();
      restartTimer.current = setTimeout(() => {
        if (shouldBeListening.current && usingFallbackRef.current) {
          console.log(`${TAG} expo: auto-restarting recognition after ${restartDelayMs}ms`);
          saveIOSAudioSession();
          beginExpoRecognition();
        }
      }, restartDelayMs);
    } else {
      setIsListening(false);
    }
  });

  useSpeechRecognitionEvent('result', (event) => {
    const best = event.results[0];
    console.log(`${TAG} expo event: result | isFinal=${event.isFinal} usingFallback=${usingFallbackRef.current} transcript="${best?.transcript?.slice(0, 60)}"`);
    if (!usingFallbackRef.current) return;
    if (!best) return;

    if (event.isFinal) {
      const segment = best.transcript.trim();
      if (!segment) return;

      const now = Date.now();
      const elapsed =
        lastFinalTimestampRef.current > 0 ? now - lastFinalTimestampRef.current : 0;
      lastFinalTimestampRef.current = now;

      const segStart = segmentStartTimeRef.current || now - 2000;
      segmentStartTimeRef.current = 0;
      resetProfileStaleTimer();

      const speaker = pitchHistoryRef
        ? resolveSpeakerFromPitch(segStart, now, elapsed)
        : resolveSpeakerByPause(elapsed);

      console.log(`${TAG} expo final segment: speaker="${speaker}" text="${segment.slice(0, 60)}"`);
      commitSegments([{ speaker, text: segment }]);
      setInterimText('');
      publishText('');
    } else {
      if (segmentStartTimeRef.current === 0) {
        segmentStartTimeRef.current = Date.now();
      }
      const interim = best.transcript.trim();
      setInterimText(interim);
      publishText(interim);
    }
  });

  useSpeechRecognitionEvent('error', (event) => {
    const code = event.error;
    console.error(`${TAG} expo event: error code="${code}" | usingFallback=${usingFallbackRef.current} | restartable=${RESTARTABLE_ERRORS.has(code as ExpoSpeechRecognitionErrorCode)}`);
    if (!usingFallbackRef.current) return;
    if (RESTARTABLE_ERRORS.has(code as ExpoSpeechRecognitionErrorCode)) return;
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
      if (transportRef.current) {
        transportRef.current.close();
        transportRef.current = null;
      }
      try {
        ExpoSpeechRecognitionModule.abort();
      } catch {
        /* unmounting */
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
    pushPcm,
  };
}
