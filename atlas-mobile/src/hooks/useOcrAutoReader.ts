/**
 * OCR accessibility controller for Vision Assist.
 *
 * • New readable text → TTS says "Text detected" (does not read the body).
 * • Single tap → play / pause / resume the OCR text.
 * • Double tap → restart reading from the beginning.
 *
 * Pause/resume uses native pause on iOS; on Android we track char offset via
 * onBoundary because expo-speech pause is iOS-only.
 */
import { useEffect, useRef, useCallback, useState } from 'react';
import { Platform } from 'react-native';
import * as Speech from 'expo-speech';
import type { SpeechOptions } from 'expo-speech';

import { similarity } from '../utils/string_similarity';
import { OCR_SIMILARITY_THRESHOLD, OCR_CACHE_TTL_MS } from '../theme';

const TEXT_DETECTED_PROMPT = 'Text detected';
const DOUBLE_TAP_MS = 320;

type PlaybackState = 'idle' | 'playing' | 'paused';

interface CacheEntry {
  text: string;
  time: number;
}

export interface UseOcrAutoReaderOptions {
  ttsRate?: number;
  enabled?: boolean;
  similarityThreshold?: number;
  cacheTtlMs?: number;
}

export interface UseOcrAutoReaderResult {
  /** Single tap: play / pause / resume. Double tap: restart from top. */
  handleTap: () => void;
  playbackState: PlaybackState;
  hasReadableText: boolean;
}

export function useOcrAutoReader(
  ocrText: string,
  isScreenActive: boolean,
  options: UseOcrAutoReaderOptions = {},
): UseOcrAutoReaderResult {
  const {
    ttsRate = 0.95,
    enabled = true,
    similarityThreshold = OCR_SIMILARITY_THRESHOLD,
    cacheTtlMs = OCR_CACHE_TTL_MS,
  } = options;

  const [playbackState, setPlaybackState] = useState<PlaybackState>('idle');
  const cacheRef = useRef<CacheEntry[]>([]);
  const latestTextRef = useRef(ocrText);
  latestTextRef.current = ocrText;

  const playbackStateRef = useRef<PlaybackState>('idle');
  const charOffsetRef = useRef(0);
  const lastTapTimeRef = useRef(0);
  const singleTapTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const announcingRef = useRef(false);
  const skipStoppedRef = useRef(false);

  const syncPlaybackState = useCallback((next: PlaybackState) => {
    playbackStateRef.current = next;
    setPlaybackState(next);
  }, []);

  const evictStale = useCallback(() => {
    const cutoff = Date.now() - cacheTtlMs;
    cacheRef.current = cacheRef.current.filter((e) => e.time > cutoff);
  }, [cacheTtlMs]);

  const stopPlayback = useCallback(() => {
    skipStoppedRef.current = true;
    Speech.stop();
    charOffsetRef.current = 0;
    syncPlaybackState('idle');
  }, [syncPlaybackState]);

  const speakPrompt = useCallback(
    (phrase: string, onDone?: () => void) => {
      announcingRef.current = true;
      Speech.stop();
      Speech.speak(phrase, {
        language: 'en-US',
        rate: ttsRate,
        pitch: 1.0,
        onDone: () => {
          announcingRef.current = false;
          onDone?.();
        },
        onStopped: () => {
          announcingRef.current = false;
        },
      });
    },
    [ttsRate],
  );

  const speakBodyFrom = useCallback(
    (offset: number) => {
      const full = latestTextRef.current.trim();
      if (!full) return;

      const slice = full.slice(offset);
      if (!slice) {
        syncPlaybackState('idle');
        charOffsetRef.current = 0;
        return;
      }

      skipStoppedRef.current = true;
      Speech.stop();
      charOffsetRef.current = offset;
      syncPlaybackState('playing');

      Speech.speak(slice, {
        language: 'en-US',
        rate: ttsRate,
        pitch: 1.0,
        onBoundary: (event: Parameters<NonNullable<SpeechOptions['onBoundary']>>[0]) => {
          charOffsetRef.current = offset + (event.charIndex ?? 0);
        },
        onDone: () => {
          charOffsetRef.current = 0;
          syncPlaybackState('idle');
        },
        onStopped: () => {
          if (skipStoppedRef.current) {
            skipStoppedRef.current = false;
            return;
          }
          if (playbackStateRef.current === 'playing') {
            syncPlaybackState('paused');
          }
        },
      });
    },
    [ttsRate, syncPlaybackState],
  );

  const pausePlayback = useCallback(async () => {
    if (playbackStateRef.current !== 'playing') return;

    if (Platform.OS === 'ios') {
      try {
        await Speech.pause();
        syncPlaybackState('paused');
        return;
      } catch {
        // fall through to offset-based pause
      }
    }

    skipStoppedRef.current = true;
    Speech.stop();
    syncPlaybackState('paused');
  }, [syncPlaybackState]);

  const resumePlayback = useCallback(async () => {
    if (playbackStateRef.current !== 'paused') return;

    if (Platform.OS === 'ios') {
      try {
        await Speech.resume();
        syncPlaybackState('playing');
        return;
      } catch {
        // fall through to offset-based resume
      }
    }

    speakBodyFrom(charOffsetRef.current);
  }, [speakBodyFrom, syncPlaybackState]);

  const startPlayback = useCallback(() => {
    charOffsetRef.current = 0;
    speakBodyFrom(0);
  }, [speakBodyFrom]);

  const restartPlayback = useCallback(() => {
    stopPlayback();
    startPlayback();
  }, [stopPlayback, startPlayback]);

  const handleTap = useCallback(() => {
    if (!enabled) return;

    const now = Date.now();
    const delta = now - lastTapTimeRef.current;

    const clearPendingSingleTap = () => {
      if (singleTapTimerRef.current) {
        clearTimeout(singleTapTimerRef.current);
        singleTapTimerRef.current = null;
      }
    };

    // Double tap → restart OCR reading from the top.
    if (delta > 0 && delta < DOUBLE_TAP_MS) {
      clearPendingSingleTap();
      lastTapTimeRef.current = 0;
      if (latestTextRef.current.trim()) {
        restartPlayback();
      }
      return;
    }

    lastTapTimeRef.current = now;

    // Pause / stop immediately — do not wait for the double-tap window.
    if (playbackStateRef.current === 'playing') {
      clearPendingSingleTap();
      void pausePlayback();
      return;
    }

    if (playbackStateRef.current === 'paused') {
      clearPendingSingleTap();
      void resumePlayback();
      return;
    }

    if (announcingRef.current) {
      clearPendingSingleTap();
      skipStoppedRef.current = true;
      Speech.stop();
      announcingRef.current = false;
      return;
    }

    void Speech.isSpeakingAsync().then((speaking) => {
      if (speaking) {
        clearPendingSingleTap();
        skipStoppedRef.current = true;
        Speech.stop();
        return;
      }

      // Idle: wait briefly so a second tap can register as double-tap (restart).
      clearPendingSingleTap();
      singleTapTimerRef.current = setTimeout(() => {
        singleTapTimerRef.current = null;
        lastTapTimeRef.current = 0;
        if (latestTextRef.current.trim()) {
          startPlayback();
        }
      }, DOUBLE_TAP_MS);
    });
  }, [enabled, pausePlayback, resumePlayback, restartPlayback, startPlayback]);

  // New OCR text → announce "Text detected" only.
  useEffect(() => {
    if (!isScreenActive || !enabled) return;
    const trimmed = ocrText.trim();
    if (!trimmed) return;

    evictStale();
    const cache = cacheRef.current;
    const mostRecent = cache.length > 0 ? cache[cache.length - 1] : null;

    if (mostRecent) {
      const sim = similarity(trimmed.toLowerCase(), mostRecent.text.toLowerCase());
      if (sim >= similarityThreshold) return;
    }

    cache.push({ text: trimmed, time: Date.now() });
    if (cache.length > 5) cache.shift();

    stopPlayback();
    speakPrompt(TEXT_DETECTED_PROMPT);
  }, [
    ocrText,
    isScreenActive,
    enabled,
    similarityThreshold,
    evictStale,
    speakPrompt,
    stopPlayback,
  ]);

  useEffect(() => {
    if (!isScreenActive || !enabled) {
      if (singleTapTimerRef.current) {
        clearTimeout(singleTapTimerRef.current);
        singleTapTimerRef.current = null;
      }
      stopPlayback();
      Speech.stop();
    }
  }, [isScreenActive, enabled, stopPlayback]);

  useEffect(() => {
    return () => {
      if (singleTapTimerRef.current) {
        clearTimeout(singleTapTimerRef.current);
      }
    };
  }, []);

  return {
    handleTap,
    playbackState,
    hasReadableText: ocrText.trim().length > 0,
  };
}
