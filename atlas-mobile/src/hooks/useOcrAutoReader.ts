/**
 * Atlas Mobile - Smart OCR Auto-Reader
 *
 * Watches the latest OCR text and auto-reads it via TTS only when the
 * content is substantially *new* (similarity to the most recent cache
 * entry falls below a configurable threshold).
 *
 * Also exposes `readLatestAloud()` for the tap-anywhere accessibility
 * fallback. This always reads regardless of the cache.
 *
 * TTS coordination: OCR auto-read calls `Speech.stop()` before speaking,
 * giving it priority over the object-detection announcer which has its
 * own cooldown-based recovery.
 */
import { useEffect, useRef, useCallback } from 'react';
import * as Speech from 'expo-speech';

import { similarity } from '../utils/string_similarity';
import { OCR_SIMILARITY_THRESHOLD, OCR_CACHE_TTL_MS } from '../theme';

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
  /** Force-read the latest OCR text aloud (tap-anywhere fallback). */
  readLatestAloud: () => void;
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

  const cacheRef = useRef<CacheEntry[]>([]);
  const latestTextRef = useRef(ocrText);
  latestTextRef.current = ocrText;

  // Evict stale cache entries.
  const evictStale = useCallback(() => {
    const cutoff = Date.now() - cacheTtlMs;
    cacheRef.current = cacheRef.current.filter((e) => e.time > cutoff);
  }, [cacheTtlMs]);

  // Speak a string with priority (stops any in-progress speech first).
  const speakWithPriority = useCallback(
    (text: string) => {
      Speech.stop();
      Speech.speak(text, {
        language: 'en-US',
        rate: ttsRate,
        pitch: 1.0,
      });
    },
    [ttsRate],
  );

  // --- Auto-read when ocrText changes ---
  useEffect(() => {
    if (!isScreenActive || !enabled) return;
    if (!ocrText || !ocrText.trim()) return;

    evictStale();

    const cache = cacheRef.current;
    const mostRecent = cache.length > 0 ? cache[cache.length - 1] : null;

    if (mostRecent) {
      const sim = similarity(
        ocrText.toLowerCase(),
        mostRecent.text.toLowerCase(),
      );
      if (sim >= similarityThreshold) {
        return; // Same text - skip
      }
    }

    speakWithPriority(ocrText);
    cache.push({ text: ocrText, time: Date.now() });

    // Keep the cache bounded.
    if (cache.length > 5) cache.shift();
  }, [ocrText, isScreenActive, enabled, similarityThreshold, evictStale, speakWithPriority]);

  // Cleanup on deactivation.
  useEffect(() => {
    if (!isScreenActive || !enabled) {
      Speech.stop();
    }
  }, [isScreenActive, enabled]);

  // --- Tap-anywhere fallback: always reads latest text ---
  const readLatestAloud = useCallback(() => {
    const text = latestTextRef.current;
    if (!text || !text.trim()) {
      return;
    }
    speakWithPriority(text);
  }, [speakWithPriority]);

  return { readLatestAloud };
}
