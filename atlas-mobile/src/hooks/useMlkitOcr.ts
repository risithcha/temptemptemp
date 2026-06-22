/**
 * Atlas Mobile - Google ML Kit Text Recognition (v2) OCR Hook
 *
 * Replaces the VisionCamera worklet OCR (`react-native-vision-camera-ocr-plus`)
 * with Google ML Kit Text Recognition v2 (`@react-native-ml-kit/text-recognition`).
 *
 * ML Kit runs on the JS thread and recognizes from a file URI; it CANNOT run
 * inside a VisionCamera frame-processor worklet.  So instead of decoding frames
 * in the worklet, this hook polls on the JS thread at the OCR cadence:
 *   1. ask the caller to capture a still (Camera.takeSnapshot -> file path),
 *   2. run ML Kit recognition on that file,
 *   3. sanitize and publish the text.
 *
 * The poll is strictly non-reentrant: a slow capture + recognize cycle can take
 * longer than the interval, so a new tick is skipped while one is in flight to
 * avoid stacking snapshots and starving the camera.
 */
import { useState, useEffect, useRef } from 'react';
import TextRecognition from '@react-native-ml-kit/text-recognition';

import { sanitizeOcrText } from '../utils/ocr_utils';
import { OCR_FPS } from '../theme';

export interface UseMlkitOcrOptions {
  /** Master switch – polling only runs while this is true. */
  enabled: boolean;
  /**
   * Capture a still frame and resolve to a readable image URI (or `null` if
   * capture failed / the camera isn't ready).  The caller owns the camera ref;
   * this hook stays presentation-agnostic.
   */
  capture: () => Promise<string | null>;
  /** Poll interval (ms).  Defaults to the shared OCR cadence (`OCR_FPS`). */
  intervalMs?: number;
}

export interface UseMlkitOcrResult {
  /** Latest sanitized OCR text (empty string when nothing recognized). */
  ocrText: string;
}

export function useMlkitOcr({
  enabled,
  capture,
  intervalMs = Math.round(1000 / OCR_FPS),
}: UseMlkitOcrOptions): UseMlkitOcrResult {
  const [ocrText, setOcrText] = useState('');

  /** True while a capture+recognize cycle is in flight (non-reentrant guard). */
  const inFlightRef = useRef(false);
  /** Tracks mount state so a late async resolve can't setState after unmount. */
  const mountedRef = useRef(true);
  /** Keep the latest capture fn without re-subscribing the interval each render. */
  const captureRef = useRef(capture);
  captureRef.current = capture;

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    if (!enabled) return;

    let cancelled = false;

    const tick = async () => {
      if (inFlightRef.current) return; // previous cycle still running – skip
      inFlightRef.current = true;
      try {
        const uri = await captureRef.current();
        if (cancelled || uri == null) return;

        const result = await TextRecognition.recognize(uri);
        if (cancelled) return;

        const cleaned = sanitizeOcrText(result?.text ?? '');
        if (cleaned && mountedRef.current) {
          setOcrText(cleaned);
        }
      } catch {
        // Snapshot/recognition can transiently fail (camera busy, no text) –
        // swallow and try again on the next tick.
      } finally {
        inFlightRef.current = false;
      }
    };

    const timer = setInterval(tick, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [enabled, intervalMs]);

  return { ocrText };
}
