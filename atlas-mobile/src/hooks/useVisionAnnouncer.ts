/**
 * Atlas Mobile - Vision Assist TTS Announcer
 *
 * Announces newly detected objects via the device's TTS engine without
 * spamming the user.  Each label has an independent cooldown timer so
 * holding the camera on a keyboard doesn't repeat "keyboard" 30 times.
 *
 * Debounce:  After the first detection arrives, the hook waits
 *            TTS_BATCH_DELAY_MS before reading the *latest* detections
 *            and building a sentence.  Subsequent frames do NOT restart
 *            the timer; they just update the ref that the timer reads.
 *
 * Cooldown:  Each label can only be re-announced after TTS_COOLDOWN_MS
 *            has elapsed since its last announcement.
 */
import { useEffect, useRef, useCallback } from 'react';
import * as Speech from 'expo-speech';

import { getSpatialDirection, type Detection } from '../utils/tensor_decoder';
import { TTS_COOLDOWN_MS, TTS_BATCH_DELAY_MS } from '../theme';

export interface UseVisionAnnouncerOptions {
  /** Master switch - if false, no announcements fire. */
  enabled?: boolean;
  /** TTS voice rate (0.5-2.0). @default 0.95 */
  ttsRate?: number;
}

export function useVisionAnnouncer(
  detections: Detection[],
  isScreenActive: boolean,
  options: UseVisionAnnouncerOptions = {},
): void {
  const { enabled = true, ttsRate = 0.95 } = options;

  const lastAnnouncedRef = useRef<Map<string, number>>(new Map());
  const batchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const latestDetectionsRef = useRef<Detection[]>(detections);

  // Always keep the ref pointing at the latest detections so the timer
  // callback reads the freshest data when it finally fires.
  latestDetectionsRef.current = detections;

  const clearBatchTimer = useCallback(() => {
    if (batchTimerRef.current) {
      clearTimeout(batchTimerRef.current);
      batchTimerRef.current = null;
    }
  }, []);

  // Kill speech and timers when the screen deactivates or TTS is disabled.
  useEffect(() => {
    if (!isScreenActive || !enabled) {
      clearBatchTimer();
      Speech.stop();
    }
  }, [isScreenActive, enabled, clearBatchTimer]);

  // Start a debounce timer when detections appear. We only
  // start the timer if one isn't already running.  Incoming frames just
  // update the ref; they do NOT reset the timer.
  useEffect(() => {
    if (!isScreenActive || !enabled) return;

    if (detections.length === 0) {
      return;
    }

    // If a timer is already ticking, let it run; it will pick up the
    // latest detections from the ref when it fires.
    if (batchTimerRef.current != null) {
      return;
    }

    batchTimerRef.current = setTimeout(() => {
      batchTimerRef.current = null;

      const currentDetections = latestDetectionsRef.current;
      const now = Date.now();
      const announceable: Detection[] = [];

      for (const det of currentDetections) {
        const lastTime = lastAnnouncedRef.current.get(det.label);
        const elapsed = lastTime != null ? now - lastTime : Infinity;

        if (lastTime == null || elapsed >= TTS_COOLDOWN_MS) {
          announceable.push(det);
          lastAnnouncedRef.current.set(det.label, now);
        }
      }

      if (announceable.length === 0) {
        return;
      }

      const sentence = buildSentence(announceable);
      Speech.speak(sentence, {
        language: 'en-US',
        rate: ttsRate,
        pitch: 1.0,
      });
    }, TTS_BATCH_DELAY_MS);

    return () => {
      // Only clear if this specific effect instance unmounts (component unmount).
      // We intentionally do NOT clear the timer on dependency change.
    };
    // detections is intentionally omitted. We read from the ref.
    // The effect only re-runs when isScreenActive or enabled changes,
    // but we still need detections in the dep array to "wake up" the
    // effect when detections go from [] to [something].
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [detections.length > 0, isScreenActive, enabled, clearBatchTimer]);

  // Cleanup on unmount.
  useEffect(() => {
    return () => {
      clearBatchTimer();
      Speech.stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
}

/**
 * Build a natural-sounding sentence from a list of detections.
 * Includes spatial direction so blind users know where objects are.
 *
 * e.g. [person@left]              => "I see a person on the left."
 *      [person@left, laptop@center] => "I see a person on the left and a laptop in the center."
 */
function buildSentence(dets: Detection[]): string {
  if (dets.length === 0) return '';

  const directionPhrase: Record<string, string> = {
    Left: 'on the left',
    Center: 'in the center',
    Right: 'on the right',
  };

  const parts = dets.map((d) => {
    const dir = getSpatialDirection(d.box);
    return `a ${d.label} ${directionPhrase[dir]}`;
  });

  if (parts.length === 1) {
    return `I see ${parts[0]}.`;
  }
  if (parts.length === 2) {
    return `I see ${parts[0]} and ${parts[1]}.`;
  }

  const allButLast = parts.slice(0, -1).join(', ');
  const last = parts[parts.length - 1];
  return `I see ${allButLast}, and ${last}.`;
}
