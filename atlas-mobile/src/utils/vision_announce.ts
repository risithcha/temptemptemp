import { useEffect, useRef, useCallback } from 'react';
import * as Speech from 'expo-speech';

import { getSpatialDirection, type Detection } from './tensor_decoder';
import { TTS_BATCH_DELAY_MS, TTS_COOLDOWN_MS } from '../theme';

export function buildVisionSentence(dets: Detection[]): string {
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

export function detectionAnnouncementKey(detections: Detection[]): string {
  if (detections.length === 0) return '';
  return detections
    .map((d) => `${d.label}@${getSpatialDirection(d.box)}`)
    .join('|');
}

export interface UseDetectionAnnouncementOptions {
  enabled?: boolean;
  ttsRate?: number;
}

/** Debounced, cooldown-aware TTS for vision detections (local or cloud). */
export function useDetectionAnnouncement(
  detections: Detection[],
  isActive: boolean,
  options: UseDetectionAnnouncementOptions = {},
): void {
  const { enabled = true, ttsRate = 0.95 } = options;

  const lastAnnouncedRef = useRef<Map<string, number>>(new Map());
  const batchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const latestDetectionsRef = useRef<Detection[]>(detections);
  latestDetectionsRef.current = detections;

  const announcementKey = detectionAnnouncementKey(detections);

  const clearBatchTimer = useCallback(() => {
    if (batchTimerRef.current) {
      clearTimeout(batchTimerRef.current);
      batchTimerRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (!isActive || !enabled) {
      clearBatchTimer();
      Speech.stop();
    }
  }, [isActive, enabled, clearBatchTimer]);

  useEffect(() => {
    if (!isActive || !enabled || !announcementKey) {
      return;
    }

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

      const sentence = buildVisionSentence(announceable);
      if (!sentence) return;

      Speech.speak(sentence, {
        language: 'en-US',
        rate: ttsRate,
        pitch: 1.0,
      });
    }, TTS_BATCH_DELAY_MS);
  }, [announcementKey, isActive, enabled, clearBatchTimer, ttsRate]);

  useEffect(() => {
    return () => {
      clearBatchTimer();
      Speech.stop();
    };
  }, [clearBatchTimer]);
}
