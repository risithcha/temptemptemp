/**
 * Atlas Mobile - Vision Assist TTS Announcer
 */
import type { Detection } from '../utils/tensor_decoder';
import { useDetectionAnnouncement } from '../utils/vision_announce';

export interface UseVisionAnnouncerOptions {
  enabled?: boolean;
  ttsRate?: number;
}

export function useVisionAnnouncer(
  detections: Detection[],
  isScreenActive: boolean,
  options: UseVisionAnnouncerOptions = {},
): void {
  useDetectionAnnouncement(detections, isScreenActive, options);
}
