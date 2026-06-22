/**
 * Vision assist orchestrator: Gemini when online, on-device YOLO when offline.
 * Switches silently — 5-pulse haptic only when falling back to local.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import type { Camera } from 'react-native-vision-camera';

import { playOfflineFallbackHaptic } from '../utils/haptic_patterns';
import type { Detection } from '../utils/tensor_decoder';
import { useGeminiVision } from './useGeminiVision';
import { useLocalYoloVision } from './useLocalYoloVision';
import { useMlkitOcr } from './useMlkitOcr';
import { isLikelyNetworkError, useNetworkReachable } from './useNetworkReachable';

export interface UseVisionAssistOptions {
  cameraRef: React.RefObject<Camera | null>;
  apiKey: string;
  isActive: boolean;
  ttsRate?: number;
}

export interface UseVisionAssistResult {
  detections: Detection[];
  fps: number;
  ocrText: string;
  /** Attach to Camera when using the on-device pipeline. */
  frameProcessor?: ReturnType<typeof useLocalYoloVision>['frameProcessor'];
  /** Camera snapshots for cloud vision + offline OCR. */
  needsPhoto: boolean;
  /** Model loaded (local) or cloud path ready. */
  pipelineReady: boolean;
  modelLoading: boolean;
}

export function useVisionAssist({
  cameraRef,
  apiKey,
  isActive,
  ttsRate = 0.95,
}: UseVisionAssistOptions): UseVisionAssistResult {
  const online = useNetworkReachable();
  const [cloudBlocked, setCloudBlocked] = useState(false);
  const prevCloudRef = useRef<boolean | null>(null);

  useEffect(() => {
    if (online === true) {
      setCloudBlocked(false);
    }
  }, [online]);

  const hasCloudKey = apiKey.length > 0;
  const useCloud = hasCloudKey && online !== false && !cloudBlocked;

  const handleConnectivityFailure = useCallback(() => {
    setCloudBlocked(true);
  }, []);

  const gemini = useGeminiVision({
    cameraRef,
    apiKey,
    isActive: isActive && useCloud,
    ttsRate,
    onConnectivityFailure: handleConnectivityFailure,
  });

  const local = useLocalYoloVision({
    isActive: isActive && !useCloud,
    ttsRate,
  });

  const captureForOcr = useCallback(async (): Promise<string | null> => {
    const cam = cameraRef.current;
    if (!cam) return null;
    try {
      const snapshot = await cam.takeSnapshot({ quality: 60 });
      const rawPath = snapshot.path;
      return rawPath.startsWith('file://') ? rawPath : `file://${rawPath}`;
    } catch {
      return null;
    }
  }, [cameraRef]);

  const { ocrText: localOcrText } = useMlkitOcr({
    enabled: isActive && !useCloud,
    capture: captureForOcr,
  });

  useEffect(() => {
    if (!isActive) {
      prevCloudRef.current = null;
      return;
    }

    if (prevCloudRef.current === true && !useCloud) {
      void playOfflineFallbackHaptic();
    }
    prevCloudRef.current = useCloud;
  }, [useCloud, isActive]);

  if (useCloud) {
    return {
      detections: gemini.detections,
      fps: gemini.fps,
      ocrText: gemini.ocrText,
      frameProcessor: undefined,
      needsPhoto: true,
      pipelineReady: true,
      modelLoading: false,
    };
  }

  return {
    detections: local.detections,
    fps: local.fps,
    ocrText: localOcrText,
    frameProcessor: local.frameProcessor,
    needsPhoto: true,
    pipelineReady: local.modelState === 'loaded',
    modelLoading: local.modelState === 'loading',
  };
}

export { isLikelyNetworkError };
