// Atlas Mobile Hooks – barrel export

export {
  useSpeechRecognition,
  type UseSpeechRecognitionOptions,
  type SpeechRecognitionState,
  type TranscriptSegment,
} from './useSpeechRecognition';

export { useAppState } from './useAppState';
export { useAndroidBackHandler } from './useAndroidBackHandler';

export {
  useVisionAnnouncer,
  type UseVisionAnnouncerOptions,
} from './useVisionAnnouncer';

export {
  useVisionAssist,
  type UseVisionAssistOptions,
  type UseVisionAssistResult,
} from './useVisionAssist';

export {
  useLocalYoloVision,
  type UseLocalYoloVisionOptions,
  type UseLocalYoloVisionResult,
} from './useLocalYoloVision';

export { useNetworkReachable, isLikelyNetworkError } from './useNetworkReachable';

export {
  useAlarmDetector,
  type AlarmDetectorState,
  type AlarmDetectorOptions,
  type AlarmAlert,
  type PitchSample,
  type PcmListener,
} from './useAlarmDetector';

export {
  useOcrAutoReader,
  type UseOcrAutoReaderOptions,
  type UseOcrAutoReaderResult,
} from './useOcrAutoReader';

export {
  useMlkitOcr,
  type UseMlkitOcrOptions,
  type UseMlkitOcrResult,
} from './useMlkitOcr';

export {
  useGeminiVision,
  type UseGeminiVisionOptions,
  type UseGeminiVisionResult,
} from './useGeminiVision';

export * from './useGeminiAudio';
