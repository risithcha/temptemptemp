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
  useAlarmDetector,
  type AlarmDetectorState,
  type AlarmDetectorOptions,
  type AlarmAlert,
  type PitchSample,
} from './useAlarmDetector';

export {
  useOcrAutoReader,
  type UseOcrAutoReaderOptions,
  type UseOcrAutoReaderResult,
} from './useOcrAutoReader';
