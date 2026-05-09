// Atlas Mobile Utilities

// Haptic feedback
export { triggerHaptic, type HapticType } from './haptics';
export { playHapticPattern, type HapticPattern } from './haptic_patterns';

// OCR text processing
export { sanitizeOcrText } from './ocr_utils';

// String similarity (Levenshtein)
export { levenshtein, similarity } from './string_similarity';

// Pitch estimation (autocorrelation)
export { estimatePitch, type PitchResult } from './pitch_utils';

// Mel-band energy feature extraction (for (2+N)D speaker clustering)
export { computeMelBandEnergies, MEL_FILTER_BANK } from './mel_utils';

// Speaker clustering
export {
  assignSpeaker,
  median,
  weightedMedian,
  type SpeakerProfile,
  type AssignResult,
} from './speaker_cluster';

// TensorFlow Lite model output processing
export {
  // Decoders
  decodePredictions,
  decodeFromRawArrays,
  // Coordinate utils
  toPixelCoordinates,
  getBoxCenter,
  getBoxArea,
  getSpatialDirection,
  mapBoxToScreen,
  // Formatting and filtering
  formatDetection,
  filterByClass,
  filterByMinArea,
  // Constants
  COCO_CLASSES,
  // Types
  type BoundingBox,
  type Detection,
  type TFLiteOutputs,
  type DecoderOptions,
  type SpatialDirection,
  type ScreenBox,
  type FrameInfo,
  // Default
  default as TensorDecoder,
} from './tensor_decoder';
