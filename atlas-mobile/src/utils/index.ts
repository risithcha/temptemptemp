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

// Mel-band energy feature extraction (alarm-suppression feature path)
export { computeMelBandEnergies, MEL_FILTER_BANK } from './mel_utils';

// YOLOv8 model output processing
export {
  // Decoder
  decodeYolo,
  labelForClass,
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
  COCO_CLASSES_80,
  VISION_EXTRA_CLASSES,
  VISION_ALLOWED_CLASSES,
  resolveToVisionLabel,
  visionClassIdForLabel,
  // Types
  type BoundingBox,
  type Detection,
  type YoloCandidates,
  type YoloDecodeOptions,
  type SpatialDirection,
  type ScreenBox,
  type FrameInfo,
  // Default
  default as TensorDecoder,
} from './tensor_decoder';
