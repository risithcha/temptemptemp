/**
 * Atlas Mobile – Shared Design Tokens
 *
 * Single source of truth for colours, typography, spacing, and model
 * constants.  Mirrors the desktop app's palette.
 */

// ---------------------------------------------------------------------------
// Colour palette
// ---------------------------------------------------------------------------
export const COLORS = {
  /** App-wide dark background */
  background: '#1a1a1a',
  /** Card / elevated surface */
  surface: '#2d2d2d',
  /** Atlas green – primary action, Hearing Assist, success states */
  primary: '#4CAF50',
  primaryDark: '#2E7D32',
  /** Atlas blue – secondary action, Vision Assist */
  secondary: '#2196F3',
  secondaryDark: '#1976D2',
  /** Primary text on dark */
  text: '#ffffff',
  /** Secondary / muted text */
  textMuted: '#888888',
  textSecondary: '#aaaaaa',
  /** Semi-transparent overlays */
  overlay: 'rgba(0, 0, 0, 0.6)',
  /** Danger / error / stop states */
  danger: '#f44336',
  dangerDark: '#D32F2F',
  /** Warning / alerts */
  warning: '#FFC107',
  warningBright: '#FFEB3B',
  /** Faded / translucent variants */
  primaryFaded: 'rgba(76, 175, 80, 0.2)',
  secondaryFaded: 'rgba(33, 150, 243, 0.2)',
  whiteFaded: 'rgba(255, 255, 255, 0.2)',
} as const;

// ---------------------------------------------------------------------------
// Bounding-box colours
// ---------------------------------------------------------------------------
export const BOX_COLORS = [
  '#4CAF50',
  '#2196F3',
  '#FF9800',
  '#E91E63',
  '#9C27B0',
  '#00BCD4',
  '#FFEB3B',
  '#FF5722',
  '#795548',
  '#607D8B',
] as const;

// ---------------------------------------------------------------------------
// Model / inference constants  (YOLOv8s INT8 TFLite)
// ---------------------------------------------------------------------------
/** YOLOv8 square input resolution (640 for the standard COCO export). */
export const MODEL_INPUT_SIZE = 640;
/** Minimum max-class confidence to keep an anchor (YOLO needs lower than SSD). */
export const CONFIDENCE_THRESHOLD = 0.35;
/** Max detections returned after NMS. */
export const MAX_DETECTIONS = 10;
/** Object-detection inference cadence inside the frame processor. */
export const INFERENCE_FPS = 5;
/** Drop detections whose normalized box area is below this (noise filter). */
export const MIN_BOX_AREA = 0.005;
/** IoU threshold for per-class non-max suppression. */
export const NMS_IOU_THRESHOLD = 0.45;
/** Number of COCO classes in the YOLOv8 output tensor. */
export const YOLO_NUM_CLASSES = 80;
/**
 * Output dequantization (R1).  For a float32-output export these stay identity
 * (`scale=1, zeroPoint=0`).  For a full-INT8 export, set them to the model's
 * output tensor `quantization` params so the worklet can map int8 -> float:
 *   value = (raw - YOLO_OUTPUT_ZERO_POINT) * YOLO_OUTPUT_SCALE
 */
export const YOLO_OUTPUT_SCALE = 1;
export const YOLO_OUTPUT_ZERO_POINT = 0;

// ---------------------------------------------------------------------------
// TTS announcement constants
// ---------------------------------------------------------------------------
/** Per-label cooldown before the same object can be re-announced (ms). */
export const TTS_COOLDOWN_MS = 8000;
/** Debounce window after detections change before building the sentence (ms). */
export const TTS_BATCH_DELAY_MS = 1500;

// ---------------------------------------------------------------------------
// OCR constants
// ---------------------------------------------------------------------------
/** Target frame rate for the OCR frame processor pass. */
export const OCR_FPS = 1;
/** Normalized Levenshtein similarity at or above which OCR text is "same". */
export const OCR_SIMILARITY_THRESHOLD = 0.7;
/** Time (ms) after which a cached OCR reading expires and can be re-read. */
export const OCR_CACHE_TTL_MS = 30_000;

// ---------------------------------------------------------------------------
// Deepgram streaming STT constants
// ---------------------------------------------------------------------------
/**
 * Preferred PCM sample rate requested from the recorder's `onAudioReady` tap
 * and used to stream to Deepgram.  Hardware may override this (R9) – the actual
 * delivered `buffer.sampleRate` is always trusted when opening the WebSocket.
 */
export const PCM_TAP_SAMPLE_RATE = 16000;
/** Mono PCM for speech. */
export const PCM_TAP_CHANNELS = 1;
/** Deepgram realtime listen endpoint (sample_rate appended at connect time). */
export const DEEPGRAM_WS_BASE =
  'wss://api.deepgram.com/v1/listen?model=nova-3&diarize=true&punctuate=true&language=en-US&encoding=linear16&channels=1';
/** Interval (ms) between Deepgram KeepAlive messages to hold the socket open. */
export const DEEPGRAM_KEEPALIVE_MS = 8000;

// ---------------------------------------------------------------------------
// Alarm detector constants
// ---------------------------------------------------------------------------
/** FFT size for the AnalyserNode (must be power of 2). */
export const ALARM_FFT_SIZE = 2048;
/** Consecutive detection frames required before confirming an alert.
 *  Logs showed the signal hits 3-4/5 consistently but drops before 5 due to
 *  natural amplitude fluctuation in alarm tones.  3 frames = 300 ms. */
export const ALARM_CONSECUTIVE_FRAMES = 3;
/** Interval (ms) between FFT reads. */
export const ALARM_POLL_INTERVAL_MS = 100;
/** Auto-clear alert after this many ms of no spike. */
export const ALARM_AUTO_CLEAR_MS = 10000;

// ---------------------------------------------------------------------------
// Voice profiling normalization ranges (for (2+N)D speaker clustering)
// ---------------------------------------------------------------------------
/** Minimum expected fundamental pitch (Hz). */
export const PITCH_MIN = 75;
/** Maximum expected fundamental pitch (Hz). */
export const PITCH_MAX = 400;
/** Minimum expected spectral centroid for speech (Hz). */
export const CENTROID_MIN = 300;
/** Maximum expected spectral centroid for speech (Hz). */
export const CENTROID_MAX = 3000;
/**
 * Normalized (2+N)D Euclidean distance threshold for same-speaker matching.
 *
 * Calibration (derived from testing):
 *   Two distinct male voices were observed at:
 *     Speaker A: pitch=116.5 Hz, centroid=881 Hz
 *     Speaker B: pitch= 96.4 Hz, centroid=621 Hz
 *
 *   Normalized 2D distance between them:
 *     dp = (116.5 - 96.4) / 325 = 0.062
 *     dc = (881 - 621) / 2700  = 0.096
 *     d2D = sqrt(0.062^2 + 0.096^2) = 0.114
 *
 *   With SPEAKER_STICKY_FACTOR=0.90 the effective distance is:
 *     0.114 * 0.90 = 0.103
 *
 *   Typical same-speaker between-sentence drift sits at 0.06-0.087
 *   (+-10-15 Hz pitch, +-150-200 Hz centroid), giving an effective
 *   sticky distance of 0.054-0.078; comfortably below 0.10.
 *
 *   Tolerance = 0.10 therefore sits between these two bands:
 *     same-speaker effective (<= 0.078) -> merges [OK]
 *     diff-speaker effective (~= 0.103) -> new profile [OK]
 */
export const VOICE_PROFILE_TOLERANCE = 0.10;
/** Minimum harmonicity confidence to accept a pitch sample as voiced speech. */
export const MIN_VOICE_CONFIDENCE = 0.3;
/** Minimum voiced samples in a segment to trust the 2D weighted median. */
export const MIN_DIARIZATION_SAMPLES = 5;
/**
 * Duration (ms) of continuous silence after which all speaker profiles are
 * purged.  Prevents stale profiles from a previous conversation segment
 * mis-assigning new speakers when the 4-profile cap is exhausted.
 * At 5 minutes this is invisible to normal use but critical for demo sessions
 * with breaks between speakers.
 */
export const PROFILE_STALE_MS = 300_000;
/** Flash interval for crisis overlay (1 Hz, below 3 Hz epilepsy threshold). */
export const ALERT_FLASH_INTERVAL_MS = 500;

// ---------------------------------------------------------------------------
// Mel-band energy feature constants  (for (2+N)D speaker clustering)
// ---------------------------------------------------------------------------
/** Number of log Mel-band energy features appended to each PitchSample. */
export const NUM_MEL_BANDS = 5;
/**
 * Normalization denominator for each log Mel-energy dimension in the
 * (2+N)D speaker distance function.  Derived from the maximum possible
 * log-band power achievable through the triangular filter bank with a
 * Uint8Array (0-255) input: ln(255^2 * ~25 bins ~= 1.6 M) ~= 14.3, rounded
 * conservatively to 9.0 to account for the typical voiced-speech range.
 */
export const MEL_FEATURE_RANGE = 9.0;
/**
 * Scalar weight applied to the summed squared Mel-feature error inside
 * `normalizedDistance`.  Balances the 5-dimensional Mel contribution
 * against the 2D pitch + centroid components so no single axis dominates.
 *
 * Set to 0.15 so that noisy Mel-band variation (room acoustics, background
 * noise) cannot by itself push a same-speaker segment over the tolerance,
 * while still allowing genuine timbral differences to widen the gap between
 * clearly distinct voices.  At 0.15 the maximum possible Mel contribution
 * to distance is sqrt(0.15 * 5) ~= 0.87; meaningful but not dominant.
 */
export const MEL_FEATURE_WEIGHT = 0.15;
/**
 * Multiplicative discount applied to the distance of the most recently
 * active speaker during `assignSpeaker` candidate selection.
 *
 * 0.90 gives a 10% discount; gentle enough that a genuinely different
 * voice (effective distance ~= 0.103 for the 116 Hz / 96 Hz test pair) still
 * exceeds VOICE_PROFILE_TOLERANCE (0.10) and triggers a new profile, while
 * same-speaker drift (effective distance <= 0.078) remains safely below it.
 * The previous value of 0.82 was too aggressive: it collapsed the effective
 * distance of distinct voices to 0.094 and caused them to merge.
 */
export const SPEAKER_STICKY_FACTOR = 0.90;

// ---------------------------------------------------------------------------
// Typography
// ---------------------------------------------------------------------------
export const TYPOGRAPHY = {
  /** Logo / brand text */
  logo: { fontSize: 48, fontWeight: 'bold' as const, letterSpacing: 8 },
  logoSmall: { fontSize: 24, fontWeight: 'bold' as const, letterSpacing: 4 },
  /** Screen titles */
  title: { fontSize: 28, fontWeight: 'bold' as const },
  /** Section headers */
  heading: { fontSize: 22, fontWeight: '600' as const },
  /** Body / caption text */
  body: { fontSize: 16 },
  bodyLarge: { fontSize: 22 },
  /** Button label */
  button: { fontSize: 18, fontWeight: 'bold' as const },
  /** Subtitle / tagline */
  subtitle: { fontSize: 16, fontWeight: '600' as const, letterSpacing: 2 },
  /** Small / meta text */
  caption: { fontSize: 13 },
  small: { fontSize: 12 },
  label: { fontSize: 11, fontWeight: '700' as const },
} as const;

// ---------------------------------------------------------------------------
// Spacing & radii
// ---------------------------------------------------------------------------
export const SPACING = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 40,
} as const;

export const RADII = {
  sm: 4,
  md: 8,
  lg: 12,
  xl: 16,
  round: 999,
} as const;

// ---------------------------------------------------------------------------
// Component-level size tokens
// ---------------------------------------------------------------------------
export const SIZES = {
  /** Tab bar total height (icon + label + padding) */
  tabBarHeight: 90,
  tabBarPaddingBottom: 28,
  /** Unified header height (excl. safe-area insets) */
  headerHeight: 56,
  /** Standard action-button min height */
  buttonMinHeight: 60,
  /** Round icon-button dimensions */
  iconButton: 44,
  iconButtonRadius: 22,
  /** Status dot */
  statusDot: 12,
  statusDotSmall: 10,
} as const;

// ---------------------------------------------------------------------------
// Per-tab accent colours  (Vision = blue, Hearing = green)
// ---------------------------------------------------------------------------
export const TAB_ACCENT = {
  Vision: COLORS.secondary,
  Hearing: COLORS.primary,
} as const;
