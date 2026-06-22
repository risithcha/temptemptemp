// YOLOv8 Object Detection Output Decoder
// Converts (pre-filtered) YOLOv8 candidate outputs into structured detections.
//
// YOLOv8 emits a SINGLE output tensor of shape [1, 84, 8400] (channels-first):
//   - 84 channels = 4 box (cx, cy, w, h) + 80 class scores
//   - 8400 anchors
//   - boxes are center-format (xywh); coordinates are either in input pixels
//     (0..inputSize) or already normalized (0..1) depending on the export.
//
// The 8400-anchor sweep + confidence thresholding happens in the VisionCamera
// worklet (see VisionScreen) so only the surviving candidates cross to JS.
// This module performs the remaining work on that small candidate set:
//   center->corner conversion, coordinate normalization, and per-class NMS.

// COCO-80 class labels (contiguous, 0-indexed) – the label order produced by
// Ultralytics YOLOv8 COCO exports.  No background class, no placeholders.
export const COCO_CLASSES_80: readonly string[] = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
  'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
  'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
  'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
  'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
  'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
  'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
  'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
  'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
] as const;

/** Gemini-only extras for interview / presentation rooms (not in on-device YOLO). */
export const VISION_EXTRA_CLASSES: readonly string[] = [
  'coffee cup',
  'water bottle',
  'microphone',
  'whiteboard',
  'projector',
  'notepad',
  'pen',
  'eyeglasses',
  'papers',
  /** Generic electronics (Gemini-only — not brand-specific). */
  'phone',
  'tablet',
  'laptop',
  'computer',
  'table',
] as const;

/** COCO-80 plus interview-room extras allowed for cloud vision announcements. */
export const VISION_ALLOWED_CLASSES: readonly string[] = [
  ...COCO_CLASSES_80,
  ...VISION_EXTRA_CLASSES,
] as const;

const VISION_LABEL_SET = new Set<string>(VISION_ALLOWED_CLASSES);

/** Common Gemini / speech variants mapped to canonical COCO-80 names. */
const COCO_LABEL_ALIASES: Record<string, string> = {
  human: 'person',
  man: 'person',
  woman: 'person',
  people: 'person',
  child: 'person',
  boy: 'person',
  girl: 'person',
  phone: 'phone',
  smartphone: 'phone',
  'mobile phone': 'phone',
  cellphone: 'phone',
  'cell phone': 'phone',
  telephone: 'phone',
  tablet: 'tablet',
  laptop: 'laptop',
  computer: 'computer',
  pc: 'computer',
  desktop: 'computer',
  'desktop computer': 'computer',
  // Normalize common brand/model names → generic device labels
  iphone: 'phone',
  android: 'phone',
  ipad: 'tablet',
  macbook: 'laptop',
  chromebook: 'laptop',
  imac: 'computer',
  television: 'tv',
  telly: 'tv',
  sofa: 'couch',
  settee: 'couch',
  plant: 'potted plant',
  houseplant: 'potted plant',
  flowerpot: 'potted plant',
  bike: 'bicycle',
  cycle: 'bicycle',
  motorbike: 'motorcycle',
  plane: 'airplane',
  aeroplane: 'airplane',
  jet: 'airplane',
  automobile: 'car',
  auto: 'car',
  'traffic signal': 'traffic light',
  stoplight: 'traffic light',
  purse: 'handbag',
  rucksack: 'backpack',
  'dining table': 'table',
  desk: 'table',
  hotdog: 'hot dog',
  fridge: 'refrigerator',
  teddy: 'teddy bear',
  'hair dryer': 'hair drier',
  blowdryer: 'hair drier',
  'blow dryer': 'hair drier',
  'remote control': 'remote',
  // Interview / presentation room extras
  coffee: 'coffee cup',
  'coffee mug': 'coffee cup',
  mug: 'coffee cup',
  mic: 'microphone',
  mike: 'microphone',
  glasses: 'eyeglasses',
  spectacles: 'eyeglasses',
  notebook: 'notepad',
  'sticky notes': 'notepad',
  'white board': 'whiteboard',
  flipchart: 'whiteboard',
  'flip chart': 'whiteboard',
  'presentation screen': 'projector',
  marker: 'pen',
  pencil: 'pen',
  paper: 'papers',
  document: 'papers',
  documents: 'papers',
};

/** Comma-separated allowed vision labels for Gemini prompts (COCO-80 + extras). */
export const VISION_CLASSES_PROMPT_LIST = VISION_ALLOWED_CLASSES.join(', ');

/** @deprecated Use VISION_CLASSES_PROMPT_LIST – kept for callers that only need COCO. */
export const COCO_CLASSES_PROMPT_LIST = VISION_CLASSES_PROMPT_LIST;

/**
 * Map a free-text label to an allowed vision class (COCO-80 or interview extras), or null.
 */
export function resolveToVisionLabel(raw: string): string | null {
  let normalized = raw.trim().toLowerCase().replace(/_/g, ' ').replace(/\s+/g, ' ');
  if (!normalized) return null;

  if (VISION_LABEL_SET.has(normalized)) return normalized;

  const alias = COCO_LABEL_ALIASES[normalized];
  if (alias) return alias;

  if (normalized.endsWith('s') && normalized.length > 3) {
    const singular = normalized.endsWith('es')
      ? normalized.slice(0, -2)
      : normalized.slice(0, -1);
    if (VISION_LABEL_SET.has(singular)) return singular;
    const singularAlias = COCO_LABEL_ALIASES[singular];
    if (singularAlias) return singularAlias;
  }

  return null;
}

/** @deprecated Use resolveToVisionLabel */
export function resolveToCocoLabel(raw: string): string | null {
  return resolveToVisionLabel(raw);
}

export function visionClassIdForLabel(label: string): number {
  const cocoIdx = COCO_CLASSES_80.indexOf(label);
  if (cocoIdx >= 0) return cocoIdx;
  const extraIdx = VISION_EXTRA_CLASSES.indexOf(label);
  if (extraIdx >= 0) return COCO_CLASSES_80.length + extraIdx;
  return -1;
}

/** Resolve a 0-indexed COCO class id to its label, with a safe fallback. */
export function labelForClass(
  classId: number,
  labels: readonly string[] = COCO_CLASSES_80,
): string {
  return labels[classId] ?? `class_${classId}`;
}

// Bounding box with normalized coordinates (0-1)
export interface BoundingBox {
  top: number;     // y1
  left: number;    // x1
  bottom: number;  // y2
  right: number;   // x2
}

// Single detection result from the model
export interface Detection {
  label: string;      // e.g., "person", "car"
  classId: number;    // 0-79 for COCO-80
  score: number;      // 0.0-1.0
  box: BoundingBox;   // normalized 0-1
}

/**
 * Pre-filtered YOLOv8 candidates emitted by the worklet.  All three arrays are
 * parallel: candidate `i` is `boxes[4i..4i+3]` (cx, cy, w, h), `scores[i]`,
 * `classIds[i]`.  Coordinates may be in input-pixel or normalized space –
 * `decodeYolo` normalizes defensively.
 */
export interface YoloCandidates {
  /** Flattened center-format boxes: [cx0, cy0, w0, h0, cx1, ...] (len = 4N). */
  boxes: number[] | Float32Array;
  /** Per-candidate confidence = max class score (len = N). */
  scores: number[] | Float32Array;
  /** Per-candidate class id 0-79 (len = N). */
  classIds: number[] | Float32Array;
}

export interface YoloDecodeOptions {
  /** Model input size in px – used to normalize pixel-space coords. @default 640 */
  inputSize?: number;
  /** IoU threshold above which lower-score same-class boxes are suppressed. @default 0.45 */
  iouThreshold?: number;
  /** Max detections returned after NMS. @default 10 */
  maxDetections?: number;
  /** Custom class labels (default: COCO-80). */
  classLabels?: readonly string[];
}

interface ScoredBox {
  box: BoundingBox;
  score: number;
  classId: number;
}

/** Intersection-over-union of two normalized boxes. */
function iou(a: BoundingBox, b: BoundingBox): number {
  const ix1 = Math.max(a.left, b.left);
  const iy1 = Math.max(a.top, b.top);
  const ix2 = Math.min(a.right, b.right);
  const iy2 = Math.min(a.bottom, b.bottom);

  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  if (inter <= 0) return 0;

  const areaA = Math.max(0, a.right - a.left) * Math.max(0, a.bottom - a.top);
  const areaB = Math.max(0, b.right - b.left) * Math.max(0, b.bottom - b.top);
  const union = areaA + areaB - inter;
  return union > 0 ? inter / union : 0;
}

/**
 * Decode pre-filtered YOLOv8 candidates into structured detections.
 *
 * Pipeline: normalize coords -> center->corner -> per-class greedy NMS ->
 * sort by score -> cap to `maxDetections`.
 */
export function decodeYolo(
  candidates: YoloCandidates,
  options: YoloDecodeOptions = {},
): Detection[] {
  const {
    inputSize = 640,
    iouThreshold = 0.45,
    maxDetections = 10,
    classLabels = COCO_CLASSES_80,
  } = options;

  const { boxes, scores, classIds } = candidates;
  const n = scores.length;
  if (n === 0) return [];

  // Coordinate-space auto-detection (R1): YOLOv8 exports differ – some emit
  // pixel coords (0..inputSize), some normalized (0..1).  Scan once: if any
  // coordinate clearly exceeds the normalized range, treat the whole batch as
  // pixel-space and divide by inputSize.
  let maxCoord = 0;
  for (let i = 0; i < boxes.length; i++) {
    const v = boxes[i];
    if (v > maxCoord) maxCoord = v;
  }
  const norm = maxCoord > 1.5 && inputSize > 0 ? 1 / inputSize : 1;

  // Center-format -> corner-format normalized boxes.
  const scored: ScoredBox[] = [];
  for (let i = 0; i < n; i++) {
    const cx = boxes[i * 4] * norm;
    const cy = boxes[i * 4 + 1] * norm;
    const w = boxes[i * 4 + 2] * norm;
    const h = boxes[i * 4 + 3] * norm;

    const left = clamp01(cx - w / 2);
    const top = clamp01(cy - h / 2);
    const right = clamp01(cx + w / 2);
    const bottom = clamp01(cy + h / 2);
    if (right <= left || bottom <= top) continue;

    scored.push({
      box: { top, left, bottom, right },
      score: scores[i],
      classId: Math.round(classIds[i]),
    });
  }

  // Sort by confidence (highest first) for greedy NMS.
  scored.sort((a, b) => b.score - a.score);

  // Per-class greedy non-max suppression.
  const kept: Detection[] = [];
  const suppressed = new Array<boolean>(scored.length).fill(false);

  for (let i = 0; i < scored.length && kept.length < maxDetections; i++) {
    if (suppressed[i]) continue;
    const cur = scored[i];

    kept.push({
      label: labelForClass(cur.classId, classLabels),
      classId: cur.classId,
      score: cur.score,
      box: cur.box,
    });

    for (let j = i + 1; j < scored.length; j++) {
      if (suppressed[j]) continue;
      if (scored[j].classId !== cur.classId) continue;
      if (iou(cur.box, scored[j].box) > iouThreshold) {
        suppressed[j] = true;
      }
    }
  }

  return kept;
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

// Convert normalized box coordinates (0-1) to pixel coordinates
export function toPixelCoordinates(
  box: BoundingBox,
  imageWidth: number,
  imageHeight: number
): BoundingBox {
  return {
    top: Math.round(box.top * imageHeight),
    left: Math.round(box.left * imageWidth),
    bottom: Math.round(box.bottom * imageHeight),
    right: Math.round(box.right * imageWidth),
  };
}

// Spatial direction relative to the camera frame (thirds of screen width)
export type SpatialDirection = 'Left' | 'Center' | 'Right';

export function getSpatialDirection(box: BoundingBox): SpatialDirection {
  const centerX = (box.left + box.right) / 2;
  if (centerX < 0.33) return 'Left';
  if (centerX > 0.66) return 'Right';
  return 'Center';
}

// Get center point of a bounding box
export function getBoxCenter(box: BoundingBox): { x: number; y: number } {
  return {
    x: (box.left + box.right) / 2,
    y: (box.top + box.bottom) / 2,
  };
}

// Calculate bounding box area (useful for size filtering)
export function getBoxArea(box: BoundingBox): number {
  const width = Math.abs(box.right - box.left);
  const height = Math.abs(box.bottom - box.top);
  return width * height;
}

// Format detection as readable string: "Person detected at [x, y] (87% confidence)"
export function formatDetection(
  detection: Detection,
  imageWidth?: number,
  imageHeight?: number
): string {
  const { label, score, box } = detection;
  const confidencePercent = Math.round(score * 100);

  if (imageWidth && imageHeight) {
    const pixelBox = toPixelCoordinates(box, imageWidth, imageHeight);
    const center = getBoxCenter(pixelBox);
    return `${label} detected at [${Math.round(center.x)}, ${Math.round(center.y)}] (${confidencePercent}% confidence)`;
  } else {
    const center = getBoxCenter(box);
    return `${label} detected at [${center.x.toFixed(2)}, ${center.y.toFixed(2)}] (${confidencePercent}% confidence)`;
  }
}

// Filter detections by class labels (e.g., ["person", "car"])
export function filterByClass(
  detections: Detection[],
  allowedLabels: string[]
): Detection[] {
  const labelSet = new Set(allowedLabels.map(l => l.toLowerCase()));
  return detections.filter(d => labelSet.has(d.label.toLowerCase()));
}

// Filter detections by minimum bounding box area (normalized 0-1)
// Use to remove noise from very small detections
export function filterByMinArea(
  detections: Detection[],
  minArea: number
): Detection[] {
  return detections.filter(d => getBoxArea(d.box) >= minArea);
}

// ---------------------------------------------------------------------------
// Screen-space bounding box (pixel coordinates, ready for overlay rendering)
// ---------------------------------------------------------------------------
export interface ScreenBox {
  x: number;       // left edge in pixels
  y: number;       // top edge in pixels
  width: number;   // width in pixels
  height: number;  // height in pixels
}

// Frame information needed for coordinate mapping
export interface FrameInfo {
  frameWidth: number;        // raw camera buffer width (before rotation)
  frameHeight: number;       // raw camera buffer height (before rotation)
  frameOrientation: string;  // e.g. 'landscape-left', 'landscape-right', 'portrait'
}

/**
 * Map a model's normalized bounding box [0-1] to screen-space pixel coordinates.
 *
 * Prerequisite: the resize plugin is called with the correct `rotation` parameter
 * so the model receives upright (portrait-oriented) content.
 *
 * Pipeline (resize plugin internals):
 *   center-crop (on raw buffer) → scale to model input → rotate
 *
 * Because the plugin rotates the pixel data and the Camera preview applies the
 * same rotation (but to the full frame), the model's normalised coords map
 * directly into portrait display space with only a vertical offset for the crop.
 *
 * Derivation (verified for landscape-left, landscape-right, portrait):
 *   cropSize = min(frameWidth, frameHeight)  ≡  displayW
 *   cropOffsetY = (displayH − cropSize) / 2
 *   display_x = model_x × cropSize
 *   display_y = cropOffsetY + model_y × cropSize
 * Then apply Camera "cover" mode scaling.
 */
export function mapBoxToScreen(
  box: BoundingBox,
  frameInfo: FrameInfo,
  screenWidth: number,
  screenHeight: number,
  _modelInputSize: number,
): ScreenBox {
  const { frameWidth, frameHeight, frameOrientation } = frameInfo;

  // Step 1: Portrait display dimensions
  const isLandscape =
    frameOrientation === 'landscape-left' ||
    frameOrientation === 'landscape-right';

  const displayW = isLandscape ? frameHeight : frameWidth;
  const displayH = isLandscape ? frameWidth : frameHeight;

  // Step 2: Un-crop.
  // The resize plugin center-crops a square (size = min(frameW, frameH)) from
  // the raw buffer, scales it, then rotates.  After rotation the cropped
  // square covers the full display width and a centred vertical band.
  const cropSize = Math.min(frameWidth, frameHeight); // === displayW
  const cropOffsetY = (displayH - cropSize) / 2;
  // Model normalised coords → display pixel coords (portrait space)
  const dispX1 = box.left * cropSize;
  const dispY1 = cropOffsetY + box.top * cropSize;
  const dispX2 = box.right * cropSize;
  const dispY2 = cropOffsetY + box.bottom * cropSize;

  // Step 3: Camera preview "cover" mode transform.
  const scale = Math.max(screenWidth / displayW, screenHeight / displayH);
  const offsetX = (screenWidth - displayW * scale) / 2;
  const offsetY = (screenHeight - displayH * scale) / 2;

  const screenX1 = dispX1 * scale + offsetX;
  const screenY1 = dispY1 * scale + offsetY;
  const screenX2 = dispX2 * scale + offsetX;
  const screenY2 = dispY2 * scale + offsetY;

  // Clamp to screen bounds
  const clampedX = Math.max(0, screenX1);
  const clampedY = Math.max(0, screenY1);
  const clampedX2 = Math.min(screenWidth, screenX2);
  const clampedY2 = Math.min(screenHeight, screenY2);

  return {
    x: clampedX,
    y: clampedY,
    width: Math.max(0, clampedX2 - clampedX),
    height: Math.max(0, clampedY2 - clampedY),
  };
}

// Default export for convenience
export default {
  decodeYolo,
  labelForClass,
  toPixelCoordinates,
  getBoxCenter,
  getBoxArea,
  getSpatialDirection,
  formatDetection,
  filterByClass,
  filterByMinArea,
  mapBoxToScreen,
  COCO_CLASSES_80,
};
