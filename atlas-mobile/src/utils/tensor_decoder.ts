// TensorFlow Lite Object Detection Output Decoder
// Converts raw tensor outputs from TFLite SSD models into structured detections.
// 
// Model outputs:
// - Output 0 (Boxes):   [1, num_detections, 4] - Bounding boxes [y1, x1, y2, x2] (0-1 normalized)
// - Output 1 (Classes): [1, num_detections] - Class IDs (0-90 for COCO)
// - Output 2 (Scores):  [1, num_detections] - Confidence scores (0.0-1.0)
// - Output 3 (Count):   [1] - Number of valid detections (optional)

// COCO class labels - 91 classes mapped from model output IDs (0-90)
// ID 0 is background/unknown
export const COCO_CLASSES: Record<number, string> = {
  0: '???',
  1: 'person',
  2: 'bicycle',
  3: 'car',
  4: 'motorcycle',
  5: 'airplane',
  6: 'bus',
  7: 'train',
  8: 'truck',
  9: 'boat',
  10: 'traffic light',
  11: 'fire hydrant',
  12: '???',
  13: 'stop sign',
  14: 'parking meter',
  15: 'bench',
  16: 'bird',
  17: 'cat',
  18: 'dog',
  19: 'horse',
  20: 'sheep',
  21: 'cow',
  22: 'elephant',
  23: 'bear',
  24: 'zebra',
  25: 'giraffe',
  26: '???',
  27: 'backpack',
  28: 'umbrella',
  29: '???',
  30: '???',
  31: 'handbag',
  32: 'tie',
  33: 'suitcase',
  34: 'frisbee',
  35: 'skis',
  36: 'snowboard',
  37: 'sports ball',
  38: 'kite',
  39: 'baseball bat',
  40: 'baseball glove',
  41: 'skateboard',
  42: 'surfboard',
  43: 'tennis racket',
  44: 'bottle',
  45: '???',
  46: 'wine glass',
  47: 'cup',
  48: 'fork',
  49: 'knife',
  50: 'spoon',
  51: 'bowl',
  52: 'banana',
  53: 'apple',
  54: 'sandwich',
  55: 'orange',
  56: 'broccoli',
  57: 'carrot',
  58: 'hot dog',
  59: 'pizza',
  60: 'donut',
  61: 'cake',
  62: 'chair',
  63: 'couch',
  64: 'potted plant',
  65: 'bed',
  66: '???',
  67: 'dining table',
  68: '???',
  69: '???',
  70: 'toilet',
  71: '???',
  72: 'tv',
  73: 'laptop',
  74: 'mouse',
  75: 'remote',
  76: 'keyboard',
  77: 'cell phone',
  78: 'microwave',
  79: 'oven',
  80: 'toaster',
  81: 'sink',
  82: 'refrigerator',
  83: '???',
  84: 'book',
  85: 'clock',
  86: 'vase',
  87: 'scissors',
  88: 'teddy bear',
  89: 'hair drier',
  90: 'toothbrush',
};

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
  classId: number;    // 0-90 for COCO
  score: number;      // 0.0-1.0
  box: BoundingBox;   // normalized 0-1
}

// Raw tensor outputs from TFLite model
export interface TFLiteOutputs {
  boxes: Float32Array | number[];    // [1, num_detections, 4] flattened
  classes: Float32Array | number[];  // [1, num_detections]
  scores: Float32Array | number[];   // [1, num_detections]
  count?: number;                    // Optional: number of valid detections
}

// Decoder configuration options
export interface DecoderOptions {
  threshold?: number;                      // Minimum confidence (default: 0.5)
  maxDetections?: number;                  // Max results to return (default: 10)
  classLabels?: Record<number, string>;    // Custom class labels (default: COCO_CLASSES)
  imageWidth?: number;                     // For pixel coordinate conversion
  imageHeight?: number;                    // For pixel coordinate conversion
}

// Decode raw TFLite detection outputs into structured Detection objects.
// Filters by confidence threshold and maps class IDs to labels.
export function decodePredictions(
  outputs: TFLiteOutputs,
  options: DecoderOptions = {}
): Detection[] {
  const {
    threshold = 0.5,
    maxDetections = 10,
    classLabels = COCO_CLASSES,
  } = options;

  const { boxes, classes, scores } = outputs;
  const detections: Detection[] = [];

  // Limit to max detections or count provided by model
  const numDetections = Math.min(
    outputs.count ?? maxDetections,
    maxDetections,
    scores.length
  );

  for (let i = 0; i < numDetections; i++) {
    const score = scores[i];

    // Skip detections below threshold
    if (score < threshold) {
      continue;
    }

    // Get class ID and label
    // SSD MobileNet V1 outputs 0-indexed class IDs
    // but COCO_CLASSES is 1-indexed, so we add 1.
    const rawClassId = Math.round(classes[i]);
    const classId = rawClassId + 1;
    const label = classLabels[classId] ?? `class_${classId}`;

    // Skip unknown/background classes
    if (label === '???' || classId === 0) {
      continue;
    }

    // Extract bounding box: [y1, x1, y2, x2]
    const boxIndex = i * 4;
    const box: BoundingBox = {
      top: boxes[boxIndex],      // y1
      left: boxes[boxIndex + 1], // x1
      bottom: boxes[boxIndex + 2], // y2
      right: boxes[boxIndex + 3],  // x2
    };

    detections.push({
      label,
      classId,
      score,
      box,
    });
  }

  // Sort by confidence score (highest first)
  detections.sort((a, b) => b.score - a.score);

  return detections;
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

// Decode from raw output array [boxes, classes, scores, count?]
// Convenience function for models that return flattened outputs
export function decodeFromRawArrays(
  rawOutputs: (Float32Array | number[])[],
  options: DecoderOptions = {}
): Detection[] {
  // Standard output order: [boxes, classes, scores, count?]
  
  if (rawOutputs.length < 3) {
    console.warn('Expected at least 3 output tensors (boxes, classes, scores)');
    return [];
  }

  const outputs: TFLiteOutputs = {
    boxes: rawOutputs[0],
    classes: rawOutputs[1],
    scores: rawOutputs[2],
    count: rawOutputs[3]?.[0],
  };

  return decodePredictions(outputs, options);
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
 *   center-crop (on raw buffer) → scale to 300×300 → rotate
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
  decodePredictions,
  decodeFromRawArrays,
  toPixelCoordinates,
  getBoxCenter,
  getBoxArea,
  getSpatialDirection,
  formatDetection,
  filterByClass,
  filterByMinArea,
  mapBoxToScreen,
  COCO_CLASSES,
};
