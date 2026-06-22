/**
 * On-device YOLOv8 object detection via VisionCamera frame processor + TFLite.
 */
import { useRef, useMemo, useState } from 'react';
import {
  useFrameProcessor,
  runAtTargetFps,
  type Frame,
} from 'react-native-vision-camera';
import { useTensorflowModel } from 'react-native-fast-tflite';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { Worklets } from 'react-native-worklets-core';

import {
  decodeYolo,
  filterByMinArea,
  type Detection,
} from '../utils/tensor_decoder';
import { useDetectionAnnouncement } from '../utils/vision_announce';
import {
  CONFIDENCE_THRESHOLD,
  INFERENCE_FPS,
  MAX_DETECTIONS,
  MIN_BOX_AREA,
  MODEL_INPUT_SIZE,
  NMS_IOU_THRESHOLD,
  YOLO_NUM_CLASSES,
} from '../theme';

const YOLO_NUM_ANCHORS = 8400;

const MODEL = require('../../assets/models/yolov8s_float32.tflite');

export interface UseLocalYoloVisionOptions {
  isActive: boolean;
  ttsRate?: number;
}

export interface UseLocalYoloVisionResult {
  detections: Detection[];
  fps: number;
  frameProcessor: ReturnType<typeof useFrameProcessor>;
  modelState: 'loading' | 'loaded' | 'error';
}

function prefilterYoloOutput(
  output: Float32Array,
  threshold: number,
): { boxes: number[]; scores: number[]; classIds: number[] } {
  'worklet';
  const boxes: number[] = [];
  const scores: number[] = [];
  const classIds: number[] = [];

  for (let i = 0; i < YOLO_NUM_ANCHORS; i += 1) {
    let bestScore = 0;
    let bestClass = 0;
    for (let c = 0; c < YOLO_NUM_CLASSES; c += 1) {
      const score = output[(4 + c) * YOLO_NUM_ANCHORS + i];
      if (score > bestScore) {
        bestScore = score;
        bestClass = c;
      }
    }
    if (bestScore < threshold) continue;

    boxes.push(
      output[i],
      output[YOLO_NUM_ANCHORS + i],
      output[2 * YOLO_NUM_ANCHORS + i],
      output[3 * YOLO_NUM_ANCHORS + i],
    );
    scores.push(bestScore);
    classIds.push(bestClass);
  }

  return { boxes, scores, classIds };
}

export function useLocalYoloVision({
  isActive,
  ttsRate = 0.95,
}: UseLocalYoloVisionOptions): UseLocalYoloVisionResult {
  const tfModel = useTensorflowModel(MODEL);
  const model = tfModel.state === 'loaded' ? tfModel.model : undefined;
  const { resize } = useResizePlugin();

  const [detections, setDetections] = useState<Detection[]>([]);
  const [fps, setFps] = useState(0);
  const lastInferenceRef = useRef(Date.now());

  const detectionCallbackRef = useRef(
    (boxes: number[], scores: number[], classIds: number[]) => {
      const now = Date.now();
      const delta = now - lastInferenceRef.current;
      lastInferenceRef.current = now;
      if (delta > 0) setFps(Math.round(1000 / delta));

      let results = decodeYolo(
        { boxes, scores, classIds },
        {
          inputSize: MODEL_INPUT_SIZE,
          iouThreshold: NMS_IOU_THRESHOLD,
          maxDetections: MAX_DETECTIONS,
        },
      );
      results = filterByMinArea(results, MIN_BOX_AREA);
      setDetections(results);
    },
  );

  const onDetectionResults = useMemo(
    () => Worklets.createRunOnJS(detectionCallbackRef.current),
    [],
  );

  const frameProcessor = useFrameProcessor(
    (frame: Frame) => {
      'worklet';
      if (model == null) return;

      runAtTargetFps(INFERENCE_FPS, () => {
        'worklet';

        const orientation = frame.orientation;
        const rotation =
          orientation === 'landscape-left'
            ? '90deg'
            : orientation === 'landscape-right'
              ? '270deg'
              : orientation === 'portrait-upside-down'
                ? '180deg'
                : '0deg';

        const resized = resize(frame, {
          scale: { width: MODEL_INPUT_SIZE, height: MODEL_INPUT_SIZE },
          rotation,
          pixelFormat: 'rgb',
          dataType: 'float32',
        });

        const outputs = model.runSync([resized]);
        const raw = outputs[0] as Float32Array | number[];
        const output =
          raw instanceof Float32Array ? raw : Float32Array.from(raw as number[]);

        const { boxes, scores, classIds } = prefilterYoloOutput(
          output,
          CONFIDENCE_THRESHOLD,
        );
        onDetectionResults(boxes, scores, classIds);
      });
    },
    [model, resize, onDetectionResults],
  );

  useDetectionAnnouncement(isActive ? detections : [], isActive, { ttsRate });

  return {
    detections: isActive ? detections : [],
    fps: isActive ? fps : 0,
    frameProcessor,
    modelState: tfModel.state,
  };
}
