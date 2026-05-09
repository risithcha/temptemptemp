/**
 * VisionScreen – Camera-based object detection with real-time bounding boxes.
 *
 * Extracted from the original monolithic App.tsx.  Now lives as a dedicated
 * screen inside the React Navigation tab navigator.
 *
 * Key lifecycle behaviour:
 *   • Camera `isActive` is tied to `useIsFocused()` so the camera pauses
 *     when the user switches to another tab (saves battery and hides the iOS
 *     green privacy indicator).
 */
import { StatusBar } from 'expo-status-bar';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Pressable,
  Platform,
  ActivityIndicator,
  Dimensions,
} from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
  runAtTargetFps,
} from 'react-native-vision-camera';
import { useTensorflowModel } from 'react-native-fast-tflite';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { useTextRecognition } from 'react-native-vision-camera-ocr-plus';
import { useState, useCallback, useRef, useMemo, useEffect } from 'react';
import { Worklets } from 'react-native-worklets-core';
import { useIsFocused, useNavigation } from '@react-navigation/native';
import Ionicons from '@expo/vector-icons/Ionicons';

import { useAppState, useVisionAnnouncer, useOcrAutoReader } from '../hooks';
import { triggerHaptic } from '../utils/haptics';
import { useSettings } from '../contexts/SettingsContext';

import {
  DetectionOverlay,
  AtlasHeader,
  ActionButton,
  OcrTextPanel,
} from '../components';
import {
  decodePredictions,
  filterByMinArea,
  type Detection,
  type TFLiteOutputs,
  type FrameInfo,
} from '../utils/tensor_decoder';
import { sanitizeOcrText } from '../utils/ocr_utils';
import {
  COLORS,
  TYPOGRAPHY,
  SPACING,
  RADII,
  SIZES,
  MODEL_INPUT_SIZE,
  CONFIDENCE_THRESHOLD,
  MAX_DETECTIONS,
  INFERENCE_FPS,
  MIN_BOX_AREA,
  OCR_FPS,
} from '../theme';

const { width: SCREEN_W, height: SCREEN_H } = Dimensions.get('window');

// ---------------------------------------------------------------------------
// VisionScreen
// ---------------------------------------------------------------------------
export default function VisionScreen() {
  const isFocused = useIsFocused();
  const navigation = useNavigation<any>();
  const appState = useAppState();
  const settings = useSettings();

  // Camera should only run when the screen is focused AND the app is foregrounded.
  const isScreenActive = isFocused && appState === 'active';

  const [facing, setFacing] = useState<'front' | 'back'>('back');
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice(facing);

  // --- Model loading ---
  const tfModel = useTensorflowModel(
    require('../../assets/models/atlas_mobilenet_quant.tflite'),
  );
  const model = tfModel.state === 'loaded' ? tfModel.model : undefined;

  // --- Resize plugin ---
  const { resize } = useResizePlugin();

  // --- Detection state ---
  const [detections, setDetections] = useState<Detection[]>([]);
  const [frameInfo, setFrameInfo] = useState<FrameInfo | null>(null);
  const [fps, setFps] = useState(0);
  const lastInferenceRef = useRef(Date.now());

  // --- OCR state ---
  const [ocrText, setOcrText] = useState('');
  const ocrOptions = useMemo(
    () => ({ language: 'latin' as const, useLightweightMode: true }),
    [],
  );
  const { scanText } = useTextRecognition(ocrOptions);

  // Bridge: worklet → JS thread
  // Wrapped in useRef so we only create the bridge once – calling
  // Worklets.createRunOnJS on every render can cause race conditions.
  const detectionCallbackRef = useRef((
    rawBoxes: number[],
    rawClasses: number[],
    rawScores: number[],
    rawCount: number,
    fWidth: number,
    fHeight: number,
    fOrientation: string,
  ) => {
      const now = Date.now();
      const delta = now - lastInferenceRef.current;
      lastInferenceRef.current = now;
      if (delta > 0) setFps(Math.round(1000 / delta));

      setFrameInfo({
        frameWidth: fWidth,
        frameHeight: fHeight,
        frameOrientation: fOrientation,
      });

      const outputs: TFLiteOutputs = {
        boxes: rawBoxes,
        classes: rawClasses,
        scores: rawScores,
        count: rawCount,
      };

      let results = decodePredictions(outputs, {
        threshold: CONFIDENCE_THRESHOLD,
        maxDetections: MAX_DETECTIONS,
      });
      results = filterByMinArea(results, MIN_BOX_AREA);
      setDetections(results);
    });

  const onDetectionResults = useMemo(
    () => Worklets.createRunOnJS(detectionCallbackRef.current),
    [],
  );

  // Bridge: worklet -> JS thread for OCR results
  const ocrCallbackRef = useRef((text: string) => {
    const cleaned = sanitizeOcrText(text);
    setOcrText(cleaned);
  });

  const onOcrResults = useMemo(
    () => Worklets.createRunOnJS(ocrCallbackRef.current),
    [],
  );

  // --- Frame Processor ---
  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      if (model == null) return;

      // Object detection at INFERENCE_FPS (5 fps)
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
          scale: {
            width: MODEL_INPUT_SIZE,
            height: MODEL_INPUT_SIZE,
          },
          rotation,
          pixelFormat: 'rgb',
          dataType: 'uint8',
        });

        const outputs = model.runSync([resized]);

        const rawBoxes = Array.from(outputs[0] as unknown as number[]);
        const rawClasses = Array.from(outputs[1] as unknown as number[]);
        const rawScores = Array.from(outputs[2] as unknown as number[]);
        const rawCount = outputs[3]
          ? (outputs[3] as unknown as number[])[0]
          : 0;

        onDetectionResults(
          rawBoxes,
          rawClasses,
          rawScores,
          rawCount,
          frame.width,
          frame.height,
          frame.orientation,
        );
      });

      // OCR at OCR_FPS (1 fps) - text doesn't change as fast as objects
      runAtTargetFps(OCR_FPS, () => {
        'worklet';
        const result = scanText(frame);
        if (result?.resultText != null && result.resultText.length > 0) {
          onOcrResults(result.resultText);
        }
      });
    },
    [model, resize, onDetectionResults, scanText, onOcrResults],
  );

  // Toggle camera facing
  const toggleCameraFacing = useCallback(() => {
    triggerHaptic('toggle');
    setFacing((c) => (c === 'back' ? 'front' : 'back'));
  }, []);

  // --- TTS announcements for detected objects ---
  useVisionAnnouncer(detections, isScreenActive, { ttsRate: settings.ttsRate });

  // --- Smart OCR auto-reader (reads new text, skips duplicates) ---
  const { readLatestAloud } = useOcrAutoReader(ocrText, isScreenActive, {
    ttsRate: settings.ttsRate,
  });

  const goHome = useCallback(() => {
    triggerHaptic('selection');
    navigation.navigate('Welcome');
  }, [navigation]);

  const goSettings = useCallback(() => {
    triggerHaptic('selection');
    navigation.navigate('Settings');
  }, [navigation]);

  // --- Render: loading / error / permission / camera ---

  if (tfModel.state === 'loading') {
    return (
      <View style={styles.container}>
        <StatusBar style="light" />
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={COLORS.primary} />
          <Text style={styles.loadingText}>Loading Atlas AI...</Text>
        </View>
      </View>
    );
  }

  if (tfModel.state === 'error') {
    return (
      <View style={styles.container}>
        <StatusBar style="light" />
        <View style={styles.loadingContainer}>
          <Ionicons name="alert-circle" size={64} color={COLORS.danger} />
          <Text style={styles.errorTitle}>Model Error</Text>
          <Text style={styles.loadingText}>
            {tfModel.error?.message ?? 'Unknown error'}
          </Text>
        </View>
      </View>
    );
  }

  if (!hasPermission) {
    return (
      <View style={styles.container}>
        <StatusBar style="light" />
        <AtlasHeader subtitle="Vision Assist" accentColor={COLORS.secondary} onHomePress={goHome} onSettingsPress={goSettings} />
        <View style={styles.permissionContent}>
          <View style={styles.cameraIconContainer}>
            <Ionicons name="camera" size={48} color={COLORS.secondary} />
          </View>
          <Text style={styles.permissionTitle}>Camera Access Required</Text>
          <Text style={styles.permissionMessage}>
            Atlas needs access to your camera to detect and describe objects in
            your environment.
          </Text>
          <ActionButton
            label="Grant Camera Permission"
            icon="camera"
            color={COLORS.secondary}
            variant="filled"
            onPress={requestPermission}
          />
        </View>
      </View>
    );
  }

  if (device == null) {
    return (
      <View style={styles.container}>
        <StatusBar style="light" />
        <View style={styles.loadingContainer}>
          <Ionicons name="camera" size={64} color={COLORS.textMuted} />
          <Text style={styles.loadingText}>No camera device found</Text>
        </View>
      </View>
    );
  }

  // ---- Main camera view ----
  return (
    <Pressable style={styles.container} onPress={readLatestAloud}>
      <StatusBar style="light" />

      {/* Camera – isActive driven by navigation focus */}
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={isScreenActive}
        frameProcessor={frameProcessor}
        pixelFormat="yuv"
        resizeMode="cover"
      />

      {/* Bounding-box overlay */}
      <DetectionOverlay detections={detections} frameInfo={frameInfo} />

      {/* OCR text panel (between overlay and bottom bar) */}
      <OcrTextPanel text={ocrText} fontSize={settings.captionFontSize} />

      {/* Top bar – unified header (transparent over camera) */}
      <AtlasHeader
        subtitle="Vision Assist"
        accentColor={COLORS.secondary}
        transparent
        onHomePress={goHome}
        onSettingsPress={goSettings}
        rightContent={
          <View style={styles.topBarRight}>
            {detections.length > 0 && (
              <View style={styles.countBadge}>
                <Text style={styles.countText}>{detections.length}</Text>
              </View>
            )}
            <TouchableOpacity
              style={styles.flipButton}
              onPress={toggleCameraFacing}
              activeOpacity={0.7}
            >
              <Ionicons
                name="camera-reverse-outline"
                size={28}
                color={COLORS.text}
              />
            </TouchableOpacity>
          </View>
        }
      />

      {/* Bottom overlay */}
      <View style={styles.bottomOverlay}>
        <View style={styles.statusContainer}>
          <View
            style={[
              styles.statusDot,
              {
                backgroundColor:
                  model != null ? COLORS.primary : COLORS.textMuted,
              },
            ]}
          />
          <Text style={styles.statusText}>
            {model != null
              ? `Detecting \u2022 ${fps} inf/s`
              : 'Loading model...'}
          </Text>
        </View>

        <Text style={styles.fpsText}>
          Model: {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE} UINT8 •{' '}
          {detections.length} object{detections.length !== 1 ? 's' : ''}
        </Text>

        {detections.length > 0 && (
          <View style={styles.detectionList}>
            {detections.slice(0, 3).map((d, i) => (
              <Text key={i} style={styles.detectionItem}>
                {d.label} ({Math.round(d.score * 100)}%)
              </Text>
            ))}
          </View>
        )}
      </View>
    </Pressable>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },

  // Loading / Error
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: SPACING.xxl,
  },
  loadingText: {
    color: COLORS.textMuted,
    fontSize: TYPOGRAPHY.body.fontSize,
    marginTop: SPACING.md,
    textAlign: 'center',
  },
  errorTitle: {
    color: COLORS.text,
    ...TYPOGRAPHY.title,
    marginTop: SPACING.md,
  },

  // Permission
  permissionContent: {
    alignItems: 'center',
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: SPACING.xl,
  },
  cameraIconContainer: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: COLORS.secondaryFaded,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: SPACING.lg,
  },
  permissionTitle: {
    ...TYPOGRAPHY.title,
    color: COLORS.text,
    textAlign: 'center',
    marginBottom: SPACING.md,
  },
  permissionMessage: {
    fontSize: TYPOGRAPHY.body.fontSize,
    color: COLORS.textMuted,
    textAlign: 'center',
    lineHeight: 24,
    marginBottom: SPACING.xxl,
    paddingHorizontal: SPACING.lg,
  },

  // Top bar overlay
  topBarRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.sm,
  },
  flipButton: {
    width: SIZES.iconButton,
    height: SIZES.iconButton,
    borderRadius: SIZES.iconButtonRadius,
    backgroundColor: COLORS.whiteFaded,
    justifyContent: 'center',
    alignItems: 'center',
  },
  countBadge: {
    backgroundColor: COLORS.secondary,
    borderRadius: RADII.lg,
    paddingHorizontal: SPACING.sm + 2,
    paddingVertical: 2,
    minWidth: SPACING.lg,
    alignItems: 'center',
  },
  countText: {
    color: COLORS.text,
    fontSize: TYPOGRAPHY.caption.fontSize,
    fontWeight: 'bold',
  },

  // Bottom overlay
  bottomOverlay: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: COLORS.overlay,
    paddingBottom: Platform.OS === 'ios' ? SPACING.xxl : SPACING.lg + 6,
    paddingTop: SPACING.md,
    alignItems: 'center',
  },
  statusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 6,
  },
  statusDot: {
    width: SIZES.statusDotSmall,
    height: SIZES.statusDotSmall,
    borderRadius: SIZES.statusDotSmall / 2,
    marginRight: SPACING.sm,
  },
  statusText: {
    color: COLORS.text,
    fontSize: TYPOGRAPHY.body.fontSize,
    textAlign: 'center',
  },
  fpsText: {
    color: COLORS.textMuted,
    fontSize: TYPOGRAPHY.small.fontSize,
    textAlign: 'center',
    marginTop: SPACING.xs,
  },
  detectionList: {
    marginTop: SPACING.sm,
    alignItems: 'center',
  },
  detectionItem: {
    color: COLORS.text,
    fontSize: TYPOGRAPHY.caption.fontSize,
    opacity: 0.85,
  },
});
