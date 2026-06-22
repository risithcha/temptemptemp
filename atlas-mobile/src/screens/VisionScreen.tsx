/**
 * VisionScreen – Camera-based object detection with real-time bounding boxes.
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
} from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
} from 'react-native-vision-camera';
import { useState, useCallback, useRef, useEffect } from 'react';
import { useIsFocused, useNavigation } from '@react-navigation/native';
import Ionicons from '@expo/vector-icons/Ionicons';
import Constants from 'expo-constants';

import { useAppState, useVisionAssist, useOcrAutoReader } from '../hooks';
import { triggerHaptic } from '../utils/haptics';
import { useSettings } from '../contexts/SettingsContext';

import { AtlasHeader, ActionButton, OcrTextPanel } from '../components';
import {
  COLORS,
  TYPOGRAPHY,
  SPACING,
  RADII,
  SIZES,
} from '../theme';

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

  const visionApiKey =
    (Constants.expoConfig?.extra as Record<string, string> | undefined)
      ?.geminiKey ?? '';

  // Imperative camera handle – used to grab still snapshots.
  const cameraRef = useRef<Camera | null>(null);

  // Camera is not ready to snapshot until onInitialized fires.
  const [isCameraReady, setIsCameraReady] = useState(false);
  const onCameraInitialized = useCallback(() => setIsCameraReady(true), []);

  // Reset readiness whenever the device changes (e.g. front/back flip).
  const prevDeviceRef = useRef(device);
  useEffect(() => {
    if (prevDeviceRef.current !== device) {
      prevDeviceRef.current = device;
      setIsCameraReady(false);
    }
  }, [device]);

  const visionActive = isScreenActive && isCameraReady;

  const {
    detections,
    fps,
    ocrText,
    frameProcessor,
    needsPhoto,
    pipelineReady,
    modelLoading,
  } = useVisionAssist({
    cameraRef,
    apiKey: visionApiKey,
    isActive: visionActive,
    ttsRate: settings.ttsRate,
  });

  const { handleTap: handleOcrTap, playbackState } =
    useOcrAutoReader(ocrText, visionActive, {
      ttsRate: settings.ttsRate,
    });

  const showDetecting =
    isCameraReady && pipelineReady && visionActive;

  // --- Camera controls ---
  const toggleCameraFacing = useCallback(() => {
    triggerHaptic('toggle');
    setFacing((c) => (c === 'back' ? 'front' : 'back'));
  }, []);

  const goHome = useCallback(() => {
    triggerHaptic('selection');
    navigation.navigate('Welcome');
  }, [navigation]);

  const goSettings = useCallback(() => {
    triggerHaptic('selection');
    navigation.navigate('Settings');
  }, [navigation]);

  // --- Render: permission / device guards ---

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

  if (modelLoading && visionActive) {
    return (
      <View style={styles.container}>
        <StatusBar style="light" />
        <AtlasHeader
          subtitle="Vision Assist"
          accentColor={COLORS.secondary}
          onHomePress={goHome}
          onSettingsPress={goSettings}
        />
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={COLORS.secondary} />
          <Text style={styles.loadingText}>Loading model...</Text>
        </View>
      </View>
    );
  }

  // ---- Main camera view ----
  return (
    <View style={styles.container}>
      <StatusBar style="light" />

      <Camera
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={isScreenActive}
        photo={needsPhoto}
        pixelFormat="yuv"
        frameProcessor={frameProcessor}
        resizeMode="cover"
        onInitialized={onCameraInitialized}
      />

      {/* Tap layer sits above camera, below header controls (overlay is box-none). */}
      <Pressable
        style={styles.tapLayer}
        onPress={handleOcrTap}
        accessibilityRole="button"
        accessibilityLabel="Vision camera. Tap to control text reading. Double tap to restart."
        accessibilityHint="Tap stops or pauses speech. Tap again to resume or start reading detected text."
      />

      <View style={styles.overlayLayer} pointerEvents="box-none">
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

        <OcrTextPanel text={ocrText} playbackState={playbackState} />

        <View style={styles.bottomOverlay} pointerEvents="none">
        <View style={styles.statusContainer}>
          <View
            style={[
              styles.statusDot,
              {
                backgroundColor:
                  showDetecting ? COLORS.primary : COLORS.textMuted,
              },
            ]}
          />
          <Text style={styles.statusText}>
            {isCameraReady
              ? showDetecting
                ? `Detecting \u2022 ${fps} inf/s`
                : 'Loading model...'
              : 'Loading model...'}
          </Text>
        </View>

        <Text style={styles.fpsText}>
          Model: 300x300 UINT8 • {detections.length} object
          {detections.length !== 1 ? 's' : ''}
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
      </View>
    </View>
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
  tapLayer: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 1,
  },
  overlayLayer: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 2,
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

  // Top bar controls
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
