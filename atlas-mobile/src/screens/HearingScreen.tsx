/**
 * HearingScreen – Live speech-to-text captioning for accessibility.
 *
 * Integrates the `useSpeechRecognition` hook to provide continuous live
 * captions.  Design mirrors the desktop Hearing Assist mode: dark
 * background, large scrolling caption area, green-accented controls.
 *
 * Lifecycle:
 *   • Stops listening when the screen loses focus (tab switch) and does
 *     NOT auto-resume. The user is always in control via the toggle.
 *
 * Audio architecture:
 *   • useAlarmDetector is the SINGLE mic owner via react-native-audio-api.
 *     Its raw PCM tap (onPcm) is wired to Deepgram via useSpeechRecognition's
 *     pushPcm, so there is exactly one recorder session.
 *   • useSpeechRecognition is declared first only to get pushPcm before
 *     passing it to the alarm detector.  The alarm detector MUST be started
 *     first (it grabs the mic before speech recognition sets up anything).
 *   • On start:  startMonitoring() grabs the mic; startListening() opens the
 *     Deepgram WebSocket (or expo fallback).
 *   • On stop:   stopMonitoring() FIRST (releases audio session cleanly),
 *     then stopListening() (closes the WebSocket / expo session).
 *
 * IMPORTANT: The stop condition checks BOTH isListening and isMonitoring to
 * avoid a double-start race where pressing the button quickly could call
 * startMonitoring() twice.
 */
import { StatusBar } from 'expo-status-bar';
import {
  StyleSheet,
  Text,
  View,
  ScrollView,
  Platform,
} from 'react-native';
import { useCallback, useEffect, useRef, useState } from 'react';
import Constants from 'expo-constants';
import { useIsFocused, useNavigation } from '@react-navigation/native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withTiming,
  cancelAnimation,
  Easing,
} from 'react-native-reanimated';
import Ionicons from '@expo/vector-icons/Ionicons';

import { useSpeechRecognition, useAppState, useAlarmDetector, useGeminiAudio } from '../hooks';
import type { AlarmAlert } from '../hooks/useAlarmDetector';
import { triggerHaptic } from '../utils/haptics';
import { useSettings } from '../contexts/SettingsContext';
import { COLORS, RADII, SPACING, TYPOGRAPHY, SIZES } from '../theme';
import { AtlasHeader, ActionButton, AlertOverlay } from '../components';

const TAG = '[HearingScreen]';

/** Gemini must exceed this to show an alert (reduces false positives on room noise). */
const GEMINI_MIN_CONFIDENCE = 0.85;
/** After Gemini rejects, ignore new anomalies for this long. */
const GEMINI_REJECT_COOLDOWN_MS = 12_000;
/** Screen-level retries when Gemini returns unavailable (503, etc.). */
const GEMINI_UNAVAILABLE_RETRIES = 3;
const GEMINI_UNAVAILABLE_RETRY_BASE_MS = 1500;

// ---------------------------------------------------------------------------
// HearingScreen
// ---------------------------------------------------------------------------
export default function HearingScreen() {
  const isFocused = useIsFocused();
  const navigation = useNavigation<any>();
  const appState = useAppState();
  const settings = useSettings();
  const scrollRef = useRef<ScrollView>(null);
  const pushPcmRef = useRef<(samples: Float32Array, sampleRate: number) => void>(() => {});

  console.log(`${TAG} render | isFocused=${isFocused} appState=${appState}`);

  // Alarm detector is the SINGLE mic owner and feeds pitch history for diarization.
  const {
    alert, // Using anomaly instead
    anomaly,
    isMonitoring,
    startMonitoring,
    stopMonitoring,
    dismissAlert,
    releaseAnomalyDebounce,
    pitchHistoryRef,
  } = useAlarmDetector({
    peakThreshold: settings.crisisThreshold,
    onPcm: (samples, sampleRate) => pushPcmRef.current(samples, sampleRate),
  });

  const {
    text,
    segments,
    interimText,
    isListening,
    error,
    isAvailable,
    startListening,
    stopListening,
    resetTranscript,
    pushPcm,
  } = useSpeechRecognition({
    lang: 'en-US',
    continuous: true,
    hapticPatternsEnabled: settings.hapticPatternsEnabled,
    pitchHistoryRef,
  });

  pushPcmRef.current = pushPcm;

  const geminiKey = Constants.expoConfig?.extra?.geminiKey ?? '';
  const { classifyAlarm } = useGeminiAudio(geminiKey);
  const [verifiedAlert, setVerifiedAlert] = useState<AlarmAlert | null>(null);
  const lastAnomalyRef = useRef<number | null>(null);
  const validatingRef = useRef(false);
  const rejectCooldownUntilRef = useRef(0);
  const unavailableRetryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isMonitoringRef = useRef(isMonitoring);
  isMonitoringRef.current = isMonitoring;

  const stopMonitoringRef = useRef(stopMonitoring);
  const stopListeningRef = useRef(stopListening);
  stopMonitoringRef.current = stopMonitoring;
  stopListeningRef.current = stopListening;

  const isListeningRef = useRef(isListening);
  isListeningRef.current = isListening;

  const validateAnomaly = useCallback(
    async (
      pcmSnapshot: Float32Array,
      sampleRate: number,
      bandLabel: string | undefined,
      timestamp: number,
      unavailableAttempt = 0,
    ) => {
      const now = Date.now();
      if (now < rejectCooldownUntilRef.current) {
        console.log(
          `${TAG} Validation skipped – Gemini reject cooldown (${Math.ceil((rejectCooldownUntilRef.current - now) / 1000)}s left)`,
        );
        return;
      }

      if (validatingRef.current) {
        console.log(`${TAG} Validation skipped – already in progress`);
        return;
      }

      validatingRef.current = true;
      const attemptLabel =
        unavailableAttempt > 0
          ? ` (retry ${unavailableAttempt + 1}/${GEMINI_UNAVAILABLE_RETRIES + 1})`
          : '';
      console.log(
        `${TAG} 🤖 Sending 3-second PCM snapshot to Gemini for validation${attemptLabel}...`,
      );

      try {
        const res = await classifyAlarm(pcmSnapshot, sampleRate, bandLabel);

        if (res.status === 'confirmed') {
          if (res.alert.confidence < GEMINI_MIN_CONFIDENCE) {
            console.log(
              `${TAG} ❌ Gemini confidence too low (${res.alert.confidence.toFixed(2)} < ${GEMINI_MIN_CONFIDENCE})`,
            );
            releaseAnomalyDebounce();
            return;
          }
          console.log(`${TAG} ✅ Gemini confirmed alarm:`, res.alert.type);
          lastAnomalyRef.current = timestamp;
          setVerifiedAlert(res.alert);
          return;
        }

        if (res.status === 'rejected') {
          console.log(`${TAG} ❌ Gemini rejected alarm (FALSE_ALARM)`);
          lastAnomalyRef.current = timestamp;
          rejectCooldownUntilRef.current = Date.now() + GEMINI_REJECT_COOLDOWN_MS;
          return;
        }

        console.warn(`${TAG} ⚠️ Gemini unavailable (${res.reason}) – will retry`);
        releaseAnomalyDebounce();

        if (
          unavailableAttempt < GEMINI_UNAVAILABLE_RETRIES &&
          isMonitoringRef.current
        ) {
          const delay = GEMINI_UNAVAILABLE_RETRY_BASE_MS * (unavailableAttempt + 1);
          console.log(`${TAG} Scheduling Gemini retry in ${delay}ms...`);
          unavailableRetryTimerRef.current = setTimeout(() => {
            void validateAnomaly(
              pcmSnapshot,
              sampleRate,
              bandLabel,
              timestamp,
              unavailableAttempt + 1,
            );
          }, delay);
        }
      } catch (e) {
        console.error(`${TAG} Gemini error:`, e);
        releaseAnomalyDebounce();
      } finally {
        validatingRef.current = false;
      }
    },
    [classifyAlarm, releaseAnomalyDebounce],
  );

  useEffect(() => {
    const timestamp = anomaly?.timestamp ?? null;
    if (timestamp == null || timestamp === lastAnomalyRef.current) {
      return;
    }

    if (unavailableRetryTimerRef.current) {
      clearTimeout(unavailableRetryTimerRef.current);
      unavailableRetryTimerRef.current = null;
    }

    const current = anomaly;
    if (!current) return;

    void validateAnomaly(
      current.pcmSnapshot,
      current.sampleRate,
      current.bandLabel,
      current.timestamp,
      0,
    );
  }, [anomaly?.timestamp, anomaly, validateAnomaly]);

  useEffect(() => {
    if (!isMonitoring && unavailableRetryTimerRef.current) {
      clearTimeout(unavailableRetryTimerRef.current);
      unavailableRetryTimerRef.current = null;
    }
  }, [isMonitoring]);

  useEffect(() => {
    return () => {
      if (unavailableRetryTimerRef.current) {
        clearTimeout(unavailableRetryTimerRef.current);
      }
    };
  }, []);

  const handleDismissAlert = useCallback(() => {
    setVerifiedAlert(null);
    lastAnomalyRef.current = null;
    dismissAlert();
  }, [dismissAlert]);

  // --- Stop when navigating away or app backgrounds ---
  useEffect(() => {
    if (isFocused && appState === 'active') return;

    if (isMonitoringRef.current) {
      stopMonitoringRef.current();
    }
    if (isListeningRef.current) {
      void stopListeningRef.current();
    }
  }, [isFocused, appState]);

  // --- Pulsing dot animation (Reanimated) ---
  const pulseScale = useSharedValue(1);
  const pulseOpacity = useSharedValue(1);

  useEffect(() => {
    if (isListening) {
      pulseScale.value = withRepeat(
        withTiming(1.4, { duration: 800, easing: Easing.inOut(Easing.ease) }),
        -1,
        true,
      );
      pulseOpacity.value = withRepeat(
        withTiming(0.4, { duration: 800, easing: Easing.inOut(Easing.ease) }),
        -1,
        true,
      );
    } else {
      cancelAnimation(pulseScale);
      cancelAnimation(pulseOpacity);
      pulseScale.value = withTiming(1, { duration: 200 });
      pulseOpacity.value = withTiming(1, { duration: 200 });
    }
  }, [isListening, pulseScale, pulseOpacity]);

  const pulseStyle = useAnimatedStyle(() => ({
    transform: [{ scale: pulseScale.value }],
    opacity: pulseOpacity.value,
  }));

  // --- Auto-scroll captions to bottom ---
  // onContentSizeChange fires synchronously after layout, removing the
  // need for a setTimeout that can miss rapid updates.

  // --- Debug: log verified alert activations only ---
  useEffect(() => {
    if (verifiedAlert) {
      console.log(
        `${TAG} ⚠️ VERIFIED ALERT ACTIVATED: type=${verifiedAlert.type} message="${verifiedAlert.message}" confidence=${verifiedAlert.confidence.toFixed(3)}`,
      );
    }
  }, [verifiedAlert]);


  // --- Toggle handler ---
  // Check BOTH isListening and isMonitoring to avoid double-start race.
  // Start:  alarm detector FIRST (grabs mic), then speech recognition.
  // Stop:   alarm detector FIRST (releases audio session cleanly).
  const isActive = isListening || isMonitoring;

  const handleToggle = useCallback(async () => {
    console.log(`${TAG} handleToggle() | isActive=${isActive} isListening=${isListening} isMonitoring=${isMonitoring}`);
    triggerHaptic('toggle');
    if (isActive) {
      console.log(`${TAG} handleToggle: STOPPING – stopMonitoring then stopListening`);
      stopMonitoring();
      await stopListening();
      console.log(`${TAG} handleToggle: STOPPED`);
    } else {
      console.log(`${TAG} handleToggle: STARTING – startMonitoring then startListening`);
      await startMonitoring();
      console.log(`${TAG} handleToggle: startMonitoring() returned`);
      await startListening();
      console.log(`${TAG} handleToggle: startListening() returned`);
    }
  }, [isActive, startListening, stopListening, startMonitoring, stopMonitoring]);

  // --- Clear handler with haptic ---
  const handleClear = useCallback(() => {
    triggerHaptic('selection');
    resetTranscript();
  }, [resetTranscript]);

  const goHome = useCallback(() => {
    triggerHaptic('selection');
    if (isMonitoring) stopMonitoring();
    if (isListening) stopListening();
    navigation.navigate('Welcome');
  }, [navigation, isListening, isMonitoring, stopListening, stopMonitoring]);

  const goSettings = useCallback(() => {
    triggerHaptic('selection');
    navigation.navigate('Settings');
  }, [navigation]);

  // --- Render ---
  return (
    <View style={styles.container}>
      <StatusBar style="light" />

      {/* Crisis alert overlay – renders on top of everything when active */}
      <AlertOverlay
        alert={verifiedAlert}
        onDismiss={handleDismissAlert}
        hapticPatternsEnabled={settings.hapticPatternsEnabled}
      />


      {/* Unified header */}
      <AtlasHeader
        subtitle="Hearing Assist"
        accentColor={COLORS.primary}
        onHomePress={goHome}
        onSettingsPress={goSettings}
        rightContent={
          <View style={styles.statusRow}>
            <Animated.View
              style={[
                styles.statusDot,
                isActive ? styles.statusDotActive : styles.statusDotIdle,
                isListening && pulseStyle,
              ]}
            />
            <Text
              style={[
                styles.statusLabel,
                isActive ? styles.statusLabelActive : styles.statusLabelIdle,
              ]}
            >
              {isListening ? 'Listening' : isMonitoring ? 'Monitoring' : 'Ready'}
            </Text>
          </View>
        }
      />

      <View style={styles.content}>

        {/* Instruction text */}
        <Text style={styles.instructions}>
          {isActive
            ? 'Speak clearly – live captions will appear below.'
            : 'Tap "Start Listening" to begin live captioning.'}
        </Text>

        {/* Error banner */}
        {error && (
          <View style={styles.errorBanner}>
            <Ionicons
              name="warning-outline"
              size={18}
              color={COLORS.warning}
            />
            <Text style={styles.errorText}>
              {error === 'not-allowed'
                ? 'Microphone permission denied. Please enable it in Settings.'
                : error === 'service-not-allowed'
                  ? 'Speech recognition is not available on this device.'
                  : `Error: ${error}`}
            </Text>
          </View>
        )}

        {/* Caption area */}
        <View style={styles.captionContainer}>
          <Text style={styles.captionHeader}>Live Captions</Text>
          <ScrollView
            ref={scrollRef}
            style={styles.captionScroll}
            contentContainerStyle={styles.captionContent}
            showsVerticalScrollIndicator
            onContentSizeChange={() => scrollRef.current?.scrollToEnd({ animated: true })}
            accessibilityLabel="Live caption transcript"
          >
            {segments.length > 0 ? (
              <>
                {segments.map((seg, i) => (
                  <View
                    key={i}
                    style={styles.segmentBlock}
                    accessible={true}
                    accessibilityRole="text"
                    accessibilityLabel={`${seg.speaker}: ${seg.text}`}
                  >
                    <Text style={styles.speakerTag}>[{seg.speaker}]</Text>
                    <Text style={[styles.captionText, { fontSize: settings.captionFontSize }]}>{seg.text}</Text>
                  </View>
                ))}
                {interimText ? (
                  <View
                    style={styles.segmentBlock}
                    accessible={true}
                    accessibilityLabel={`In progress: ${interimText}`}
                  >
                    <Text style={[styles.interimText, { fontSize: settings.captionFontSize }]}>{interimText}</Text>
                  </View>
                ) : null}
              </>
            ) : text ? (
              <Text style={[styles.captionText, { fontSize: settings.captionFontSize }]}>{text}</Text>
            ) : (
              <Text style={styles.placeholderText}>
                {isActive
                  ? 'Waiting for speech...'
                  : 'Your transcribed speech will appear here...\n\nTips:\n• Speak clearly and at a normal pace\n• Reduce background noise for best results\n• Each chunk of speech will be transcribed in real time\n• Pauses between speakers are detected automatically'}
              </Text>
            )}
          </ScrollView>
        </View>

        {/* Controls */}
        <View style={styles.controlsRow}>
          <ActionButton
            label="Clear"
            icon="trash-outline"
            color={COLORS.secondary}
            variant="outlined"
            onPress={handleClear}
            disabled={!text}
          />

          <ActionButton
            label={isActive ? 'Stop Listening' : 'Start Listening'}
            icon={isActive ? 'mic-off' : 'mic'}
            iconSize={28}
            color={isActive ? COLORS.danger : COLORS.primary}
            variant="filled"
            onPress={handleToggle}
            disabled={!isAvailable && !isActive}
            fullWidth
          />
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
  content: {
    flex: 1,
    paddingHorizontal: SPACING.lg,
    paddingBottom: Platform.OS === 'ios' ? SPACING.md : SPACING.lg,
  },

  // Status indicator (in header rightContent)
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    flexShrink: 0,
  },
  statusDot: {
    width: SIZES.statusDot,
    height: SIZES.statusDot,
    borderRadius: SIZES.statusDot / 2,
    marginRight: SPACING.sm,
  },
  statusDotActive: {
    backgroundColor: COLORS.primary,
  },
  statusDotIdle: {
    backgroundColor: COLORS.textMuted,
  },
  statusLabel: {
    fontSize: TYPOGRAPHY.caption.fontSize,
    fontWeight: 'bold',
    flexShrink: 0,
  },
  statusLabelActive: {
    color: COLORS.primary,
  },
  statusLabelIdle: {
    color: COLORS.textMuted,
  },

  // Instructions
  instructions: {
    fontSize: TYPOGRAPHY.caption.fontSize,
    color: COLORS.textSecondary,
    textAlign: 'center',
    marginBottom: SPACING.md,
    marginTop: SPACING.sm,
  },

  // Error
  errorBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(244, 67, 54, 0.15)',
    borderRadius: RADII.md,
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    marginBottom: SPACING.md,
  },
  errorText: {
    color: COLORS.danger,
    fontSize: TYPOGRAPHY.caption.fontSize,
    marginLeft: SPACING.sm,
    flex: 1,
  },

  // Caption area
  captionContainer: {
    flex: 1,
    borderWidth: 2,
    borderColor: COLORS.primary,
    borderRadius: RADII.lg,
    overflow: 'hidden',
    marginBottom: SPACING.md,
  },
  captionHeader: {
    fontSize: TYPOGRAPHY.body.fontSize,
    fontWeight: 'bold',
    color: COLORS.primary,
    paddingHorizontal: SPACING.md,
    paddingTop: SPACING.md,
    paddingBottom: SPACING.xs,
  },
  captionScroll: {
    flex: 1,
  },
  captionContent: {
    paddingHorizontal: SPACING.md,
    paddingBottom: SPACING.lg,
  },
  segmentBlock: {
    marginBottom: SPACING.md,
  },
  speakerTag: {
    fontSize: TYPOGRAPHY.caption.fontSize,
    fontWeight: 'bold',
    color: COLORS.primary,
    marginBottom: SPACING.xs,
    letterSpacing: 0.5,
  },
  captionText: {
    fontSize: TYPOGRAPHY.bodyLarge.fontSize,
    color: COLORS.text,
    lineHeight: 34,
  },
  interimText: {
    fontSize: TYPOGRAPHY.bodyLarge.fontSize,
    color: COLORS.textMuted,
    lineHeight: 34,
    fontStyle: 'italic',
  },
  placeholderText: {
    fontSize: TYPOGRAPHY.body.fontSize,
    color: COLORS.textMuted,
    lineHeight: 26,
    fontStyle: 'italic',
  },

  // Controls
  controlsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.md,
    paddingBottom: SPACING.sm,
  },
});
