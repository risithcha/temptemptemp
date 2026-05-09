/**
 * AlertOverlay - Full-screen crisis mode overlay for emergency sound detection.
 *
 * Ported from the desktop frontend's crisis mode (main.py).
 * Flashes between red and orange at 1 Hz (well below the 3 Hz epilepsy
 * threshold) with a large warning icon and dismissal button.
 */
import { useEffect, useCallback } from 'react';
import { StyleSheet, Text, View, TouchableOpacity } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withTiming,
  cancelAnimation,
  Easing,
} from 'react-native-reanimated';
import Ionicons from '@expo/vector-icons/Ionicons';

import { triggerHaptic } from '../utils/haptics';
import { playHapticPattern, type HapticPattern } from '../utils/haptic_patterns';
import type { AlarmAlert } from '../hooks/useAlarmDetector';
import {
  COLORS,
  TYPOGRAPHY,
  SPACING,
  RADII,
  ALERT_FLASH_INTERVAL_MS,
} from '../theme';

export interface AlertOverlayProps {
  alert: AlarmAlert | null;
  onDismiss: () => void;
  /** When true, fires distinct vibration patterns per alarm type. */
  hapticPatternsEnabled?: boolean;
}

const ALARM_TYPE_TO_PATTERN: Record<string, HapticPattern> = {
  FIRE_ALARM: 'fireAlarm',
  SMOKE_ALARM: 'fireAlarm',
  SIREN: 'siren',
};

export function AlertOverlay({ alert, onDismiss, hapticPatternsEnabled = true }: AlertOverlayProps) {
  const flashProgress = useSharedValue(0);

  useEffect(() => {
    if (alert) {
      if (hapticPatternsEnabled) {
        const pattern = ALARM_TYPE_TO_PATTERN[alert.type];
        if (pattern) {
          playHapticPattern(pattern);
        } else {
          triggerHaptic('error');
        }
      } else {
        triggerHaptic('error');
      }
      flashProgress.value = withRepeat(
        withTiming(1, {
          duration: ALERT_FLASH_INTERVAL_MS,
          easing: Easing.inOut(Easing.ease),
        }),
        -1,
        true,
      );
    } else {
      cancelAnimation(flashProgress);
      flashProgress.value = 0;
    }

    return () => {
      cancelAnimation(flashProgress);
    };
  }, [alert, flashProgress]);

  const animatedBg = useAnimatedStyle(() => {
    const r = 244 + (255 - 244) * flashProgress.value;
    const g = 67 + (152 - 67) * flashProgress.value;
    const b = 54 + (0 - 54) * flashProgress.value;
    return {
      backgroundColor: `rgba(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)}, 0.92)`,
    };
  });

  const handleDismiss = useCallback(() => {
    triggerHaptic('toggle');
    onDismiss();
  }, [onDismiss]);

  if (!alert) return null;

  return (
    <Animated.View style={[styles.overlay, animatedBg]}>
      <View style={styles.content}>
        <Ionicons name="warning" size={80} color="#fff" />
        <Text style={styles.title}>{alert.message}</Text>
        <Text style={styles.subtitle}>
          Confidence: {Math.round(alert.confidence * 100)}%
        </Text>

        <TouchableOpacity
          style={styles.dismissButton}
          onPress={handleDismiss}
          activeOpacity={0.7}
        >
          <Ionicons name="close-circle" size={24} color={COLORS.danger} />
          <Text style={styles.dismissText}>Dismiss</Text>
        </TouchableOpacity>
      </View>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  overlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 999,
  },
  content: {
    alignItems: 'center',
    paddingHorizontal: SPACING.xl,
  },
  title: {
    color: '#fff',
    ...TYPOGRAPHY.title,
    fontSize: 28,
    textAlign: 'center',
    marginTop: SPACING.lg,
    fontWeight: 'bold',
    textShadowColor: 'rgba(0,0,0,0.5)',
    textShadowOffset: { width: 0, height: 2 },
    textShadowRadius: 4,
  },
  subtitle: {
    color: 'rgba(255,255,255,0.8)',
    fontSize: TYPOGRAPHY.body.fontSize,
    marginTop: SPACING.sm,
    textAlign: 'center',
  },
  dismissButton: {
    marginTop: SPACING.xxl,
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.sm,
    backgroundColor: 'rgba(255,255,255,0.95)',
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.md,
    borderRadius: RADII.round,
  },
  dismissText: {
    color: COLORS.danger,
    fontSize: TYPOGRAPHY.button.fontSize,
    fontWeight: 'bold',
  },
});
