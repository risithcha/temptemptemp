/**
 * Atlas Mobile - Haptic Vocabulary for Deaf-Blind Users
 *
 * Defines distinct vibration rhythms for emergency events and interaction
 * cues.  Uses the built-in RN `Vibration` API on Android for custom
 * timed patterns, and falls back to `expo-haptics` notification types
 * on iOS (which doesn't support arbitrary pattern arrays).
 *
 * Pattern arrays follow `Vibration.vibrate()` format:
 *   [wait, vibrate, wait, vibrate, ...]  (all values in milliseconds)
 */
import { Vibration, Platform } from 'react-native';
import * as Haptics from 'expo-haptics';

export type HapticPattern = 'fireAlarm' | 'siren' | 'newSpeaker';

// Android pattern arrays: alternating [pause, vibrate, pause, vibrate, ...]
const PATTERNS: Record<HapticPattern, number[]> = {
  // SOS rhythm: ... --- ... (3 short, 3 long, 3 short)
  fireAlarm: [
    0,   100, 80, 100, 80, 100,   // 3 short (. . .)
    200, 300, 120, 300, 120, 300,  // 3 long  (- - -)
    200, 100, 80, 100, 80, 100,   // 3 short (. . .)
  ],
  // Alternating long pulses (wailing siren feel)
  siren: [0, 400, 200, 400, 200, 400, 200, 400],
  // Two gentle taps
  newSpeaker: [0, 50, 60, 50],
};

/**
 * Play a named haptic pattern.
 *
 * On Android, fires a timed vibration sequence via `Vibration.vibrate()`.
 * On iOS, fires the closest `expo-haptics` notification type since
 * custom vibration patterns aren't supported without a native module.
 */
export function playHapticPattern(pattern: HapticPattern): void {
  try {
    if (Platform.OS === 'android') {
      Vibration.vibrate(PATTERNS[pattern]);
    } else {
      // iOS fallback: best available approximation via system haptics
      switch (pattern) {
        case 'fireAlarm':
          Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
          break;
        case 'siren':
          Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
          break;
        case 'newSpeaker':
          Haptics.selectionAsync();
          break;
      }
    }
  } catch {
    // Haptic feedback is best-effort - never crash the app.
  }
}
