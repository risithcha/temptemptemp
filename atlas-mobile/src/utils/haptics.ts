/**
 * Atlas Mobile – Haptic Feedback Utility
 *
 * Provides vibration feedback for accessibility.  Blind and
 * low-vision users rely on haptics to confirm that a button press
 * registered or that a navigation change occurred.
 *
 * Every call is wrapped in a try/catch so haptic failures are
 * silently ignored - they should never crash the app.
 *
 * iOS caveat: the Taptic Engine is suppressed while the system camera
 * is active, so haptics fired on the VisionScreen camera-flip button
 * will be silently dropped by iOS.  This is expected.
 */
import * as Haptics from 'expo-haptics';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Semantic haptic event types used throughout the app. */
export type HapticType =
  | 'selection' // Tab switch, clear button - lightest touch
  | 'toggle' // Start/Stop listen, camera flip - medium impact
  | 'success' // Task completed successfully
  | 'warning' // Non-critical alert
  | 'error'; // A hard error occurred

// Core function

/**
 * Fire a haptic vibration pattern matching the given semantic type.
 *
 * @param type - The kind of interaction that just happened.
 *
 * Mapping:
 *   selection -> `Haptics.selectionAsync()`        (lightest)
 *   toggle    -> `Haptics.impactAsync(Medium)`      (moderate)
 *   success   -> `Haptics.notificationAsync(Success)`
 *   warning   -> `Haptics.notificationAsync(Warning)`
 *   error     -> `Haptics.notificationAsync(Error)`
 */
export async function triggerHaptic(type: HapticType): Promise<void> {
  try {
    switch (type) {
      case 'selection':
        await Haptics.selectionAsync();
        break;

      case 'toggle':
        await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
        break;

      case 'success':
        await Haptics.notificationAsync(
          Haptics.NotificationFeedbackType.Success,
        );
        break;

      case 'warning':
        await Haptics.notificationAsync(
          Haptics.NotificationFeedbackType.Warning,
        );
        break;

      case 'error':
        await Haptics.notificationAsync(
          Haptics.NotificationFeedbackType.Error,
        );
        break;
    }
  } catch {
    // Haptic feedback is best-effort - never crash.
  }
}
