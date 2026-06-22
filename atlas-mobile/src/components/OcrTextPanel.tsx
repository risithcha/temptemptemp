/**
 * Passive OCR indicator for sighted helpers — blind users rely on TTS + taps.
 */
import { StyleSheet, Text, View } from 'react-native';
import Ionicons from '@expo/vector-icons/Ionicons';

import { COLORS, TYPOGRAPHY, SPACING, RADII } from '../theme';

export interface OcrTextPanelProps {
  text: string;
  /** @deprecated Visual-only; caption size unused in badge mode. */
  fontSize?: number;
  /** e.g. "playing" | "paused" — optional hint for sighted helpers. */
  playbackState?: 'idle' | 'playing' | 'paused';
}

export function OcrTextPanel({ text, playbackState = 'idle' }: OcrTextPanelProps) {
  if (!text || !text.trim()) return null;

  const statusLabel =
    playbackState === 'playing'
      ? 'Reading'
      : playbackState === 'paused'
        ? 'Paused'
        : 'Text detected';

  return (
    <View
      style={styles.badge}
      pointerEvents="none"
      accessibilityElementsHidden
      importantForAccessibility="no"
    >
      <Ionicons name="text-outline" size={16} color={COLORS.primary} />
      <Text style={styles.badgeText} numberOfLines={1}>
        {statusLabel}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    position: 'absolute',
    bottom: 130,
    alignSelf: 'center',
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.overlay,
    borderRadius: RADII.round,
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    gap: SPACING.sm,
    zIndex: 5,
    elevation: 5,
  },
  badgeText: {
    color: COLORS.primary,
    fontSize: TYPOGRAPHY.caption.fontSize,
    fontWeight: '600',
  },
});
