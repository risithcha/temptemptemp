/**
 * AtlasHeader – Unified header displayed on every screen.
 *
 * Shows the "ATLAS" brand mark on the left and an optional subtitle
 * (e.g. "Vision Assist" / "Hearing Assist") on the right.  Handles the
 * top safe-area inset internally so screens don't need to worry about
 * the notch / Dynamic Island.
 *
 * Variants:
 *   • `transparent` – for use over the camera (Vision).
 *   • solid (default) – dark background (Hearing / permission screens).
 */
import React from 'react';
import { StyleSheet, Text, View, TouchableOpacity } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import Ionicons from '@expo/vector-icons/Ionicons';
import { COLORS, TYPOGRAPHY, SPACING, SIZES, RADII } from '../theme';

interface AtlasHeaderProps {
  /** e.g. "Vision Assist" / "Hearing Assist" */
  subtitle?: string;
  /** Accent colour for the subtitle text (defaults to COLORS.primary) */
  accentColor?: string;
  /** Render with transparent background (for camera overlay) */
  transparent?: boolean;
  /** Optional content rendered on the far-right of the header row */
  rightContent?: React.ReactNode;
  /** If provided, a Home button is shown on the left that calls this handler. */
  onHomePress?: () => void;
  /** If provided, a gear icon is shown next to the home button. */
  onSettingsPress?: () => void;
}

export function AtlasHeader({
  subtitle,
  accentColor = COLORS.primary,
  transparent = false,
  rightContent,
  onHomePress,
  onSettingsPress,
}: AtlasHeaderProps) {
  return (
    <SafeAreaView
      edges={['top']}
      style={[
        styles.safeArea,
        transparent ? styles.transparent : styles.solid,
      ]}
    >
      <View style={styles.row}>
        {/* Home button (navigates back to WelcomeScreen) */}
        {onHomePress ? (
          <TouchableOpacity
            style={styles.homeButton}
            onPress={onHomePress}
            activeOpacity={0.7}
            accessibilityLabel="Return to home screen"
            accessibilityRole="button"
          >
            <Ionicons name="home-outline" size={20} color={COLORS.text} />
          </TouchableOpacity>
        ) : null}

        {/* Settings gear */}
        {onSettingsPress ? (
          <TouchableOpacity
            style={styles.homeButton}
            onPress={onSettingsPress}
            activeOpacity={0.7}
            accessibilityLabel="Open settings"
            accessibilityRole="button"
          >
            <Ionicons name="settings-outline" size={20} color={COLORS.text} />
          </TouchableOpacity>
        ) : null}

        {/* Brand mark */}
        <Text style={styles.logo}>ATLAS</Text>

        {/* Subtitle */}
        {subtitle ? (
          <Text style={[styles.subtitle, { color: accentColor }]}>
            {subtitle}
          </Text>
        ) : null}

        {/* Spacer pushes rightContent to the edge */}
        <View style={styles.spacer} />

        {rightContent}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    width: '100%',
  },
  solid: {
    backgroundColor: COLORS.background,
  },
  transparent: {
    backgroundColor: COLORS.overlay,
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    zIndex: 10,
  },
  row: {
    height: SIZES.headerHeight,
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: SPACING.lg,
  },
  homeButton: {
    width: 36,
    height: 36,
    borderRadius: RADII.md,
    backgroundColor: COLORS.whiteFaded,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: SPACING.sm,
  },
  logo: {
    ...TYPOGRAPHY.logoSmall,
    color: COLORS.text,
    marginRight: SPACING.sm,
  },
  subtitle: {
    ...TYPOGRAPHY.subtitle,
    // color set dynamically via props
  },
  spacer: {
    flex: 1,
  },
});
