/**
 * ActionButton – Standardised pressable button used across all screens.
 *
 * Two visual variants:
 *   • `filled`   – solid background (primary action: Start/Stop, Grant Permission)
 *   • `outlined` – transparent with a coloured border (secondary action: Clear)
 *
 * Both variants share the same height, border-radius, font, and padding so
 * Vision and Hearing screens look identical.
 */
import React from 'react';
import { StyleSheet, Text, TouchableOpacity, type ViewStyle } from 'react-native';
import Ionicons from '@expo/vector-icons/Ionicons';
import { COLORS, TYPOGRAPHY, SPACING, RADII, SIZES } from '../theme';

type IoniconsName = React.ComponentProps<typeof Ionicons>['name'];

interface ActionButtonProps {
  label: string;
  /** Ionicons icon name rendered to the left of the label */
  icon?: IoniconsName;
  iconSize?: number;
  /** The dominant colour — fill for `filled`, border for `outlined` */
  color?: string;
  variant?: 'filled' | 'outlined';
  onPress: () => void;
  disabled?: boolean;
  /** Stretch to fill available width */
  fullWidth?: boolean;
  style?: ViewStyle;
}

export function ActionButton({
  label,
  icon,
  iconSize = 24,
  color = COLORS.primary,
  variant = 'filled',
  onPress,
  disabled = false,
  fullWidth = false,
  style,
}: ActionButtonProps) {
  const isFilled = variant === 'filled';

  const containerStyle: ViewStyle[] = [
    styles.base,
    isFilled
      ? { backgroundColor: color }
      : { borderWidth: 2, borderColor: color, backgroundColor: 'transparent' },
    ...(fullWidth ? [styles.fullWidth] : []),
    ...(disabled ? [styles.disabled] : []),
    ...(style ? [style] : []),
  ];

  const textColor = isFilled ? COLORS.text : color;
  const disabledTextColor = COLORS.textMuted;

  return (
    <TouchableOpacity
      style={containerStyle}
      onPress={onPress}
      disabled={disabled}
      activeOpacity={0.8}
    >
      {icon ? (
        <Ionicons
          name={icon}
          size={iconSize}
          color={disabled ? disabledTextColor : textColor}
          style={styles.icon}
        />
      ) : null}
      <Text
        style={[
          styles.label,
          { color: disabled ? disabledTextColor : textColor },
        ]}
      >
        {label}
      </Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  base: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: SIZES.buttonMinHeight,
    borderRadius: RADII.lg,
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.md,
  },
  fullWidth: {
    flex: 1,
  },
  disabled: {
    opacity: 0.45,
  },
  icon: {
    marginRight: SPACING.sm,
  },
  label: {
    ...TYPOGRAPHY.button,
  },
});
