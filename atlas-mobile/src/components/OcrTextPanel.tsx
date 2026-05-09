/**
 * OcrTextPanel - Collapsible overlay that displays OCR-detected text
 * on top of the camera preview in Vision Assist mode.
 *
 * Tapping the panel toggles between an expanded scrollable view and a
 * compact "Text detected" badge so it doesn't permanently block the
 * camera feed.
 */
import { useState, useRef, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  ScrollView,
  TouchableOpacity,
} from 'react-native';
import Ionicons from '@expo/vector-icons/Ionicons';

import { COLORS, TYPOGRAPHY, SPACING, RADII } from '../theme';

export interface OcrTextPanelProps {
  text: string;
  /** Override font size for the OCR text (from user settings). */
  fontSize?: number;
}

export function OcrTextPanel({ text, fontSize }: OcrTextPanelProps) {
  const [expanded, setExpanded] = useState(false);
  const scrollRef = useRef<ScrollView>(null);

  useEffect(() => {
    if (expanded && text) {
      const t = setTimeout(() => {
        scrollRef.current?.scrollToEnd({ animated: true });
      }, 100);
      return () => clearTimeout(t);
    }
  }, [text, expanded]);

  if (!text || !text.trim()) return null;

  if (!expanded) {
    return (
      <TouchableOpacity
        style={styles.badge}
        onPress={() => setExpanded(true)}
        activeOpacity={0.8}
      >
        <Ionicons name="text-outline" size={16} color={COLORS.primary} />
        <Text style={styles.badgeText} numberOfLines={1}>
          Text detected - tap to view
        </Text>
      </TouchableOpacity>
    );
  }

  return (
    <View style={styles.panel}>
      <TouchableOpacity
        style={styles.header}
        onPress={() => setExpanded(false)}
        activeOpacity={0.8}
      >
        <Ionicons name="text-outline" size={18} color={COLORS.primary} />
        <Text style={styles.headerText}>Detected Text (OCR)</Text>
        <Ionicons
          name="chevron-down-outline"
          size={18}
          color={COLORS.textMuted}
        />
      </TouchableOpacity>

      <ScrollView
        ref={scrollRef}
        style={styles.scroll}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator
      >
        <Text style={[styles.ocrText, fontSize != null && { fontSize }]}>{text}</Text>
      </ScrollView>
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
    maxWidth: 200,
  },

  panel: {
    position: 'absolute',
    bottom: 130,
    left: SPACING.md,
    right: SPACING.md,
    maxHeight: 180,
    backgroundColor: 'rgba(0, 0, 0, 0.85)',
    borderRadius: RADII.lg,
    borderWidth: 1,
    borderColor: COLORS.primary,
    overflow: 'hidden',
    zIndex: 5,
    elevation: 5,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    gap: SPACING.sm,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.surface,
  },
  headerText: {
    flex: 1,
    color: COLORS.primary,
    fontSize: TYPOGRAPHY.caption.fontSize,
    fontWeight: 'bold',
  },
  scroll: {
    maxHeight: 120,
  },
  scrollContent: {
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
  },
  ocrText: {
    color: COLORS.text,
    fontSize: TYPOGRAPHY.body.fontSize,
    lineHeight: 24,
  },
});
