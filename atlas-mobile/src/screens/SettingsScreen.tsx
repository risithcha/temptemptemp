/**
 * SettingsScreen - Accessibility preferences with persistent sliders.
 *
 * Controls TTS voice speed, caption text size, crisis mode sensitivity,
 * and the haptic vocabulary toggle.  All values are persisted to
 * AsyncStorage via SettingsContext.
 */
import { StatusBar } from 'expo-status-bar';
import {
  StyleSheet,
  Text,
  View,
  ScrollView,
  Switch,
  Platform,
} from 'react-native';
import { useCallback } from 'react';
import { useNavigation } from '@react-navigation/native';
import Slider from '@react-native-community/slider';
import Ionicons from '@expo/vector-icons/Ionicons';

import { AtlasHeader, ActionButton } from '../components';
import {
  useSettings,
  useSettingsActions,
  SETTINGS_DEFAULTS,
} from '../contexts/SettingsContext';
import { triggerHaptic } from '../utils/haptics';
import { COLORS, TYPOGRAPHY, SPACING, RADII } from '../theme';

export default function SettingsScreen() {
  const navigation = useNavigation<any>();
  const settings = useSettings();
  const actions = useSettingsActions();

  const goBack = useCallback(() => {
    triggerHaptic('selection');
    navigation.goBack();
  }, [navigation]);

  const handleReset = useCallback(() => {
    triggerHaptic('toggle');
    actions.resetToDefaults();
  }, [actions]);

  return (
    <View style={styles.container}>
      <StatusBar style="light" />
      <AtlasHeader
        subtitle="Settings"
        accentColor={COLORS.textSecondary}
        onHomePress={goBack}
      />

      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* TTS Voice Speed */}
        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Ionicons name="volume-high-outline" size={22} color={COLORS.secondary} />
            <Text style={styles.cardTitle}>TTS Voice Speed</Text>
          </View>
          <Text style={styles.cardDescription}>
            Adjust how fast detected objects are announced. Lower values are
            slower, higher values are faster.
          </Text>
          <Slider
            style={styles.slider}
            minimumValue={0.5}
            maximumValue={2.0}
            step={0.05}
            value={settings.ttsRate}
            onValueChange={actions.setTtsRate}
            minimumTrackTintColor={COLORS.secondary}
            maximumTrackTintColor={COLORS.surface}
            thumbTintColor={COLORS.secondary}
          />
          <View style={styles.sliderLabels}>
            <Text style={styles.sliderMin}>0.5x</Text>
            <Text style={styles.sliderValue}>
              {settings.ttsRate.toFixed(2)}x
            </Text>
            <Text style={styles.sliderMax}>2.0x</Text>
          </View>
        </View>

        {/* Caption Text Size */}
        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Ionicons name="text-outline" size={22} color={COLORS.primary} />
            <Text style={styles.cardTitle}>Caption Text Size</Text>
          </View>
          <Text style={styles.cardDescription}>
            Controls the font size used for live captions and OCR text.
          </Text>
          <Slider
            style={styles.slider}
            minimumValue={16}
            maximumValue={32}
            step={1}
            value={settings.captionFontSize}
            onValueChange={actions.setCaptionFontSize}
            minimumTrackTintColor={COLORS.primary}
            maximumTrackTintColor={COLORS.surface}
            thumbTintColor={COLORS.primary}
          />
          <View style={styles.sliderLabels}>
            <Text style={styles.sliderMin}>16</Text>
            <Text style={styles.sliderValue}>
              {settings.captionFontSize}px
            </Text>
            <Text style={styles.sliderMax}>32</Text>
          </View>
          <Text
            style={[
              styles.previewText,
              { fontSize: settings.captionFontSize },
            ]}
          >
            Preview text at current size
          </Text>
        </View>

        {/* Crisis Mode Sensitivity */}
        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Ionicons name="warning-outline" size={22} color={COLORS.warning} />
            <Text style={styles.cardTitle}>Crisis Mode Sensitivity</Text>
          </View>
          <Text style={styles.cardDescription}>
            FFT peak threshold for emergency sound detection. Lower values are
            more sensitive (more false positives), higher values require louder
            alarms.
          </Text>
          <Slider
            style={styles.slider}
            minimumValue={40}
            maximumValue={160}
            step={5}
            value={settings.crisisThreshold}
            onValueChange={actions.setCrisisThreshold}
            minimumTrackTintColor={COLORS.warning}
            maximumTrackTintColor={COLORS.surface}
            thumbTintColor={COLORS.warning}
          />
          <View style={styles.sliderLabels}>
            <Text style={styles.sliderMin}>40 (sensitive)</Text>
            <Text style={styles.sliderValue}>
              {settings.crisisThreshold}
            </Text>
            <Text style={styles.sliderMax}>160 (strict)</Text>
          </View>
        </View>

        {/* Haptic Patterns Toggle */}
        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Ionicons name="phone-portrait-outline" size={22} color={COLORS.textSecondary} />
            <Text style={styles.cardTitle}>Haptic Vocabulary</Text>
          </View>
          <Text style={styles.cardDescription}>
            Distinct vibration patterns for fire alarms (SOS pulse), sirens
            (alternating pulses), and new speaker detection (light tap).
          </Text>
          <View style={styles.switchRow}>
            <Text style={styles.switchLabel}>
              {settings.hapticPatternsEnabled ? 'Enabled' : 'Disabled'}
            </Text>
            <Switch
              value={settings.hapticPatternsEnabled}
              onValueChange={actions.setHapticPatternsEnabled}
              trackColor={{
                false: COLORS.surface,
                true: COLORS.primaryFaded,
              }}
              thumbColor={
                settings.hapticPatternsEnabled
                  ? COLORS.primary
                  : COLORS.textMuted
              }
            />
          </View>
        </View>

        {/* Reset */}
        <ActionButton
          label="Reset to Defaults"
          icon="refresh-outline"
          color={COLORS.danger}
          variant="outlined"
          onPress={handleReset}
          fullWidth
        />

        <View style={styles.bottomSpacer} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  scroll: {
    flex: 1,
  },
  scrollContent: {
    paddingHorizontal: SPACING.lg,
    paddingTop: SPACING.md,
    paddingBottom: SPACING.xxl,
  },

  card: {
    backgroundColor: COLORS.surface,
    borderRadius: RADII.xl,
    padding: SPACING.lg,
    marginBottom: SPACING.md,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.sm,
    marginBottom: SPACING.sm,
  },
  cardTitle: {
    ...TYPOGRAPHY.heading,
    color: COLORS.text,
    fontSize: 18,
  },
  cardDescription: {
    fontSize: TYPOGRAPHY.caption.fontSize,
    color: COLORS.textMuted,
    lineHeight: 20,
    marginBottom: SPACING.md,
  },

  slider: {
    width: '100%',
    height: 40,
  },
  sliderLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: SPACING.xs,
  },
  sliderMin: {
    fontSize: TYPOGRAPHY.small.fontSize,
    color: COLORS.textMuted,
  },
  sliderValue: {
    fontSize: TYPOGRAPHY.body.fontSize,
    color: COLORS.text,
    fontWeight: 'bold',
  },
  sliderMax: {
    fontSize: TYPOGRAPHY.small.fontSize,
    color: COLORS.textMuted,
  },

  previewText: {
    color: COLORS.text,
    marginTop: SPACING.md,
    textAlign: 'center',
    lineHeight: 36,
  },

  switchRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: SPACING.xs,
  },
  switchLabel: {
    fontSize: TYPOGRAPHY.body.fontSize,
    color: COLORS.text,
    fontWeight: '600',
  },

  bottomSpacer: {
    height: Platform.OS === 'ios' ? SPACING.xxl : SPACING.lg,
  },
});
