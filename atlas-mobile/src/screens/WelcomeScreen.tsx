/**
 * WelcomeScreen – First screen the user sees on app launch.
 *
 * Displays the Atlas brand, a short tagline, and two entry-point buttons
 * that match the desktop landing page:
 *   • Vision Assist  (blue)
 *   • Hearing Assist (green)
 *
 * Tapping either navigates into the main tab navigator and lands on
 * the chosen tab.
 */
import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import Ionicons from '@expo/vector-icons/Ionicons';

import { ActionButton } from '../components';
import { COLORS, TYPOGRAPHY, SPACING, RADII } from '../theme';

interface WelcomeScreenProps {
  navigation: any;
}

export default function WelcomeScreen({ navigation }: WelcomeScreenProps) {
  const handleVision = () => {
    navigation.navigate('Main', { screen: 'Vision' });
  };

  const handleHearing = () => {
    navigation.navigate('Main', { screen: 'Hearing' });
  };

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.container}>
        {/* Brand section */}
        <View style={styles.brandSection}>
          <Text style={styles.logo}>ATLAS</Text>
          <Text style={styles.tagline}>Your Personal Accessibility Assistant</Text>
        </View>

        {/* Feature cards */}
        <View style={styles.cardSection}>
          <Text style={styles.welcomeText}>Welcome to Atlas</Text>
          <Text style={styles.description}>
            Choose a mode to get started. You can switch between them anytime
            using the bottom tabs.
          </Text>

          {/* Vision card */}
          <View style={[styles.card, { borderColor: COLORS.secondary }]}>
            <View style={styles.cardHeader}>
              <View style={[styles.iconCircle, { backgroundColor: COLORS.secondaryFaded }]}>
                <Ionicons name="eye" size={28} color={COLORS.secondary} />
              </View>
              <View style={styles.cardTextGroup}>
                <Text style={[styles.cardTitle, { color: COLORS.secondary }]}>
                  Vision Assist
                </Text>
                <Text style={styles.cardDescription}>
                  Real-time object detection using your camera
                </Text>
              </View>
            </View>
            <ActionButton
              label="Start Vision"
              icon="eye-outline"
              color={COLORS.secondary}
              variant="filled"
              onPress={handleVision}
              fullWidth
            />
          </View>

          {/* Hearing card */}
          <View style={[styles.card, { borderColor: COLORS.primary }]}>
            <View style={styles.cardHeader}>
              <View style={[styles.iconCircle, { backgroundColor: COLORS.primaryFaded }]}>
                <Ionicons name="ear" size={28} color={COLORS.primary} />
              </View>
              <View style={styles.cardTextGroup}>
                <Text style={[styles.cardTitle, { color: COLORS.primary }]}>
                  Hearing Assist
                </Text>
                <Text style={styles.cardDescription}>
                  Live speech-to-text captioning
                </Text>
              </View>
            </View>
            <ActionButton
              label="Start Hearing"
              icon="ear-outline"
              color={COLORS.primary}
              variant="filled"
              onPress={handleHearing}
              fullWidth
            />
          </View>
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  container: {
    flex: 1,
    paddingHorizontal: SPACING.lg,
  },

  // Brand
  brandSection: {
    alignItems: 'center',
    paddingTop: SPACING.xxl,
    paddingBottom: SPACING.lg,
  },
  logo: {
    ...TYPOGRAPHY.logo,
    color: COLORS.text,
  },
  tagline: {
    ...TYPOGRAPHY.subtitle,
    color: COLORS.textSecondary,
    marginTop: SPACING.sm,
    textAlign: 'center',
  },

  // Content
  cardSection: {
    flex: 1,
    justifyContent: 'center',
  },
  welcomeText: {
    ...TYPOGRAPHY.title,
    color: COLORS.text,
    textAlign: 'center',
    marginBottom: SPACING.sm,
  },
  description: {
    fontSize: TYPOGRAPHY.body.fontSize,
    color: COLORS.textMuted,
    textAlign: 'center',
    lineHeight: 24,
    marginBottom: SPACING.xl,
    paddingHorizontal: SPACING.sm,
  },

  // Cards
  card: {
    backgroundColor: COLORS.surface,
    borderWidth: 1,
    borderRadius: RADII.xl,
    padding: SPACING.lg,
    marginBottom: SPACING.md,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: SPACING.md,
  },
  iconCircle: {
    width: 52,
    height: 52,
    borderRadius: 26,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: SPACING.md,
  },
  cardTextGroup: {
    flex: 1,
  },
  cardTitle: {
    ...TYPOGRAPHY.heading,
  },
  cardDescription: {
    fontSize: TYPOGRAPHY.caption.fontSize,
    color: COLORS.textSecondary,
    marginTop: SPACING.xs,
  },
});
