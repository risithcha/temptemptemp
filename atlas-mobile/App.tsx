/**
 * Atlas Mobile – Navigation Router
 *
 * This is the app's entry component.  It sets up a bottom-tab navigator
 * with two modes that mirror the desktop app:
 *   • Vision Assist  (camera + object detection)
 *   • Hearing Assist (live speech-to-text captioning)
 *
 * Each screen manages its own lifecycle (Camera `isActive`, speech
 * recognition start/stop) via `useIsFocused` + `useAppState` so
 * resources are released when the user switches tabs or backgrounds
 * the app.
 *
 * Accessibility:
 *   • Haptic feedback on every tab switch.
 *   • Android hardware back button: double-press to exit.
 */
import React from 'react';
import { StatusBar } from 'expo-status-bar';
import { View, Text, StyleSheet } from 'react-native';
import {
  NavigationContainer,
  useNavigationContainerRef,
} from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import Ionicons from '@expo/vector-icons/Ionicons';

import { HearingScreen } from './src/screens';
import WelcomeScreen from './src/screens/WelcomeScreen';
import SettingsScreen from './src/screens/SettingsScreen';
import { useAndroidBackHandler } from './src/hooks';
import { triggerHaptic } from './src/utils/haptics';
import { SettingsProvider } from './src/contexts/SettingsContext';
import { COLORS, TYPOGRAPHY, SPACING, SIZES, TAB_ACCENT } from './src/theme';

// ---------------------------------------------------------------------------
// Lazy-load VisionScreen so that react-native-vision-camera,
// react-native-fast-tflite, and react-native-worklets-core native modules
// are NOT eagerly initialised at app startup.  If they crash during init,
// the rest of the app can still render.
// ---------------------------------------------------------------------------
const LazyVisionScreen = React.lazy(
  () => import('./src/screens/VisionScreen'),
);

function VisionScreenWrapper() {
  return (
    <React.Suspense
      fallback={
        <View style={lazyStyles.container}>
          <StatusBar style="light" />
          <View style={lazyStyles.center}>
            <Text style={lazyStyles.text}>Loading Vision...</Text>
          </View>
        </View>
      }
    >
      <LazyVisionScreen />
    </React.Suspense>
  );
}

const lazyStyles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.background },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  text: { color: COLORS.textSecondary, fontSize: TYPOGRAPHY.body.fontSize },
});

// ---------------------------------------------------------------------------
// Error Boundary – catches JS crashes so the app doesn't just vanish
// ---------------------------------------------------------------------------
interface EBProps {
  children: React.ReactNode;
}
interface EBState {
  hasError: boolean;
  error: Error | null;
}

class ErrorBoundary extends React.Component<EBProps, EBState> {
  state: EBState = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error): EBState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('[Atlas ErrorBoundary]', error, info.componentStack);
  }

  render() {
    if (this.state.hasError) {
      return (
        <View style={ebStyles.container}>
          <StatusBar style="light" />
          <Ionicons name="bug-outline" size={64} color={COLORS.danger} />
          <Text style={ebStyles.title}>Atlas Crashed</Text>
          <Text style={ebStyles.message}>
            {this.state.error?.message ?? 'Unknown error'}
          </Text>
          <Text style={ebStyles.hint}>
            Please restart the app. If this keeps happening, check the
            development build error overlay for native crash details.
          </Text>
        </View>
      );
    }
    return this.props.children;
  }
}

const ebStyles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
    justifyContent: 'center',
    alignItems: 'center',
    padding: SPACING.xl,
  },
  title: {
    color: COLORS.text,
    fontSize: TYPOGRAPHY.title.fontSize,
    fontWeight: TYPOGRAPHY.title.fontWeight,
    marginTop: SPACING.md,
  },
  message: {
    color: COLORS.danger,
    fontSize: TYPOGRAPHY.caption.fontSize,
    textAlign: 'center',
    marginTop: SPACING.sm,
  },
  hint: {
    color: COLORS.textMuted,
    fontSize: TYPOGRAPHY.caption.fontSize,
    textAlign: 'center',
    marginTop: SPACING.md,
    lineHeight: 20,
  },
});

const Tab = createBottomTabNavigator();
const Stack = createNativeStackNavigator();

/**
 * Tab navigator – the core two-tab experience.
 */
function TabNavigator() {
  return (
    <Tab.Navigator
      screenOptions={{
        headerShown: false,
        animation: 'fade',
        lazy: true,
        freezeOnBlur: true,
        tabBarStyle: {
          backgroundColor: COLORS.background,
          borderTopColor: COLORS.surface,
          borderTopWidth: 1,
          height: SIZES.tabBarHeight,
          paddingBottom: SIZES.tabBarPaddingBottom,
          paddingTop: SPACING.sm,
        },
        tabBarInactiveTintColor: COLORS.textMuted,
        tabBarLabelStyle: {
          fontSize: TYPOGRAPHY.small.fontSize,
          fontWeight: '600',
        },
      }}
      screenListeners={{
        tabPress: () => {
          triggerHaptic('selection');
        },
      }}
    >
      <Tab.Screen
        name="Vision"
        component={VisionScreenWrapper}
        options={{
          tabBarLabel: 'Vision',
          tabBarActiveTintColor: TAB_ACCENT.Vision,
          tabBarIcon: ({ color, size, focused }) => (
            <Ionicons
              name={focused ? 'eye' : 'eye-outline'}
              size={size}
              color={color}
            />
          ),
          tabBarAccessibilityLabel: 'Vision Assist Mode',
        }}
      />
      <Tab.Screen
        name="Hearing"
        component={HearingScreen}
        options={{
          tabBarLabel: 'Hearing',
          tabBarActiveTintColor: TAB_ACCENT.Hearing,
          tabBarIcon: ({ color, size, focused }) => (
            <Ionicons
              name={focused ? 'ear' : 'ear-outline'}
              size={size}
              color={color}
            />
          ),
          tabBarAccessibilityLabel: 'Hearing Assist Mode',
        }}
      />
    </Tab.Navigator>
  );
}

/**
 * Root stack: Welcome --> Main (tabs).
 * The Welcome screen is shown once per launch
 * uses `replace` so the user can't swipe back to it.
 */
function AppNavigator() {
  return (
    <Stack.Navigator
      screenOptions={{ headerShown: false, animation: 'fade' }}
    >
      <Stack.Screen name="Welcome" component={WelcomeScreen} />
      <Stack.Screen name="Main" component={TabNavigator} />
      <Stack.Screen
        name="Settings"
        component={SettingsScreen}
        options={{ animation: 'slide_from_right' }}
      />
    </Stack.Navigator>
  );
}

export default function App() {
  const navigationRef = useNavigationContainerRef();
  useAndroidBackHandler(navigationRef);

  return (
    <SafeAreaProvider>
      <SettingsProvider>
        <ErrorBoundary>
          <StatusBar style="light" />
          <NavigationContainer ref={navigationRef}>
            <AppNavigator />
          </NavigationContainer>
        </ErrorBoundary>
      </SettingsProvider>
    </SafeAreaProvider>
  );
}
