/**
 * Atlas Mobile – Android Hardware Back-Button Handler
 *
 * In a flat bottom-tab layout the default Android back-press behaviour
 * immediately exits the app, which is bad for users who just want
 * to switch tabs.
 *
 * This hook implements the pattern:
 *   1. If the user is NOT on the primary tab ("Vision") -> navigate
 *      there instead of exiting.
 *   2. If the user IS on the primary tab -> require a **double-press**
 *      within 2 seconds to exit, showing a toast on the first press.
 *
 * Implementation note:
 *   We use a NavigationContainerRef (passed from the App component)
 *   instead of useNavigationState / useFocusEffect because this hook
 *   is called at the app root level, OUTSIDE of any navigator screen.
 *   useNavigationState requires the calling component to be a child of
 *   a <Tab.Screen> or <Stack.Screen>, which AppNavigator is not.
 *   Thanks "Navigating without the navigation prop".
 */
import { useEffect, useRef } from 'react';
import { BackHandler, ToastAndroid, Platform } from 'react-native';
import type { NavigationContainerRef } from '@react-navigation/native';

/** The tab name that acts as "home". */
const PRIMARY_TAB = 'Vision';

/** Time window (ms) in which a second back press exits the app. */
const EXIT_WINDOW_MS = 2000;

/** Whether the current platform is Android. */
const IS_ANDROID = Platform.OS === 'android';

/**
 * @param navigationRef – ref obtained via `useNavigationContainerRef()`
 *   in the App component and passed to `<NavigationContainer ref={…}>`.
 */
export function useAndroidBackHandler(
  navigationRef: React.RefObject<NavigationContainerRef<any> | null>,
): void {
  const lastBackPress = useRef<number>(0);

  useEffect(() => {
    // iOS has no hardware back button – skip listener entirely.
    if (!IS_ANDROID) return;

    const onBackPress = (): boolean => {
      // Safety: if the navigation tree isn't ready, let the system handle it.
      if (!navigationRef.current?.isReady()) return false;

      const currentRouteName = navigationRef.current.getCurrentRoute()?.name;

      // Not on primary tab -> navigate there.
      if (currentRouteName !== PRIMARY_TAB) {
        navigationRef.current.navigate(PRIMARY_TAB as never);
        return true; // handled
      }

      // On primary tab -> double-press gate.
      const now = Date.now();
      if (now - lastBackPress.current < EXIT_WINDOW_MS) {
        // Second press within window -> let the system exit.
        return false;
      }

      lastBackPress.current = now;
      ToastAndroid.show('Press back again to exit', ToastAndroid.SHORT);
      return true; // handled - wait for second press
    };

    const subscription = BackHandler.addEventListener(
      'hardwareBackPress',
      onBackPress,
    );

    return () => subscription.remove();
  }, [navigationRef]);
}
