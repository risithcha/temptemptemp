import { registerRootComponent } from 'expo';
import { LogBox } from 'react-native';

import App from './App';

// Suppress noisy warnings in dev (still visible in logcat)
LogBox.ignoreLogs([
  'ViewPropTypes will be removed',
  'ColorPropType will be removed',
]);

// Global unhandled JS error handler – logs to console so we can
// see the error in dev-client or adb logcat even if the app crashes.
const originalHandler = ErrorUtils.getGlobalHandler();
ErrorUtils.setGlobalHandler((error, isFatal) => {
  console.error(
    `[Atlas Global Error] ${isFatal ? 'FATAL' : 'non-fatal'}:`,
    error?.message ?? error,
    error?.stack ?? '',
  );
  originalHandler(error, isFatal);
});

// registerRootComponent calls AppRegistry.registerComponent('main', () => App);
// It also ensures that whether you load the app in Expo Go or in a native build,
// the environment is set up appropriately
registerRootComponent(App);
