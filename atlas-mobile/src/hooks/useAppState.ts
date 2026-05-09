/**
 * Atlas Mobile – useAppState Hook
 *
 * Tiny hook that tracks the React Native `AppState` value reactively.
 * Returns `'active'`, `'background'`, or on iOS `'inactive'`.
 *
 * Used alongside `useIsFocused()` to compute whether heavy resources
 * (camera, microphone) should be running
 */
import { useState, useEffect } from 'react';
import { AppState, type AppStateStatus } from 'react-native';

export function useAppState(): AppStateStatus {
  const [appState, setAppState] = useState<AppStateStatus>(
    AppState.currentState,
  );

  useEffect(() => {
    const subscription = AppState.addEventListener('change', setAppState);
    return () => subscription.remove();
  }, []);

  return appState;
}
