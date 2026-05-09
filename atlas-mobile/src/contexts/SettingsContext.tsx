/**
 * Atlas Mobile - Persistent Accessibility Settings
 *
 * Loads user preferences from AsyncStorage on mount and exposes them via
 * React Context.  Every setter persists the new value immediately.
 *
 * Consumers call `useSettings()` to read values and `useSettingsActions()`
 * to update them (split to avoid unnecessary re-renders in read-only
 * consumers).
 */
import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useMemo,
} from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------
export interface AppSettings {
  /** TTS voice rate (0.5 - 2.0, default 0.95) */
  ttsRate: number;
  /** Caption / OCR text font size (16 - 32, default 22) */
  captionFontSize: number;
  /** FFT peak threshold for crisis mode (40 - 160, default 80) */
  crisisThreshold: number;
  /** Whether the advanced haptic vocabulary is enabled */
  hapticPatternsEnabled: boolean;
}

export interface SettingsActions {
  setTtsRate: (v: number) => void;
  setCaptionFontSize: (v: number) => void;
  setCrisisThreshold: (v: number) => void;
  setHapticPatternsEnabled: (v: boolean) => void;
  resetToDefaults: () => void;
}

const DEFAULTS: AppSettings = {
  ttsRate: 0.95,
  captionFontSize: 22,
  crisisThreshold: 80,
  hapticPatternsEnabled: true,
};

const STORAGE_KEY = '@atlas/settings';

// ---------------------------------------------------------------------------
// Contexts
// ---------------------------------------------------------------------------
const SettingsCtx = createContext<AppSettings>(DEFAULTS);
const SettingsActionsCtx = createContext<SettingsActions>({
  setTtsRate: () => {},
  setCaptionFontSize: () => {},
  setCrisisThreshold: () => {},
  setHapticPatternsEnabled: () => {},
  resetToDefaults: () => {},
});

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------
export function SettingsProvider({ children }: { children: React.ReactNode }) {
  const [settings, setSettings] = useState<AppSettings>(DEFAULTS);
  const [loaded, setLoaded] = useState(false);

  // Load persisted settings once on mount
  useEffect(() => {
    (async () => {
      try {
        const raw = await AsyncStorage.getItem(STORAGE_KEY);
        if (raw) {
          const parsed = JSON.parse(raw) as Partial<AppSettings>;
          setSettings((prev) => ({ ...prev, ...parsed }));
        }
      } catch {
        // First launch or corrupted data - use defaults.
      }
      setLoaded(true);
    })();
  }, []);

  const persist = useCallback((next: AppSettings) => {
    setSettings(next);
    AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(next)).catch(() => {});
  }, []);

  const actions = useMemo<SettingsActions>(
    () => ({
      setTtsRate: (v) =>
        persist({ ...settings, ttsRate: Math.max(0.5, Math.min(2.0, v)) }),
      setCaptionFontSize: (v) =>
        persist({ ...settings, captionFontSize: Math.max(16, Math.min(32, v)) }),
      setCrisisThreshold: (v) =>
        persist({ ...settings, crisisThreshold: Math.max(40, Math.min(160, v)) }),
      setHapticPatternsEnabled: (v) =>
        persist({ ...settings, hapticPatternsEnabled: v }),
      resetToDefaults: () => persist({ ...DEFAULTS }),
    }),
    [settings, persist],
  );

  if (!loaded) return null;

  return (
    <SettingsCtx.Provider value={settings}>
      <SettingsActionsCtx.Provider value={actions}>
        {children}
      </SettingsActionsCtx.Provider>
    </SettingsCtx.Provider>
  );
}

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------
export function useSettings(): AppSettings {
  return useContext(SettingsCtx);
}

export function useSettingsActions(): SettingsActions {
  return useContext(SettingsActionsCtx);
}

export { DEFAULTS as SETTINGS_DEFAULTS };
