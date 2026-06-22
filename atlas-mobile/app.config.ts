import type { ConfigContext, ExpoConfig } from 'expo/config';

/**
 * Dynamic Expo config.
 *
 * Expo loads the static `app.json` first and passes it in as `config`; we layer
 * on the runtime API keys from the environment so they never live in
 * source control.  Provide them via EAS secrets or a local shell env var.
 *
 * Read at runtime with:
 *   Constants.expoConfig?.extra?.deepgramKey
 *   Constants.expoConfig?.extra?.geminiKey
 */
export default ({ config }: ConfigContext): ExpoConfig => ({
  ...config,
  // `name`/`slug` are guaranteed by app.json; assert for the ExpoConfig type.
  name: config.name ?? 'Atlas',
  slug: config.slug ?? 'atlas-mobile',
  extra: {
    ...config.extra,
    deepgramKey: process.env.DEEPGRAM_API_KEY ?? '',
    geminiKey: process.env.GEMINI_API_KEY ?? '',
  },
});
