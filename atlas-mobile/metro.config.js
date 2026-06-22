const { getDefaultConfig } = require('expo/metro-config');

/** @type {import('expo/metro-config').MetroConfig} */
const config = getDefaultConfig(__dirname);

// Add tflite and txt as asset extensions for TensorFlow Lite models and labels
config.resolver.assetExts.push('tflite', 'txt');

// react-native-audio-api's Gradle task downloads iOS/macOS xcframeworks with symlinks.
// Metro's FallbackWatcher hits EACCES on those junctions on Windows — exclude them.
config.resolver.blockList = [
  ...config.resolver.blockList,
  /react-native-audio-api[\\/]common[\\/]cpp[\\/]audioapi[\\/]external(?:[\\/].*)?$/,
  /react-native-audio-api[\\/]android[\\/]src[\\/]main[\\/]jniLibs(?:[\\/].*)?$/,
  /react-native-audio-api[\\/]android[\\/]audioapi-binaries-temp(?:[\\/].*)?$/,
];

module.exports = config;
