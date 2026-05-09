const { getDefaultConfig } = require('expo/metro-config');

/** @type {import('expo/metro-config').MetroConfig} */
const config = getDefaultConfig(__dirname);

// Add tflite and txt as asset extensions for TensorFlow Lite models and labels
config.resolver.assetExts.push('tflite', 'txt');

module.exports = config;
