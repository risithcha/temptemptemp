/**
 * Encodes a Float32Array containing [-1.0, 1.0] PCM data into a Base64-encoded WAV string.
 * @param pcmData The raw audio samples.
 * @param sampleRate The sample rate (e.g., 16000 or 44100).
 * @param numChannels The number of audio channels (e.g., 1 for mono).
 * @returns A base64 string containing the full WAV file.
 */
export function encodePcmToBase64Wav(
  pcmData: Float32Array,
  sampleRate: number,
  numChannels: number = 1,
): string {
  const bytesPerSample = 2; // 16-bit PCM
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = pcmData.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  // Helper to write strings
  const writeString = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  // RIFF chunk descriptor
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true); // chunk size
  writeString(8, 'WAVE');

  // fmt sub-chunk
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true); // subchunk1size (16 for PCM)
  view.setUint16(20, 1, true); // audio format (1 = PCM)
  view.setUint16(22, numChannels, true); // num channels
  view.setUint32(24, sampleRate, true); // sample rate
  view.setUint32(28, byteRate, true); // byte rate
  view.setUint16(32, blockAlign, true); // block align
  view.setUint16(34, bytesPerSample * 8, true); // bits per sample

  // data sub-chunk
  writeString(36, 'data');
  view.setUint32(40, dataSize, true); // subchunk2size

  // Write PCM samples
  let offset = 44;
  for (let i = 0; i < pcmData.length; i++) {
    // Clamp sample between -1.0 and +1.0
    const s = Math.max(-1, Math.min(1, pcmData[i]));
    // Convert to 16-bit signed integer
    const val = s < 0 ? s * 0x8000 : s * 0x7fff;
    view.setInt16(offset, val, true);
    offset += 2;
  }

  const uint8View = new Uint8Array(buffer);
  return encodeBase64(uint8View);
}

const BASE64_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';

function encodeBase64(bytes: Uint8Array): string {
  let result = '';
  const len = bytes.length;
  for (let i = 0; i < len; i += 3) {
    const b1 = bytes[i];
    const b2 = i + 1 < len ? bytes[i + 1] : 0;
    const b3 = i + 2 < len ? bytes[i + 2] : 0;

    const enc1 = b1 >> 2;
    const enc2 = ((b1 & 3) << 4) | (b2 >> 4);
    const enc3 = ((b2 & 15) << 2) | (b3 >> 6);
    const enc4 = b3 & 63;

    result += BASE64_CHARS[enc1];
    result += BASE64_CHARS[enc2];
    result += i + 1 < len ? BASE64_CHARS[enc3] : '=';
    result += i + 2 < len ? BASE64_CHARS[enc4] : '=';
  }
  return result;
}
