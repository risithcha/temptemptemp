export {
  DeepgramTransport,
  floatTo16BitPCM,
  type DeepgramTransportCallbacks,
} from './transport';

export {
  wordsToSegments,
  mergeSegments,
  extractAlternative,
  speakerLabel,
  type DiarizedSegment,
  type DeepgramWord,
  type DeepgramResultMessage,
} from './diarization';
