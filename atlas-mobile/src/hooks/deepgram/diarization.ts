/**
 * Deepgram diarization mapper.
 *
 * Converts Deepgram Nova-3 streaming results (with `diarize=true`) into the
 * app's `{ speaker, text }` segment shape.  Deepgram tags each word with an
 * integer `speaker` index; we group consecutive same-speaker words into
 * segments and label them "Speaker N" to match the existing UI.
 */

/** Structurally identical to `TranscriptSegment` in useSpeechRecognition. */
export interface DiarizedSegment {
  speaker: string;
  text: string;
}

/** A single word in a Deepgram alternative. */
export interface DeepgramWord {
  word: string;
  punctuated_word?: string;
  start: number;
  end: number;
  confidence: number;
  /** 0-based speaker index (present when diarize=true). */
  speaker?: number;
}

/** A streaming "Results" message from Deepgram's listen WebSocket. */
export interface DeepgramResultMessage {
  type?: string;
  is_final?: boolean;
  speech_final?: boolean;
  channel?: {
    alternatives?: Array<{
      transcript?: string;
      words?: DeepgramWord[];
    }>;
  };
}

/** Map a 0-based speaker index to a display label ("Speaker 1", ...). */
export function speakerLabel(index: number): string {
  return `Speaker ${index + 1}`;
}

/** Extract the best alternative's transcript + words from a result message. */
export function extractAlternative(msg: DeepgramResultMessage): {
  transcript: string;
  words: DeepgramWord[];
} {
  const alt = msg.channel?.alternatives?.[0];
  return {
    transcript: (alt?.transcript ?? '').trim(),
    words: alt?.words ?? [],
  };
}

/**
 * Group a result's words into speaker-tagged segments.  When no per-word
 * speaker data is present (diarization off / single speaker), this yields a
 * single segment under "Speaker 1".
 */
export function wordsToSegments(
  words: DeepgramWord[],
  fallbackTranscript = '',
): DiarizedSegment[] {
  if (words.length === 0) {
    const text = fallbackTranscript.trim();
    return text ? [{ speaker: speakerLabel(0), text }] : [];
  }

  const segments: DiarizedSegment[] = [];
  let current: DiarizedSegment | null = null;
  let currentSpeaker = -1;

  for (const w of words) {
    const spk = w.speaker ?? 0;
    const token = (w.punctuated_word ?? w.word ?? '').trim();
    if (!token) continue;

    if (current == null || spk !== currentSpeaker) {
      current = { speaker: speakerLabel(spk), text: token };
      segments.push(current);
      currentSpeaker = spk;
    } else {
      current.text += ` ${token}`;
    }
  }

  return segments;
}

/**
 * Append finalized segments onto the running list, merging across the boundary
 * when the same speaker continues (so a speaker's turn isn't split just because
 * Deepgram emitted two final results for it).
 */
export function mergeSegments(
  existing: DiarizedSegment[],
  incoming: DiarizedSegment[],
): DiarizedSegment[] {
  if (incoming.length === 0) return existing;
  const merged = existing.map((s) => ({ ...s }));

  for (const seg of incoming) {
    const last = merged.length > 0 ? merged[merged.length - 1] : null;
    if (last && last.speaker === seg.speaker) {
      last.text = last.text ? `${last.text} ${seg.text}` : seg.text;
    } else {
      merged.push({ ...seg });
    }
  }
  return merged;
}
