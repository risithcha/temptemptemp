import { useState, useCallback } from 'react';
import { encodePcmToBase64Wav } from '../utils/wav_encoder';
import { EMERGENCY_CHIRP_HINT, type AlarmAlert } from './useAlarmDetector';

const GEMINI_MODELS = ['gemini-2.5-flash', 'gemini-2.0-flash'] as const;
const MAX_ROUNDS = 4;

const ALARM_MESSAGES: Record<string, string> = {
  FIRE_ALARM: 'FIRE ALARM DETECTED',
  SMOKE_ALARM: 'SMOKE ALARM DETECTED',
  SIREN: 'EMERGENCY SIREN DETECTED',
};

const SYSTEM_PROMPT =
  'You classify emergency audio for a deaf user\'s safety app. Output JSON only.\n\n' +
  'CRITICAL – FIRE_ALARM vs SMOKE_ALARM:\n' +
  '• FIRE_ALARM (building evacuation): repeated alarm bursts meant to evacuate. Includes:\n' +
  '  - T3 horn code: three beeps/buzzes, pause, repeat (very common fire evac signal)\n' +
  '  - T4 horn code: four beeps, pause, repeat\n' +
  '  - Continuous or pulsing horns, bells, klaxons, whoops, sirens inside buildings\n' +
  '  - ANY grouped/repeating high-pitched beeps with pauses between groups\n' +
  '  - Loud alarm tones even if they sound like "beeps" through a phone speaker\n' +
  '• SMOKE_ALARM (residential detector): periodic high-pitched beeps/chirps from a wall smoke or CO detector.\n' +
  '  - Repeating chirp every few seconds (common smoke alarm pattern)\n' +
  '  - Single isolated thin chirp (low-battery warning)\n' +
  '  - Steady beeps without smooth frequency sweeps – NOT a vehicle siren\n' +
  '  - Through a phone speaker these often sound like mid-range beeps, not thin 3 kHz tones\n' +
  '  - Do NOT label smooth rising/falling vehicle siren wails as SMOKE_ALARM\n\n' +
  'If the clip has repeating beep groups in threes or fours (T3/T4 evac code) → FIRE_ALARM, not SMOKE_ALARM.\n' +
  'Phone speakers distort alarms: evacuation horns and T3/T4 grouped beeps often sound like generic mid-pitch beeps — still classify as FIRE_ALARM if beeps arrive in groups of 3 or 4.\n' +
  'SMOKE_ALARM requires a lone detector chirp (one beep then long silence), NOT grouped horn codes.\n' +
  'When unsure between fire and smoke → choose SMOKE_ALARM only for a single isolated chirp with long pause; choose FIRE_ALARM for any grouped beep pattern.\n\n' +
  '• SIREN: emergency vehicle siren ONLY – clear smooth rising/falling frequency sweeps (wail/yelp).\n' +
  '  - NOT periodic smoke-detector chirps\n' +
  '• FALSE_ALARM: speech, music, TV, appliances, computer fans, electrical hum, room tone, HVAC, silence, or ordinary background noise.\n' +
  'If the audio is steady background noise with no distinct alarm or siren pattern, return FALSE_ALARM.';

const USER_PROMPT_BASE =
  'Classify this 3-second clip. Only confirm an alarm if a clear emergency sound is present. ' +
  'Steady room noise, computer/electronics hum, or ambiguous tones = FALSE_ALARM. ' +
  'Periodic smoke-detector chirps (single beeps repeating) = SMOKE_ALARM. ' +
  'Grouped repeating beeps (3 or 4 per group, T3/T4 evac) = FIRE_ALARM. ' +
  'Smooth rising/falling vehicle siren wail = SIREN.';

const RESPONSE_SCHEMA = {
  type: 'OBJECT',
  properties: {
    type: {
      type: 'STRING',
      enum: ['FIRE_ALARM', 'SMOKE_ALARM', 'SIREN', 'FALSE_ALARM'],
      description:
        'FIRE_ALARM for evacuation horns and grouped beep codes (T3/T4). ' +
        'SMOKE_ALARM only for a lone residential detector chirp, never grouped beeps.',
    },
    confidence: {
      type: 'NUMBER',
      description: 'Confidence 0.0–1.0',
    },
  },
  required: ['type', 'confidence'],
};

function geminiEndpoint(model: (typeof GEMINI_MODELS)[number]): string {
  return `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export type ClassifyAlarmResult =
  | { status: 'confirmed'; alert: AlarmAlert }
  | { status: 'rejected' }
  | { status: 'unavailable'; reason: string };

export interface UseGeminiAudioResult {
  classifyAlarm: (
    pcmData: Float32Array,
    sampleRate: number,
    fftHint?: string,
  ) => Promise<ClassifyAlarmResult>;
  isAnalyzing: boolean;
  error: string | null;
}

function extractResponseText(json: unknown): string {
  const candidate = (json as { candidates?: Array<{ content?: { parts?: Array<{ text?: string }> } }> })
    ?.candidates?.[0];
  const parts = candidate?.content?.parts ?? [];
  return parts
    .map((part) => part.text?.trim() ?? '')
    .filter(Boolean)
    .join('\n')
    .trim();
}

function parseAlarmPayload(text: string): AlarmAlert | null {
  if (!text) return null;

  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidates = [text, fenced?.[1]?.trim()].filter(Boolean) as string[];

  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate) as AlarmAlert;
      if (parsed?.type) return parsed;
    } catch {
      // try substring next
    }

    const jsonMatch = candidate.match(/\{[\s\S]*\}/);
    if (!jsonMatch) continue;

    try {
      const parsed = JSON.parse(jsonMatch[0]) as AlarmAlert;
      if (parsed?.type) return parsed;
    } catch {
      // continue
    }
  }

  return null;
}

function normalizeAlert(result: AlarmAlert): AlarmAlert {
  const raw = result.type.toUpperCase().replace(/\s+/g, '_');
  const type =
    raw === 'FIRE_ALARM' || raw === 'SMOKE_ALARM' || raw === 'SIREN' ? raw : 'FIRE_ALARM';
  return {
    type,
    message: ALARM_MESSAGES[type],
    confidence: typeof result.confidence === 'number' ? result.confidence : 0.8,
  };
}

async function requestClassification(
  apiKey: string,
  pcmData: Float32Array,
  sampleRate: number,
  fftHint: string | undefined,
  model: (typeof GEMINI_MODELS)[number],
): Promise<ClassifyAlarmResult> {
  const base64Wav = encodePcmToBase64Wav(pcmData, sampleRate, 1);
  const hintLine = !fftHint
    ? ''
    : fftHint === EMERGENCY_CHIRP_HINT
      ? ' Local FFT detected an emergency beep/chirp but cannot reliably separate fire vs smoke (phone speaker smears both bands).' +
        ' Classify ONLY from the audio pattern:' +
        ' FIRE_ALARM = beeps in groups of 3 or 4 (T3/T4 evac), horns, bells, klaxons, or rapid repeated alarm bursts.' +
        ' SMOKE_ALARM = exactly one chirp type with a long pause before the next (typical wall detector, NOT grouped horn codes).' +
        ' Do NOT default to SMOKE_ALARM when beeps are grouped.'
      : ` Local FFT pre-analysis suggests: ${fftHint}. Audio is the primary evidence — override the hint when needed.` +
        ' SMOKE_ALARM = one isolated chirp with long pause (residential detector).' +
        ' FIRE_ALARM = grouped beeps in threes or fours (T3/T4 evac), horns, bells, or klaxons.' +
        ' SIREN = smooth continuous rising/falling vehicle wail.' +
        ' If you hear three or four beeps grouped together, return FIRE_ALARM not SMOKE_ALARM.' +
        ' If you hear a sweeping vehicle siren, return SIREN even if the hint says SMOKE_ALARM.';

  const body = {
    system_instruction: {
      parts: [{ text: SYSTEM_PROMPT }],
    },
    contents: [
      {
        role: 'user',
        parts: [
          {
            inline_data: {
              mime_type: 'audio/wav',
              data: base64Wav,
            },
          },
          { text: USER_PROMPT_BASE + hintLine },
        ],
      },
    ],
    generationConfig: {
      // Gemini 2.5 Flash spends thinking tokens against maxOutputTokens by default,
      // which can leave zero tokens for the JSON reply.
      thinkingConfig: { thinkingBudget: 0 },
      responseMimeType: 'application/json',
      responseSchema: RESPONSE_SCHEMA,
      maxOutputTokens: 256,
      temperature: 0.1,
    },
  };

  let response: Response;
  try {
    response = await fetch(`${geminiEndpoint(model)}?key=${apiKey}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    return { status: 'unavailable', reason: msg };
  }

  if (!response.ok) {
    const errText = await response.text();
    return {
      status: 'unavailable',
      reason: `Audio API ${response.status}: ${errText.slice(0, 200)}`,
    };
  }

  const json = await response.json();
  const finishReason: string | undefined =
    json?.candidates?.[0]?.finishReason ?? json?.candidates?.[0]?.finish_reason;
  const text = extractResponseText(json);
  const result = parseAlarmPayload(text);

  if (!result) {
    const reason =
      finishReason === 'MAX_TOKENS'
        ? 'Gemini response truncated (MAX_TOKENS)'
        : text
          ? `Unparseable Gemini response: ${text.slice(0, 120)}`
          : 'Empty Gemini response';
    return { status: 'unavailable', reason };
  }

  if (result.type === 'FALSE_ALARM') {
    return { status: 'rejected' };
  }

  const alert = normalizeAlert(result);
  if (__DEV__) {
    console.log('[useGeminiAudio] raw:', text, '→', alert.type);
  }

  return { status: 'confirmed', alert };
}

export function useGeminiAudio(apiKey: string): UseGeminiAudioResult {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const classifyAlarm = useCallback(
    async (
      pcmData: Float32Array,
      sampleRate: number,
      fftHint?: string,
    ): Promise<ClassifyAlarmResult> => {
      if (!apiKey) {
        const reason = 'No API key provided';
        setError(reason);
        return { status: 'unavailable', reason };
      }

      setIsAnalyzing(true);
      setError(null);

      try {
        let lastUnavailable: ClassifyAlarmResult | null = null;

        for (let round = 1; round <= MAX_ROUNDS; round += 1) {
          for (const model of GEMINI_MODELS) {
            const result = await requestClassification(
              apiKey,
              pcmData,
              sampleRate,
              fftHint,
              model,
            );
            if (result.status === 'confirmed' || result.status === 'rejected') {
              return result;
            }

            lastUnavailable = result;
            console.warn(
              `[useGeminiAudio] ${model} round ${round}/${MAX_ROUNDS} failed: ${result.reason}`,
            );
            await sleep(350);
          }

          if (round < MAX_ROUNDS) {
            await sleep(1000 * 2 ** (round - 1));
          }
        }

        const reason =
          lastUnavailable?.status === 'unavailable' ? lastUnavailable.reason : 'Unknown error';
        setError(reason);
        return { status: 'unavailable', reason };
      } finally {
        setIsAnalyzing(false);
      }
    },
    [apiKey],
  );

  return { classifyAlarm, isAnalyzing, error };
}
