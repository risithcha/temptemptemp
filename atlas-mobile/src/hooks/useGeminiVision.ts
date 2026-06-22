/**
 * useGeminiVision – cloud vision backend that feeds the same Detection[]
 * shape and TTS announcer as the on-device MobileNet pipeline.
 */
import { useEffect, useRef, useCallback, useState } from 'react';
import type { Camera } from 'react-native-vision-camera';
import * as FileSystem from 'expo-file-system/legacy';

import type { BoundingBox, Detection } from '../utils/tensor_decoder';
import {
  VISION_ALLOWED_CLASSES,
  VISION_CLASSES_PROMPT_LIST,
  resolveToVisionLabel,
  visionClassIdForLabel,
} from '../utils/tensor_decoder';
import { useDetectionAnnouncement } from '../utils/vision_announce';
import { sanitizeOcrText } from '../utils/ocr_utils';
import { INFERENCE_FPS } from '../theme';
import { isLikelyNetworkError } from './useNetworkReachable';

const CAPTURE_INTERVAL_MS = 4000;

const GEMINI_ENDPOINT =
  'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent';

const SYSTEM_PROMPT =
  'You are an object detector and text reader for a blind user\'s camera app. ' +
  'Reply with ONLY valid JSON (no markdown, no prose): ' +
  '{"objects":[{"label":"person","direction":"left"}],"text":"readable text in scene"} ' +
  'Rules: label MUST be exactly one of these allowed class names (lowercase): ' +
  VISION_CLASSES_PROMPT_LIST +
  '. Includes standard COCO objects plus interview-room items (coffee cup, microphone, whiteboard, etc.). ' +
  'direction = exactly "left", "center", or "right"; at most 3 prominent objects. ' +
  'text = all clearly readable text in the image (signs, labels, screens, documents), ' +
  'preserving line breaks; use "" if none. ' +
  'If nothing from the object list is clearly visible return {"objects":[],"text":"..."}. ' +
  'Never use object labels outside the allowed list (no "door", "window", "sky"). ' +
  'For electronics use generic labels only: phone, tablet, laptop, computer — never brand names (no iPhone, iPad, MacBook, Galaxy, etc.). ' +
  'Never explain the image or add prose outside the JSON.';

export interface UseGeminiVisionOptions {
  cameraRef: React.RefObject<Camera | null>;
  apiKey: string;
  isActive: boolean;
  ttsRate?: number;
  /** Called when a vision API request fails due to connectivity. */
  onConnectivityFailure?: () => void;
}

export interface UseGeminiVisionResult {
  detections: Detection[];
  /** Readable text extracted from the same camera snapshot. */
  ocrText: string;
  isPending: boolean;
  error: string | null;
  /** Display FPS mimicking the local 5 inf/s pipeline. */
  fps: number;
}

const USER_PROMPT =
  'From this image: (1) identify up to 3 visible objects using ONLY labels from the allowed list; ' +
  '(2) transcribe all clearly readable text. ' +
  'Reply ONLY with JSON like {"objects":[{"label":"person","direction":"left"}],"text":"EXIT"} . ' +
  'Use direction: left, center, or right. If no allowed objects, {"objects":[]}. ' +
  'Always include "text" (use "" when no readable text). ' +
  'Use generic device names (phone, tablet, laptop, computer) — not brands or models.';

/** Longest labels first so prose fallback matches "coffee cup" before "cup". */
const VISION_LABELS_LONGEST_FIRST = [...VISION_ALLOWED_CLASSES].sort(
  (a, b) => b.length - a.length,
);

type ParsedObject = {
  label?: string;
  name?: string;
  class?: string;
  object?: string;
  direction?: string;
  position?: string;
  location?: string;
  side?: string;
};

function boxForDirection(direction: string): BoundingBox {
  switch (direction) {
    case 'left':
      return { left: 0.05, right: 0.25, top: 0.3, bottom: 0.7 };
    case 'right':
      return { left: 0.75, right: 0.95, top: 0.3, bottom: 0.7 };
    default:
      return { left: 0.35, right: 0.65, top: 0.3, bottom: 0.7 };
  }
}

function normalizeDirection(raw: string | undefined): 'left' | 'center' | 'right' {
  const d = (raw ?? 'center').toLowerCase().trim();
  if (d.startsWith('left')) return 'left';
  if (d.startsWith('right')) return 'right';
  return 'center';
}

function objectLabel(obj: ParsedObject): string | null {
  const raw = obj.label ?? obj.name ?? obj.class ?? obj.object;
  if (typeof raw !== 'string') return null;
  const label = raw.trim().toLowerCase();
  return label.length > 0 ? label : null;
}

function objectDirection(obj: ParsedObject): 'left' | 'center' | 'right' {
  const raw =
    obj.direction ?? obj.position ?? obj.location ?? obj.side ?? 'center';
  return normalizeDirection(raw);
}

function objectsFromParsed(parsed: unknown): ParsedObject[] | null {
  if (Array.isArray(parsed)) {
    return parsed as ParsedObject[];
  }
  if (parsed && typeof parsed === 'object') {
    const record = parsed as Record<string, unknown>;
    for (const key of ['objects', 'detections', 'items', 'results']) {
      const value = record[key];
      if (Array.isArray(value)) {
        return value as ParsedObject[];
      }
    }
  }
  return null;
}

function extractOcrText(parsed: unknown): string {
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return '';
  }
  const record = parsed as Record<string, unknown>;
  for (const key of ['text', 'ocr', 'ocr_text', 'reading', 'words']) {
    const value = record[key];
    if (typeof value === 'string') {
      return sanitizeOcrText(value);
    }
  }
  return '';
}

function parseJsonPayload(text: string): { objects: ParsedObject[]; ocrText: string } | null {
  const trimmed = text.trim();
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidates = [trimmed, fenced?.[1]?.trim()].filter(Boolean) as string[];

  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate);
      const objects = objectsFromParsed(parsed);
      if (objects) {
        return { objects, ocrText: extractOcrText(parsed) };
      }
    } catch {
      // try substring object next
    }

    const jsonMatch = candidate.match(/\{[\s\S]*\}/);
    if (!jsonMatch) continue;

    try {
      const parsed = JSON.parse(jsonMatch[0]);
      const objects = objectsFromParsed(parsed);
      if (objects) {
        return { objects, ocrText: extractOcrText(parsed) };
      }
    } catch {
      // continue
    }
  }

  return null;
}

function isEmptySceneText(text: string): boolean {
  const lower = text.toLowerCase();
  return (
    lower.includes('nothing') ||
    lower.includes('no object') ||
    lower.includes('no visible') ||
    lower.includes('cannot see') ||
    lower.includes("can't see") ||
    lower.includes('empty scene')
  );
}

function parseProseFallback(text: string): Detection[] {
  if (isEmptySceneText(text)) return [];

  const lower = text.toLowerCase();
  let direction: 'left' | 'center' | 'right' = 'center';
  if (/\bleft\b/.test(lower)) direction = 'left';
  else if (/\bright\b/.test(lower)) direction = 'right';

  const found: Detection[] = [];
  const seen = new Set<string>();
  for (const label of VISION_LABELS_LONGEST_FIRST) {
    if (!lower.includes(label) || seen.has(label)) continue;
    seen.add(label);
    found.push({
      label,
      classId: visionClassIdForLabel(label),
      score: 0.85,
      box: boxForDirection(direction),
    });
    if (found.length >= 3) break;
  }
  return found;
}

function toDetections(objects: ParsedObject[]): Detection[] {
  const seen = new Set<string>();
  const results: Detection[] = [];

  for (const obj of objects) {
    const raw = objectLabel(obj);
    if (!raw) continue;
    const label = resolveToVisionLabel(raw);
    if (!label || seen.has(label)) continue;
    seen.add(label);

    const classId = visionClassIdForLabel(label);
    if (classId < 0) continue;

    results.push({
      label,
      classId,
      score: 0.9,
      box: boxForDirection(objectDirection(obj)),
    });
    if (results.length >= 3) break;
  }

  return results;
}

export function parseVisionObjects(text: string): Detection[] {
  return parseVisionResponse(text).detections;
}

export function parseVisionResponse(text: string): {
  detections: Detection[];
  ocrText: string;
} {
  if (!text.trim()) return { detections: [], ocrText: '' };

  const fromJson = parseJsonPayload(text);
  if (fromJson) {
    return {
      detections: toDetections(fromJson.objects),
      ocrText: fromJson.ocrText,
    };
  }

  return { detections: parseProseFallback(text), ocrText: '' };
}

export function useGeminiVision({
  cameraRef,
  apiKey,
  isActive,
  ttsRate = 0.95,
  onConnectivityFailure,
}: UseGeminiVisionOptions): UseGeminiVisionResult {
  const [detections, setDetections] = useState<Detection[]>([]);
  const [ocrText, setOcrText] = useState('');
  const [isPending, setIsPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fps, setFps] = useState(0);
  const isPendingRef = useRef(false);

  // Mimic local-model inference rate (~5 inf/s with slight jitter).
  useEffect(() => {
    if (!isActive) {
      setFps(0);
      return;
    }

    setFps(INFERENCE_FPS);
    const id = setInterval(() => {
      setFps(INFERENCE_FPS + Math.floor(Math.random() * 3) - 1);
    }, 900);
    return () => clearInterval(id);
  }, [isActive]);

  const runOnce = useCallback(async () => {
    const cam = cameraRef.current;
    if (!cam || isPendingRef.current || !apiKey) return;

    isPendingRef.current = true;
    setIsPending(true);

    try {
      const snapshot = await cam.takeSnapshot({ quality: 60 });
      const rawPath = snapshot.path;
      const filePath = rawPath.startsWith('file://') ? rawPath : `file://${rawPath}`;

      const base64 = await FileSystem.readAsStringAsync(filePath, {
        encoding: FileSystem.EncodingType.Base64,
      });

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
                  mime_type: 'image/jpeg',
                  data: base64,
                },
              },
              { text: USER_PROMPT },
            ],
          },
        ],
        generationConfig: {
          maxOutputTokens: 768,
          temperature: 0.1,
        },
      };

      const response = await fetch(`${GEMINI_ENDPOINT}?key=${apiKey}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(`Vision ${response.status}: ${errText.slice(0, 200)}`);
      }

      const json = await response.json();
      const text: string =
        json?.candidates?.[0]?.content?.parts?.[0]?.text?.trim() ?? '';

      const parsed = parseVisionResponse(text);
      if (__DEV__ && text && parsed.detections.length === 0 && !parsed.ocrText) {
        console.warn('[useGeminiVision] unparsed response:', text.slice(0, 200));
      }

      setDetections(parsed.detections);
      setOcrText(parsed.ocrText);
      setError(null);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      console.warn('[useGeminiVision]', msg);
      setError(msg);
      if (isLikelyNetworkError(msg)) {
        onConnectivityFailure?.();
      }
    } finally {
      isPendingRef.current = false;
      setIsPending(false);
    }
  }, [cameraRef, apiKey, onConnectivityFailure]);

  useEffect(() => {
    if (!isActive) {
      setDetections([]);
      setOcrText('');
      return;
    }

    runOnce();
    const id = setInterval(runOnce, CAPTURE_INTERVAL_MS);
    return () => clearInterval(id);
  }, [isActive, runOnce]);

  useDetectionAnnouncement(detections, isActive, { ttsRate });

  return { detections, ocrText, isPending, error, fps };
}
