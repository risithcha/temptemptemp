/**
 * Atlas Mobile - OCR Text Sanitization
 *
 * Cleans raw ML Kit output before display: strips stray characters,
 * collapses whitespace, and filters out gibberish-only lines.
 */

const JUNK_CHARS = /[^\p{L}\p{N}\p{P}\p{Z}\n]/gu;
const MULTI_SPACE = /[ \t]{2,}/g;
const MULTI_NEWLINE = /\n{3,}/g;

/**
 * A line is considered gibberish if it contains zero alphanumeric characters
 * or is entirely whitespace. Intentionally permissive so short labels
 * ("OK", "5G", etc.) are preserved.
 */
function isGibberish(line: string): boolean {
  const trimmed = line.trim();
  if (trimmed.length === 0) return true;
  const alphanumeric = trimmed.replace(/[^a-zA-Z0-9]/g, '').length;
  return alphanumeric === 0;
}

export function sanitizeOcrText(raw: string): string {
  if (!raw) return '';

  let text = raw
    .replace(JUNK_CHARS, '')
    .replace(MULTI_SPACE, ' ')
    .replace(MULTI_NEWLINE, '\n\n');

  const lines = text.split('\n');
  const cleaned = lines.filter((l) => !isGibberish(l));
  text = cleaned.join('\n').trim();

  return text;
}
