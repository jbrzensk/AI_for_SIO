// A token can genuinely be whitespace-only (a plain newline is a common
// real prediction from a small model). Rendered as-is, a DOM element's
// text content would be empty/invisible. This function only affects
// DISPLAY — callers must keep using the real, unmodified token text for
// tokenization, decoding, and anywhere the actual value matters.
export function visibleLabel(text) {
  if (text === '') return '(empty)';
  if (/^\s+$/.test(text)) {
    return text.replace(/\n/g, '⏎').replace(/\t/g, '→').replace(/ /g, '·');
  }
  return text;
}
