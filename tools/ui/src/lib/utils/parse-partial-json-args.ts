// JSON delimiters used while scanning partial streamed JSON. Single-char
// tokens so they only need eq-comparison, but naming them keeps the
// scanner readable and keeps the literal source-of-truth in one place.
const JSON_QUOTE = '"';
const JSON_BACKSLASH = '\\';
const JSON_OBJECT_OPEN = '{';
const JSON_OBJECT_CLOSE = '}';
const JSON_ARRAY_OPEN = '[';
const JSON_ARRAY_CLOSE = ']';

// Trailing punctuation to strip before re-closing a partial object/array.
// Matches an optional trailing comma plus any trailing whitespace; lets
// us re-emit a syntactically-valid JSON document without an orphaned
// comma when the model cut off mid-key.
const TRAILING_JSON_PUNCTUATION_REGEX = /,?\s*$/;

// Parse partial tool-arg JSON streamed token-by-token. Closes any
// unterminated string and dangling open containers (in reverse order),
// so parsers can still surface keys already received while the call
// is still in flight.
export function parsePartialJsonArgs(toolArgsString: string): Record<string, unknown> | null {
	try {
		const parsed: unknown = JSON.parse(toolArgsString);
		if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
			return parsed as Record<string, unknown>;
		}
		return null;
	} catch {
		let inString = false;
		let escape = false;
		const stack: ('{' | '[')[] = [];

		for (let i = 0; i < toolArgsString.length; i++) {
			const ch = toolArgsString[i];
			if (escape) {
				escape = false;
				continue;
			}
			if (ch === JSON_BACKSLASH && inString) {
				escape = true;
				continue;
			}
			if (ch === JSON_QUOTE) {
				inString = !inString;
				continue;
			}
			if (inString) continue;
			if (ch === JSON_OBJECT_OPEN) stack.push(JSON_OBJECT_OPEN);
			else if (ch === JSON_OBJECT_CLOSE) {
				if (stack.length === 0 || stack[stack.length - 1] !== JSON_OBJECT_OPEN) return null;
				stack.pop();
			} else if (ch === JSON_ARRAY_OPEN) stack.push(JSON_ARRAY_OPEN);
			else if (ch === JSON_ARRAY_CLOSE) {
				if (stack.length === 0 || stack[stack.length - 1] !== JSON_ARRAY_OPEN) return null;
				stack.pop();
			}
		}

		let completed = toolArgsString;
		if (escape) {
			// Dangling escape at end of partial JSON: escape the trailing
			// backslash as a literal so we can close the string cleanly.
			completed += JSON_BACKSLASH;
		}
		if (inString) completed += JSON_QUOTE;
		if (!inString) completed = completed.replace(TRAILING_JSON_PUNCTUATION_REGEX, '');

		// Close in reverse nesting order: innermost container first.
		for (let i = stack.length - 1; i >= 0; i--) {
			completed += stack[i] === JSON_OBJECT_OPEN ? JSON_OBJECT_CLOSE : JSON_ARRAY_CLOSE;
		}

		try {
			const parsed: unknown = JSON.parse(completed);
			if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
				return parsed as Record<string, unknown>;
			}
			return null;
		} catch {
			return null;
		}
	}
}
