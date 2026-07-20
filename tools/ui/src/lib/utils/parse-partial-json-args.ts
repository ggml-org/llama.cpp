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

/** Bounded cache for parsePartialJsonArgs results. */
const PARTIAL_JSON_CACHE_MAX_SIZE = 32;
const partialJsonCache = new Map<string, Record<string, unknown> | null>();

function cacheResult(input: string, result: Record<string, unknown> | null): void {
	if (partialJsonCache.size >= PARTIAL_JSON_CACHE_MAX_SIZE) {
		partialJsonCache.delete(partialJsonCache.keys().next().value!);
	}
	partialJsonCache.set(input, result);
}

/**
 * Re-close `toolArgsString` after discarding the final, still-incomplete member.
 *
 * Used when the normal closure produced invalid JSON because the tail is a bare
 * key (`,"start_l`) or a key with no value yet (`,"start_line":`). Truncating at
 * the last member separator that sits outside a string keeps every member that
 * did arrive intact. Returns null when there is nothing to salvage.
 */
function dropTrailingPartialMember(toolArgsString: string, stack: ('{' | '[')[]): string | null {
	let inString = false;
	let escape = false;
	let cutIndex = -1;
	const open: ('{' | '[')[] = [];
	// Containers still open at `cutIndex`. Closing the caller's final stack
	// instead would emit closers for containers that the truncation removed,
	// producing something like `{"path":"x"}]}`.
	let openAtCut: ('{' | '[')[] = [];

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

		if (ch === JSON_OBJECT_OPEN || ch === JSON_ARRAY_OPEN) open.push(ch);
		else if (ch === JSON_OBJECT_CLOSE || ch === JSON_ARRAY_CLOSE) open.pop();
		// Only separators in the outermost object delimit the members we keep.
		else if (ch === ',' && open.length === 1) {
			cutIndex = i;
			openAtCut = [...open];
		}
	}

	if (cutIndex === -1) {
		// No complete member at all - the object is still just `{"pa`.
		return stack.length > 0 && stack[0] === JSON_OBJECT_OPEN ? '{}' : null;
	}

	let completed = toolArgsString.slice(0, cutIndex);

	for (let i = openAtCut.length - 1; i >= 0; i--) {
		completed += openAtCut[i] === JSON_OBJECT_OPEN ? JSON_OBJECT_CLOSE : JSON_ARRAY_CLOSE;
	}

	return completed;
}

// Parse partial tool-arg JSON streamed token-by-token. Closes any
// unterminated string and dangling open containers (in reverse order),
// so parsers can still surface keys already received while the call
// is still in flight. Memoized: the char-by-char scanner runs on every
// render during streaming even when toolArgs hasn't changed.
export function parsePartialJsonArgs(toolArgsString: string): Record<string, unknown> | null {
	const cached = partialJsonCache.get(toolArgsString);
	if (cached !== undefined) return cached;

	let result: Record<string, unknown> | null;

	try {
		const parsed: unknown = JSON.parse(toolArgsString);
		result =
			parsed && typeof parsed === 'object' && !Array.isArray(parsed)
				? (parsed as Record<string, unknown>)
				: null;
	} catch {
		result = scanPartialJson(toolArgsString);
	}

	cacheResult(toolArgsString, result);
	return result;
}

/** Char-by-char scanner for unterminated partial JSON. */
function scanPartialJson(toolArgsString: string): Record<string, unknown> | null {
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
		// Dangling escape at the end of partial JSON: the sequence is half of
		// an escape (`\` of a `\n`), so drop it. Keeping it as a literal
		// backslash flashes a stray character on screen that turns into a
		// newline once the next token lands.
		completed = completed.slice(0, -1);
	}
	if (inString) completed += JSON_QUOTE;
	if (!inString) completed = completed.replace(TRAILING_JSON_PUNCTUATION_REGEX, '');

	// Close in reverse nesting order: innermost container first.
	for (let i = stack.length - 1; i >= 0; i--) {
		completed += stack[i] === JSON_OBJECT_OPEN ? JSON_OBJECT_CLOSE : JSON_ARRAY_CLOSE;
	}

	try {
		const parsed: unknown = JSON.parse(completed);
		return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
			? (parsed as Record<string, unknown>)
			: null;
	} catch {
		// A member that has arrived as a bare key so far (`,"start_l`, or
		// `,"start_line":` with no value yet) closes into `{"a":1,"start_l"}`,
		// which is invalid - so the whole object would vanish for as many
		// frames as the key name takes to stream, blanking the UI between
		// parameters. Drop that trailing partial member and keep the members
		// already complete.
		const withoutTrailingMember = dropTrailingPartialMember(toolArgsString, stack);

		if (withoutTrailingMember !== null) {
			try {
				const parsed: unknown = JSON.parse(withoutTrailingMember);
				if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
					return parsed as Record<string, unknown>;
				}
			} catch {
				return null;
			}
		}

		return null;
	}
}

/**
 * Name of the argument whose value may still be mid-flight, or `null` when the
 * blob is already valid JSON and every value is final.
 *
 * Object keys arrive in order, so only the last one present can be truncated.
 * Callers use this to avoid deriving stable display values (a basename, a
 * highlight language) from a half-streamed string - doing so makes headers walk
 * through every path segment and makes the highlighter swap grammars mid-render.
 */
export function truncatedArgKey(toolArgsString: string): string | null {
	try {
		JSON.parse(toolArgsString);

		return null;
	} catch {
		const partial = parsePartialJsonArgs(toolArgsString);
		if (!partial) return null;

		const keys = Object.keys(partial);

		return keys.length > 0 ? keys[keys.length - 1] : null;
	}
}
