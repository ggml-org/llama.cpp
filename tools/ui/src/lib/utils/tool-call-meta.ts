// Generic helper for parsing tool-result blobs (the "out" side of a
// tool call). Used by the per-tool meta parsers under
// `src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/`.
// Each tool needs to surface fields like `error`, `result`, `bytes`,
// `edits_applied` without repeating the try/JSON.parse/object guard inline.

import { JSON_OBJECT_OPEN } from '$lib/constants';

/**
 * Parse a tool-result blob into a JSON object, or `null` if it isn't
 * one. Returns null for:
 *   - missing / empty input,
 *   - a JSON object that turns out to be an array or primitive,
 *   - any parse failure (always returns null rather than throwing).
 */
export function tryParseToolResultObject(
	toolResultString: string | undefined
): Record<string, unknown> | null {
	if (!toolResultString) return null;

	// Tool results are usually large plain text (file contents, stdout); only
	// a JSON object root can carry fields, so skip the parse otherwise
	const trimmed = toolResultString.trimStart();

	if (trimmed[0] !== JSON_OBJECT_OPEN) return null;

	try {
		const parsed: unknown = JSON.parse(trimmed);

		if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
			return parsed as Record<string, unknown>;
		}

		return null;
	} catch {
		return null;
	}
}
