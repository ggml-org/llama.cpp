// Generic helper for parsing tool-result blobs (the "out" side of a
// tool call). Used by the per-tool meta parsers under
// `src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/`.
// Each tool needs to surface fields like `error`, `result`, `bytes`,
// `edits_applied` without repeating the try/JSON.parse/object guard inline.

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
	try {
		const parsed: unknown = JSON.parse(toolResultString);
		if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
			return parsed as Record<string, unknown>;
		}
		return null;
	} catch {
		return null;
	}
}
