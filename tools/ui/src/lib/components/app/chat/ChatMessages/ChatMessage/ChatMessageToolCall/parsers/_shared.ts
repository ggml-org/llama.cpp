// Helpers shared by the per-tool meta parsers under
// `src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/`.
// Each tool needs the same first three steps (tool-name check,
// args-present check, JSON parse) - keeping them here lets each parser
// stay focused on its own format quirks.

import { TOOL_ARG_STRING_FIELD_PATTERN_TEMPLATE } from '$lib/constants';
import { BuiltInTool } from '$lib/enums';
import type { AgenticSection } from '$lib/types/agentic';
import { parsePartialJsonArgs } from '$lib/utils/parse-partial-json-args';

/**
 * Strict (final-state) JSON parser for a tool-args blob. Mirrors the
 * behaviour the per-tool components used before extraction: an
 * invalid JSON blob, a JSON array, or a JSON primitive all map to
 * `null` so callers don't have to guard against surprise shapes.
 */
function parseFinalToolArgs(blob: string): Record<string, unknown> | null {
	try {
		const parsed: unknown = JSON.parse(blob);

		if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
			return parsed as Record<string, unknown>;
		}

		return null;
	} catch {
		return null;
	}
}

// Compiled per key on first use; the key set is tiny and fixed.
const toolArgStringRegexes = new Map<string, RegExp>();

/**
 * Extract a string field from a JSON tool-args blob without parsing the
 * whole document. write_file and edit_file args embed full file contents,
 * yet the block title needs only the path; a targeted key match plus a
 * JSON.parse of the captured string literal alone keeps title rendering
 * O(path) instead of O(blob). Returns undefined when the key is missing
 * or its value is not a string; callers fall back to the full parse.
 */
export function extractToolArgString(
	toolArgs: string,
	keys: readonly string[]
): string | undefined {
	for (const key of keys) {
		let pattern = toolArgStringRegexes.get(key);

		if (!pattern) {
			pattern = new RegExp(TOOL_ARG_STRING_FIELD_PATTERN_TEMPLATE.replace('{key}', key));
			toolArgStringRegexes.set(key, pattern);
		}

		const match = pattern.exec(toolArgs);

		if (!match) continue;

		try {
			const value: unknown = JSON.parse(`"${match[1]}"`);

			if (typeof value === 'string') return value;
		} catch {
			// fall through to the next key; the full parse is the fallback
		}
	}

	return undefined;
}

/**
 * Parse a section's toolArgs against an expected tool name. Returns
 * `null` when:
 *   - the section's toolName doesn't match (component isn't for this
 *     tool);
 *   - the section has no args yet (call hasn't started streaming);
 *   - or the args blob can't be parsed.
 *
 * Pass `{ partial: true }` for tools that need to render incrementally
 * as each token lands (read_file, edit_file, write_file).
 */
export function parseToolArgs(
	expected: BuiltInTool,
	section: AgenticSection,
	options: { partial?: boolean } = {}
): Record<string, unknown> | null {
	if (section.toolName !== expected || !section.toolArgs) return null;

	return options.partial
		? parsePartialJsonArgs(section.toolArgs)
		: parseFinalToolArgs(section.toolArgs);
}
