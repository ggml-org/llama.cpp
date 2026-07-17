// Helpers shared by the per-tool meta parsers under
// `src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/`.
// Each tool needs the same first three steps (tool-name check,
// args-present check, JSON parse) - keeping them here lets each parser
// stay focused on its own format quirks.

import { BuiltInTool } from '$lib/enums';
import { parsePartialJsonArgs } from '$lib/utils/parse-partial-json-args';
import type { AgenticSection } from '$lib/utils/agentic';

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
