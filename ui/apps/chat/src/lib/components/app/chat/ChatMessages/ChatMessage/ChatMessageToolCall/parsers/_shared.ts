// Helpers shared by the per-tool meta parsers under
// `src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/`.
// Each tool needs the same first three steps (tool-name check,
// args-present check, JSON parse), plus the same path aliases - keeping
// them here lets each parser stay focused on its own format quirks.

import { parsePartialJsonArgs } from '@ggml-org/llama-harness';
import { TOOL_ARG_PATH_KEYS } from '#lib/constants/index.ts';
import { BuiltInTool } from '#lib/enums/index.ts';
import type { AgenticSection } from '#lib/types/agentic.js';

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
 * Read the path argument out of an already parsed args object, checking
 * the aliases in `TOOL_ARG_PATH_KEYS`. Returns undefined when the call
 * carries no usable path - the path is display-only, so callers must
 * still render whatever else the call streamed.
 */
export function extractToolArgPath(args?: Record<string, unknown> | null): string | undefined {
	if (!args) return undefined;

	for (const key of TOOL_ARG_PATH_KEYS) {
		const value = args[key];

		if (typeof value === 'string' && value) return value;
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
