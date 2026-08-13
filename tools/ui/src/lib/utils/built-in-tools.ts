import { BUILTIN_TOOL_UI } from '$lib/constants';
import type { BuiltinToolUiEntry } from '$lib/types';

/**
 * Resolve the UI metadata (label + icon) for a built-in tool by its name.
 * Falls back to null for unknown or non-built-in tools so callers can render
 * a generic chrome instead.
 */
export function getBuiltinToolUi(toolName: string | undefined): BuiltinToolUiEntry | null {
	if (!toolName) return null;

	return (BUILTIN_TOOL_UI as Record<string, BuiltinToolUiEntry>)[toolName] ?? null;
}
