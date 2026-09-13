import { TOOL_UI } from '$lib/constants';
import type { ToolUiEntry } from '$lib/types';

/**
 * Resolve the UI metadata (label + icon) for a server or browser tool by its
 * name. Falls back to null for unknown tools so callers can render a generic
 * chrome instead.
 */
export function getToolUi(toolName: string | undefined): ToolUiEntry | null {
	if (!toolName) return null;

	return (TOOL_UI as Record<string, ToolUiEntry>)[toolName] ?? null;
}
