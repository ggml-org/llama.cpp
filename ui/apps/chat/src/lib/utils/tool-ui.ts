import { TOOL_UI } from '#lib/constants/index.ts';
import { ToolSource } from '#lib/enums/tools.enums.ts';
import type { OpenAIToolDefinition, ToolEntry, ToolUiEntry } from '#lib/types/index.ts';

/**
 * Resolve the UI metadata (label + icon) for a server or browser tool by its
 * name. Falls back to null for unknown tools so callers can render a generic
 * chrome instead.
 */
export function getToolUi(toolName: string | undefined): ToolUiEntry | null {
	if (!toolName) return null;

	return (TOOL_UI as Record<string, ToolUiEntry>)[toolName] ?? null;
}

/**
 * The name and icon a tool is listed with: built-in tools have friendlier ones than their
 * function names, the others show the function name.
 */
export function getToolDisplay(entry: ToolEntry): {
	icon: ToolUiEntry['icon'] | null;
	label: string;
} {
	const name = entry.definition.function.name;
	const ui =
		entry.source === ToolSource.SERVER || entry.source === ToolSource.BROWSER
			? getToolUi(name)
			: null;

	return { icon: ui?.icon ?? null, label: ui?.label ?? name };
}

/** The names of the arguments a tool takes. */
export function getToolArgNames(definition: OpenAIToolDefinition): string[] {
	const properties = (definition.function.parameters as { properties?: object } | undefined)
		?.properties;

	return properties ? Object.keys(properties) : [];
}
