import { SETTINGS_KEYS, TOOL_CALL_SHOWN_ARG_NONE } from '#lib/constants/index.ts';
import type { SettingsConfigType } from '#lib/types/index.ts';

/** The per-tool choices, by tool name; an unreadable value counts as none. */
export function parseToolCallShownArgOverrides(raw: unknown): Record<string, string> {
	if (typeof raw !== 'string' || !raw.trim()) return {};

	try {
		const parsed: unknown = JSON.parse(raw);

		if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return {};

		return Object.fromEntries(
			Object.entries(parsed).filter(
				(entry): entry is [string, string] => typeof entry[1] === 'string'
			)
		);
	} catch {
		return {};
	}
}

/**
 * The arg a tool's calls show in their line: the tool's own choice when it
 * has one, else the global one; null when the line is the plain sentence.
 */
export function toolCallShownArg(
	config: Partial<SettingsConfigType>,
	toolName: string | undefined
): string | null {
	const overrides = parseToolCallShownArgOverrides(
		config[SETTINGS_KEYS.TOOL_CALL_SHOWN_ARG_OVERRIDES]
	);
	const own = toolName !== undefined ? overrides[toolName] : undefined;
	const global = config[SETTINGS_KEYS.TOOL_CALL_SHOWN_ARG];
	const arg = own ?? (typeof global === 'string' ? global : TOOL_CALL_SHOWN_ARG_NONE);

	return arg.trim() || null;
}
