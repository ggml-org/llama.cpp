import type { ToolOverride } from '$lib/types/database';

/**
 * Parses a JSON-serialized list of tool overrides from settings config.
 * Invalid input yields an empty list, entries missing {key, enabled} are dropped.
 */
export function parseToolOverrides(raw: unknown): ToolOverride[] {
	if (typeof raw !== 'string' || raw.length === 0) return [];

	try {
		const parsed = JSON.parse(raw);
		if (!Array.isArray(parsed)) return [];
		return parsed.filter(
			(o: unknown) => typeof o === 'object' && o !== null && 'key' in o && 'enabled' in o
		) as ToolOverride[];
	} catch {
		return [];
	}
}
