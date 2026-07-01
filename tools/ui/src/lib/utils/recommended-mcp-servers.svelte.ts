import { browser } from '$app/environment';
import { SvelteSet } from 'svelte/reactivity';
import {
	MCP_RECOMMENDATIONS_OPTED_IN_LOCALSTORAGE_KEY,
	RECOMMENDED_MCP_SERVER_IDS
} from '$lib/constants';

/**
 * Predefined MCP server IDs the user accepted in the suggestions dialog.
 *
 * Source of truth for whether a recommendation's auto-seeded entry in the
 * MCP servers setting should be surfaced in /mcp-servers alongside any
 * custom servers the user has added. The dialog itself persists writes here
 * whenever the user clicks "Enable selected"; consumers (e.g. the settings
 * page) read the set and filter out unaccepted recommendations.
 */
const optedInRecommendationIds = new SvelteSet<string>(
	parseOptedInRecommendationIds(
		browser ? localStorage.getItem(MCP_RECOMMENDATIONS_OPTED_IN_LOCALSTORAGE_KEY) : null
	)
);

/**
 * Parses the raw localStorage payload holding the opted-in recommendation
 * IDs. Defensive against missing, malformed and improperly-typed values —
 * callers can rely on the return value being a plain `string[]`.
 *
 * Exported so the parsing rules can be exercised independently of the
 * module-level reactive {@link SvelteSet} state.
 */
export function parseOptedInRecommendationIds(raw: string | null): string[] {
	if (!raw) return [];

	try {
		const parsed = JSON.parse(raw);
		if (!Array.isArray(parsed)) return [];

		return parsed.filter((id): id is string => typeof id === 'string');
	} catch {
		return [];
	}
}

/**
 * Returns a reactive, read-only view of the opted-in recommendation IDs.
 * Reading inside a `$derived` triggers re-evaluation when the dialog writes
 * new IDs via {@link addOptedInRecommendationIds}.
 */
export function getOptedInRecommendationIds(): ReadonlySet<string> {
	return optedInRecommendationIds;
}

/**
 * Persist that the user has opted in to the given recommendation IDs.
 *
 * IDs not present in {@link RECOMMENDED_MCP_SERVER_IDS} are dropped so a
 * tampered localStorage value cannot smuggle unrelated server IDs into the
 * opted-in set. Existing IDs are preserved.
 */
export function addOptedInRecommendationIds(ids: Iterable<string>): void {
	if (!browser) return;

	let changed = false;
	for (const id of ids) {
		if (typeof id !== 'string' || !RECOMMENDED_MCP_SERVER_IDS.has(id)) continue;
		if (optedInRecommendationIds.has(id)) continue;
		optedInRecommendationIds.add(id);
		changed = true;
	}

	if (!changed) return;

	localStorage.setItem(
		MCP_RECOMMENDATIONS_OPTED_IN_LOCALSTORAGE_KEY,
		JSON.stringify([...optedInRecommendationIds])
	);
}

export function isOptedInRecommendation(id: string): boolean {
	return optedInRecommendationIds.has(id);
}
