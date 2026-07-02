import { SvelteSet } from 'svelte/reactivity';
import { RECOMMENDED_MCP_SERVER_IDS } from '$lib/constants';
import type { McpServerOverride } from '$lib/types/database';

/**
 * Derives the set of recommended MCP server IDs the user has opted in to
 * from the given pending overrides. The opted-in state is tracked as
 * conversation-level MCP overrides in the conversations store, so this
 * function simply filters the overrides for recommended server IDs.
 *
 * Callers should pass `conversationsStore.pendingMcpServerOverrides` inside
 * a `$derived` or `$derived.by` to keep the result reactive.
 */
export function getOptedInRecommendationIds(
	pendingOverrides: readonly McpServerOverride[]
): ReadonlySet<string> {
	const result = new SvelteSet<string>();
	for (const override of pendingOverrides) {
		if (RECOMMENDED_MCP_SERVER_IDS.has(override.serverId) && override.enabled) {
			result.add(override.serverId);
		}
	}
	return result;
}
