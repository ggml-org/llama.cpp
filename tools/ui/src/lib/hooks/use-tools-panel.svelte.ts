import { CLI_FLAGS } from '$lib/constants';
import { ToolSource } from '$lib/enums';
import { conversationsStore, mcpStore, toolsStore } from '$lib/stores';
import type { ToolEntry, ToolGroup } from '$lib/types';
import { SvelteSet } from 'svelte/reactivity';

export interface UseToolsPanelReturn {
	readonly expandedGroups: SvelteSet<string>;
	readonly categoryGroups: ToolGroup[];
	readonly mcpGroups: ToolGroup[];
	readonly totalToolCount: number;
	readonly noToolsInfoMessage: string | null;
	isGroupChecked(group: ToolGroup): boolean;
	getEnabledToolCount(group: ToolGroup): number;
	getGroupCheckState(group: ToolGroup): { checked: boolean; indeterminate: boolean };
	getFavicon(group: ToolGroup): string | null;
	isGroupDisabled(group: ToolGroup): boolean;
	isToolEnabled(entry: ToolEntry): boolean;
	isToolParentDisabled(entry: ToolEntry): boolean;
	toggleTool(entry: ToolEntry): void;
	toggleGroupExpanded(key: string): void;
	/** Toggle all tools in a group by its stable key (avoids stale group object references). */
	toggleGroupByKey(key: string): void;
	handleOpen(): void;
}

/**
 * Shared reactive state and helpers for the tools panel UI.
 *
 * Used by both the desktop dropdown (`ChatFormActionAddToolsSubmenu`)
 * and the mobile sheet (`ChatFormActionAddSheet`) to avoid
 * duplicating group filtering, checked-state derivation, and favicon logic.
 *
 * All toggle state routes through `conversationsStore.preferences`: with an
 * active conversation it edits that conversation's tool policy, on the
 * new-chat screen it edits the global defaults seeded into new conversations.
 */
export function useToolsPanel(): UseToolsPanelReturn {
	const expandedGroups = new SvelteSet<string>();
	const groups = $derived(toolsStore.toolGroups);
	// non-MCP groups are 1:1 with tool categories; MCP tools group per server
	const categoryGroups = $derived(groups.filter((g) => g.source !== ToolSource.MCP));
	const mcpGroups = $derived(groups.filter((g) => g.source === ToolSource.MCP));
	const totalToolCount = $derived(groups.reduce((n, g) => n + g.tools.length, 0));
	const noToolsInfoMessage = $derived.by(() => {
		if (toolsStore.loading) return null;

		if (toolsStore.toolGroups.length > 0) return null;

		// Tools endpoint is unreachable (404) — server started without --tools
		if (toolsStore.isToolsEndpointUnreachable) {
			return `To enable Server Tools you need to run llama-server with ${CLI_FLAGS.TOOLS} all or ${CLI_FLAGS.TOOLS} <name> flag. To see MCP Tools you need to add / enable MCP Server(s).`;
		}

		// Other errors — return null so UI shows "Failed to load tools"
		if (toolsStore.error) return null;

		return `To enable Server Tools you need to run llama-server with ${CLI_FLAGS.TOOLS} all or ${CLI_FLAGS.TOOLS} <name> flag. To see MCP Tools you need to add / enable MCP Server(s).`;
	});

	function isGroupChecked(group: ToolGroup): boolean {
		return conversationsStore.preferences.isGroupChecked(group);
	}

	function getEnabledToolCount(group: ToolGroup): number {
		return group.tools.filter((tool) => conversationsStore.preferences.isToolActive(tool)).length;
	}

	/**
	 * Group checkbox state: checked is the parent flag (category on, or the
	 * server key on for MCP groups); indeterminate marks the mixed case where
	 * the parent is on but nothing or only part of the group is enabled.
	 * isToolActive folds the parent gates into the count, so a disabled parent
	 * always yields plain unchecked.
	 */
	function getGroupCheckState(group: ToolGroup): { checked: boolean; indeterminate: boolean } {
		const checked = isGroupChecked(group);
		const enabledCount = getEnabledToolCount(group);
		const indeterminate =
			group.tools.length > 0 && (enabledCount === 0 ? checked : enabledCount < group.tools.length);

		return { checked, indeterminate };
	}

	function getFavicon(group: ToolGroup): string | null {
		if (group.source !== ToolSource.MCP || !group.serverId) return null;

		return mcpStore.getServerFavicon(group.serverId);
	}

	function isGroupDisabled(group: ToolGroup): boolean {
		// MCP server groups gray out while the whole MCP category is off
		return (
			group.source === ToolSource.MCP &&
			!conversationsStore.preferences.isCategoryEnabled(ToolSource.MCP)
		);
	}

	function isToolEnabled(entry: ToolEntry): boolean {
		return conversationsStore.preferences.isToolEnabled(entry.key);
	}

	function isToolParentDisabled(entry: ToolEntry): boolean {
		return conversationsStore.preferences.isToolParentDisabled(entry);
	}

	function toggleTool(entry: ToolEntry): void {
		void conversationsStore.preferences.toggleTool(entry.key);
	}

	function toggleGroupExpanded(key: string): void {
		if (expandedGroups.has(key)) {
			expandedGroups.delete(key);
		} else {
			expandedGroups.add(key);
		}
	}

	function toggleGroupByKey(key: string): void {
		// Find current group by key to get up-to-date tool references
		const group = groups.find((g) => g.key === key);

		if (!group) return;

		void conversationsStore.preferences.toggleGroup(group);
	}

	function handleOpen(): void {
		if (toolsStore.serverTools.length === 0 && !toolsStore.loading) {
			toolsStore.fetchServerTools();
		}

		mcpStore.runHealthChecksForServers(mcpStore.getServers().filter((s) => s.enabled));
	}

	return {
		get categoryGroups() {
			return categoryGroups;
		},
		expandedGroups,
		getEnabledToolCount,
		getFavicon,
		getGroupCheckState,
		handleOpen,
		isGroupChecked,
		isGroupDisabled,
		isToolEnabled,
		isToolParentDisabled,
		get mcpGroups() {
			return mcpGroups;
		},
		get noToolsInfoMessage() {
			return noToolsInfoMessage;
		},
		toggleGroupByKey,
		toggleGroupExpanded,
		toggleTool,
		get totalToolCount() {
			return totalToolCount;
		}
	};
}
