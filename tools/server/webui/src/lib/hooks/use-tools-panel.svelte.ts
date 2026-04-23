import { SvelteSet } from 'svelte/reactivity';
import { ToolSource } from '$lib/enums';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import { mcpStore } from '$lib/stores/mcp.svelte';
import { toolsStore } from '$lib/stores/tools.svelte';
import type { ToolGroup } from '$lib/types';

export interface UseToolsPanelReturn {
	readonly expandedGroups: SvelteSet<string>;
	readonly groups: ToolGroup[];
	readonly activeGroups: ToolGroup[];
	readonly totalToolCount: number;
	readonly noToolsInfoMessage: string | null;
	getGroupCheckedState(group: ToolGroup): { checked: boolean; indeterminate: boolean };
	getEnabledToolCount(group: ToolGroup): number;
	getFavicon(group: { source: ToolSource; label: string }): string | null;
	isGroupDisabled(group: ToolGroup): boolean;
	toggleGroupExpanded(label: string): void;
	handleOpen(): void;
}

/**
 * Shared reactive state and helpers for the tools panel UI.
 *
 * Used by both the desktop dropdown (`ChatFormActionToolsSubmenu`)
 * and the mobile sheet (`ChatFormActionAttachmentsSheet`) to avoid
 * duplicating group filtering, checked-state derivation, and favicon logic.
 */
export function useToolsPanel(): UseToolsPanelReturn {
	const expandedGroups = new SvelteSet<string>();

	const groups = $derived(toolsStore.toolGroups);
	const activeGroups = $derived(
		groups.filter(
			(g) =>
				g.source !== ToolSource.MCP ||
				!g.serverId ||
				conversationsStore.isMcpServerEnabledForChat(g.serverId)
		)
	);
	const totalToolCount = $derived(activeGroups.reduce((n, g) => n + g.tools.length, 0));
	const noToolsInfoMessage = $derived.by(() => {
		if (toolsStore.loading) return null;
		if (toolsStore.toolGroups.length > 0) return null;
		// Tools endpoint is unreachable (404) — server started without --tools
		if (toolsStore.isToolsEndpointUnreachable) {
			return 'To enable Built-In Tools you need to run llama-server with --tools all or --tools <name> flag. To see MCP Tools you need to add / enable MCP Server(s).';
		}
		// Other errors — return null so UI shows "Failed to load tools"
		if (toolsStore.error) return null;
		return 'To enable Built-In Tools you need to run llama-server with --tools all or --tools <name> flag. To see MCP Tools you need to add / enable MCP Server(s).';
	});

	function getGroupCheckedState(group: ToolGroup): { checked: boolean; indeterminate: boolean } {
		return {
			checked: toolsStore.isGroupFullyEnabled(group),
			indeterminate: toolsStore.isGroupPartiallyEnabled(group)
		};
	}

	function getEnabledToolCount(group: ToolGroup): number {
		return group.tools.filter((tool) => toolsStore.isToolEnabled(tool.function.name)).length;
	}

	function getFavicon(group: { source: ToolSource; label: string }): string | null {
		if (group.source !== ToolSource.MCP) return null;

		for (const server of mcpStore.getServersSorted()) {
			if (mcpStore.getServerLabel(server) === group.label) {
				return mcpStore.getServerFavicon(server.id);
			}
		}

		return null;
	}

	function isGroupDisabled(group: ToolGroup): boolean {
		return (
			group.source === ToolSource.MCP &&
			!!group.serverId &&
			!conversationsStore.isMcpServerEnabledForChat(group.serverId)
		);
	}

	function toggleGroupExpanded(label: string): void {
		if (expandedGroups.has(label)) {
			expandedGroups.delete(label);
		} else {
			expandedGroups.add(label);
		}
	}

	function handleOpen(): void {
		if (toolsStore.builtinTools.length === 0 && !toolsStore.loading) {
			toolsStore.fetchBuiltinTools();
		}
		mcpStore.runHealthChecksForServers(mcpStore.getServersSorted().filter((s) => s.enabled));
	}

	return {
		expandedGroups,
		get groups() {
			return groups;
		},
		get activeGroups() {
			return activeGroups;
		},
		get totalToolCount() {
			return totalToolCount;
		},
		get noToolsInfoMessage() {
			return noToolsInfoMessage;
		},
		getGroupCheckedState,
		getEnabledToolCount,
		getFavicon,
		isGroupDisabled,
		toggleGroupExpanded,
		handleOpen
	};
}
