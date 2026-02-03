<script lang="ts">
	import { cn } from '$lib/components/ui/utils';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { mcpResources, mcpResourcesLoading } from '$lib/stores/mcp-resources.svelte';
	import type { MCPResourceInfo } from '$lib/types';
	import { SvelteSet } from 'svelte/reactivity';
	import { parseResourcePath } from './mcp-resource-browser';
	import McpResourceBrowserHeader from './McpResourceBrowserHeader.svelte';
	import McpResourceBrowserEmptyState from './McpResourceBrowserEmptyState.svelte';
	import McpResourceBrowserServerItem from './McpResourceBrowserServerItem.svelte';

	interface Props {
		onSelect?: (resource: MCPResourceInfo, shiftKey?: boolean) => void;
		onToggle?: (resource: MCPResourceInfo, checked: boolean) => void;
		onAttach?: (resource: MCPResourceInfo) => void;
		selectedUris?: Set<string>;
		expandToUri?: string;
		class?: string;
	}

	let {
		onSelect,
		onToggle,
		onAttach,
		selectedUris = new Set(),
		expandToUri,
		class: className
	}: Props = $props();

	let expandedServers = new SvelteSet<string>();
	let expandedFolders = new SvelteSet<string>();
	let lastExpandedUri = $state<string | undefined>(undefined);

	const resources = $derived(mcpResources());
	const isLoading = $derived(mcpResourcesLoading());

	$effect(() => {
		if (expandToUri && resources.size > 0 && expandToUri !== lastExpandedUri) {
			autoExpandToResource(expandToUri);
			lastExpandedUri = expandToUri;
		}
	});

	function autoExpandToResource(uri: string) {
		for (const [serverName, serverRes] of resources.entries()) {
			const resource = serverRes.resources.find((r) => r.uri === uri);
			if (resource) {
				expandedServers.add(serverName);

				const pathParts = parseResourcePath(uri);
				if (pathParts.length > 1) {
					let currentPath = '';
					for (let i = 0; i < pathParts.length - 1; i++) {
						currentPath = `${currentPath}/${pathParts[i]}`;
						const folderId = `${serverName}:${currentPath}`;
						expandedFolders.add(folderId);
					}
				}
				break;
			}
		}
	}

	function toggleServer(serverName: string) {
		if (expandedServers.has(serverName)) {
			expandedServers.delete(serverName);
		} else {
			expandedServers.add(serverName);
		}
	}

	function toggleFolder(folderId: string) {
		if (expandedFolders.has(folderId)) {
			expandedFolders.delete(folderId);
		} else {
			expandedFolders.add(folderId);
		}
	}

	function handleRefresh() {
		mcpStore.fetchAllResources();
	}
</script>

<div class={cn('flex flex-col gap-2', className)}>
	<McpResourceBrowserHeader {isLoading} onRefresh={handleRefresh} />

	<div class="flex flex-col gap-1">
		{#if resources.size === 0}
			<McpResourceBrowserEmptyState {isLoading} />
		{:else}
			{#each [...resources.entries()] as [serverName, serverRes] (serverName)}
				<McpResourceBrowserServerItem
					{serverName}
					{serverRes}
					isExpanded={expandedServers.has(serverName)}
					{selectedUris}
					{expandedFolders}
					onToggleServer={() => toggleServer(serverName)}
					onToggleFolder={toggleFolder}
					{onSelect}
					{onToggle}
					{onAttach}
				/>
			{/each}
		{/if}
	</div>
</div>
