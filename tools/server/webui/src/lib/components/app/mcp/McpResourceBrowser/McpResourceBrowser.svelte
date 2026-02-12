<script lang="ts">
	import { cn } from '$lib/components/ui/utils';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { mcpResources, mcpResourcesLoading } from '$lib/stores/mcp-resources.svelte';
	import type { MCPServerResources, MCPResourceInfo } from '$lib/types';
	import { SvelteMap, SvelteSet } from 'svelte/reactivity';
	import { parseResourcePath } from '$lib/utils';
	import McpResourceBrowserHeader from './McpResourceBrowserHeader.svelte';
	import McpResourceBrowserEmptyState from './McpResourceBrowserEmptyState.svelte';
	import McpResourceBrowserServerItem from './McpResourceBrowserServerItem.svelte';

	interface Props {
		onSelect?: (resource: MCPResourceInfo, shiftKey?: boolean) => void;
		onToggle?: (resource: MCPResourceInfo, checked: boolean) => void;
		selectedUris?: Set<string>;
		expandToUri?: string;
		class?: string;
	}

	let {
		onSelect,
		onToggle,
		selectedUris = new Set(),
		expandToUri,
		class: className
	}: Props = $props();

	let expandedServers = new SvelteSet<string>();
	let expandedFolders = new SvelteSet<string>();
	let searchQuery = $state('');

	const resources = $derived(mcpResources());
	const isLoading = $derived(mcpResourcesLoading());

	const filteredResources = $derived.by(() => {
		if (!searchQuery.trim()) {
			return resources;
		}

		const query = searchQuery.toLowerCase();
		const filtered = new SvelteMap();

		for (const [serverName, serverRes] of resources.entries()) {
			const filteredResources = serverRes.resources.filter((r) => {
				return (
					r.title?.toLowerCase().includes(query) ||
					r.uri.toLowerCase().includes(query) ||
					serverName.toLowerCase().includes(query)
				);
			});

			if (filteredResources.length > 0 || query.trim()) {
				filtered.set(serverName, {
					...serverRes,
					resources: filteredResources
				});
			}
		}

		return filtered;
	});

	$effect(() => {
		if (expandToUri && resources.size > 0) {
			autoExpandToResource(expandToUri);
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
	<McpResourceBrowserHeader
		{isLoading}
		onRefresh={handleRefresh}
		onSearch={(q) => (searchQuery = q)}
		{searchQuery}
	/>

	<div class="flex flex-col gap-1">
		{#if filteredResources.size === 0}
			<McpResourceBrowserEmptyState {isLoading} />
		{:else}
			{#each [...filteredResources.entries()] as [serverName, serverRes] (serverName)}
				<McpResourceBrowserServerItem
					serverName={serverName as string}
					serverRes={serverRes as MCPServerResources}
					isExpanded={expandedServers.has(serverName as string)}
					{selectedUris}
					{expandedFolders}
					onToggleServer={() => toggleServer(serverName as string)}
					onToggleFolder={toggleFolder}
					{onSelect}
					{onToggle}
					{searchQuery}
				/>
			{/each}
		{/if}
	</div>
</div>
