<script lang="ts">
	import { FolderOpen, ChevronDown, ChevronRight, Loader2 } from '@lucide/svelte';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import * as Collapsible from '$lib/components/ui/collapsible';
	import { cn } from '$lib/components/ui/utils';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import type { MCPResourceInfo, MCPServerResources } from '$lib/types';
	import { SvelteSet } from 'svelte/reactivity';
	import {
		type ResourceTreeNode,
		buildResourceTree,
		countTreeResources,
		getDisplayName,
		getResourceIcon,
		sortTreeChildren
	} from './mcp-resource-browser';

	interface Props {
		serverName: string;
		serverRes: MCPServerResources;
		isExpanded: boolean;
		selectedUris: Set<string>;
		expandedFolders: SvelteSet<string>;
		onToggleServer: () => void;
		onToggleFolder: (folderId: string) => void;
		onSelect?: (resource: MCPResourceInfo, shiftKey?: boolean) => void;
		onToggle?: (resource: MCPResourceInfo, checked: boolean) => void;
		searchQuery?: string;
	}

	let {
		serverName,
		serverRes,
		isExpanded,
		selectedUris,
		expandedFolders,
		onToggleServer,
		onToggleFolder,
		onSelect,
		onToggle,
		searchQuery = ''
	}: Props = $props();

	const hasResources = $derived(serverRes.resources.length > 0);
	const displayName = $derived(mcpStore.getServerDisplayName(serverName));
	const favicon = $derived(mcpStore.getServerFavicon(serverName));
	const resourceTree = $derived(buildResourceTree(serverRes.resources, serverName, searchQuery));

	function handleResourceClick(resource: MCPResourceInfo, event: MouseEvent) {
		onSelect?.(resource, event.shiftKey);
	}

	function handleCheckboxChange(resource: MCPResourceInfo, checked: boolean) {
		onToggle?.(resource, checked);
	}

	function isResourceSelected(resource: MCPResourceInfo): boolean {
		return selectedUris.has(resource.uri);
	}
</script>

{#snippet renderTreeNode(node: ResourceTreeNode, depth: number, parentPath: string)}
	{@const isFolder = !node.resource && node.children.size > 0}
	{@const folderId = `${serverName}:${parentPath}/${node.name}`}
	{@const isFolderExpanded = expandedFolders.has(folderId)}

	{#if isFolder}
		{@const folderCount = countTreeResources(node)}
		<Collapsible.Root open={isFolderExpanded} onOpenChange={() => onToggleFolder(folderId)}>
			<Collapsible.Trigger
				class="flex w-full items-center gap-2 rounded px-2 py-1 text-sm hover:bg-muted/50"
			>
				{#if isFolderExpanded}
					<ChevronDown class="h-3 w-3" />
				{:else}
					<ChevronRight class="h-3 w-3" />
				{/if}

				<FolderOpen class="h-3.5 w-3.5 text-muted-foreground" />

				<span class="font-medium">{node.name}</span>

				<span class="text-xs text-muted-foreground">({folderCount})</span>
			</Collapsible.Trigger>

			<Collapsible.Content>
				<div class="ml-4 flex flex-col gap-0.5 border-l border-border/50 pl-2">
					{#each sortTreeChildren( [...node.children.values()] ) as child (child.resource?.uri || `${serverName}:${parentPath}/${node.name}/${child.name}`)}
						{@render renderTreeNode(child, depth + 1, `${parentPath}/${node.name}`)}
					{/each}
				</div>
			</Collapsible.Content>
		</Collapsible.Root>
	{:else if node.resource}
		{@const resource = node.resource}
		{@const ResourceIcon = getResourceIcon(resource)}
		{@const isSelected = isResourceSelected(resource)}
		{@const resourceDisplayName = resource.title || getDisplayName(node.name)}

		<div class="group flex w-full items-center gap-2">
			{#if onToggle}
				<Checkbox
					checked={isSelected}
					onCheckedChange={(checked: boolean | 'indeterminate') =>
						handleCheckboxChange(resource, checked === true)}
					class="h-4 w-4"
				/>
			{/if}

			<button
				class={cn(
					'flex flex-1 items-center gap-2 rounded px-2 py-1 text-left text-sm transition-colors',
					'hover:bg-muted/50',
					isSelected && 'bg-muted'
				)}
				onclick={(e: MouseEvent) => handleResourceClick(resource, e)}
				title={resourceDisplayName}
			>
				<ResourceIcon class="h-3.5 w-3.5 shrink-0 text-muted-foreground" />

				<span class="min-w-0 flex-1 truncate text-left">
					{resourceDisplayName}
				</span>
			</button>
		</div>
	{/if}
{/snippet}

<Collapsible.Root open={isExpanded} onOpenChange={onToggleServer}>
	<Collapsible.Trigger
		class="flex w-full items-center gap-2 rounded px-2 py-1.5 text-sm hover:bg-muted/50"
	>
		{#if isExpanded}
			<ChevronDown class="h-3.5 w-3.5" />
		{:else}
			<ChevronRight class="h-3.5 w-3.5" />
		{/if}

		{#if favicon}
			<img
				src={favicon}
				alt=""
				class="h-4 w-4 shrink-0 rounded-sm"
				onerror={(e) => {
					(e.currentTarget as HTMLImageElement).style.display = 'none';
				}}
			/>
		{/if}

		<span class="font-medium">{displayName}</span>

		<span class="text-xs text-muted-foreground">
			({serverRes.resources.length})
		</span>

		{#if serverRes.loading}
			<Loader2 class="ml-auto h-3 w-3 animate-spin text-muted-foreground" />
		{/if}
	</Collapsible.Trigger>

	<Collapsible.Content>
		<div class="ml-4 flex flex-col gap-0.5 border-l border-border/50 pl-2">
			{#if serverRes.error}
				<div class="py-1 text-xs text-red-500">
					Error: {serverRes.error}
				</div>
			{:else if !hasResources}
				<div class="py-1 text-xs text-muted-foreground">No resources</div>
			{:else}
				{#each sortTreeChildren( [...resourceTree.children.values()] ) as child (child.resource?.uri || `${serverName}:${child.name}`)}
					{@render renderTreeNode(child, 1, '')}
				{/each}
			{/if}
		</div>
	</Collapsible.Content>
</Collapsible.Root>
