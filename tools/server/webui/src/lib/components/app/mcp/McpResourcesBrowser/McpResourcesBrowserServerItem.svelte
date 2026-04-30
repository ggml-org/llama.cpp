<script lang="ts">
	import { FolderOpen, ChevronDown, ChevronRight, Loader2, Braces } from '@lucide/svelte';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import * as Collapsible from '$lib/components/ui/collapsible';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import type { MCPResourceInfo, MCPResourceTemplateInfo, MCPServerResources } from '$lib/types';
	import { SvelteSet } from 'svelte/reactivity';
	import {
		type ResourceTreeNode,
		buildResourceTree,
		countTreeResources,
		sortTreeChildren
	} from './mcp-resources-browser';
	import { getDisplayName, getResourceIcon } from '$lib/utils';

	interface Props {
		serverName: string;
		serverRes: MCPServerResources;
		isExpanded: boolean;
		selectedUris: Set<string>;
		selectedTemplateUri?: string | null;
		expandedFolders: SvelteSet<string>;
		onToggleServer: () => void;
		onToggleFolder: (folderId: string) => void;
		onSelect?: (resource: MCPResourceInfo, shiftKey?: boolean) => void;
		onToggle?: (resource: MCPResourceInfo, checked: boolean) => void;
		onTemplateSelect?: (template: MCPResourceTemplateInfo) => void;
		searchQuery?: string;
	}

	let {
		serverName,
		serverRes,
		isExpanded,
		selectedUris,
		selectedTemplateUri,
		expandedFolders,
		onToggleServer,
		onToggleFolder,
		onSelect,
		onToggle,
		onTemplateSelect,
		searchQuery = ''
	}: Props = $props();

	const hasResources = $derived(serverRes.resources.length > 0);
	const hasTemplates = $derived(serverRes.templates.length > 0);
	const hasContent = $derived(hasResources || hasTemplates);
	const displayName = $derived(mcpStore.getServerDisplayName(serverName));
	const favicon = $derived(mcpStore.getServerFavicon(serverName));
	const resourceTree = $derived(buildResourceTree(serverRes.resources, serverName, searchQuery));

	const templateInfos = $derived<MCPResourceTemplateInfo[]>(
		serverRes.templates.map((t) => ({
			uriTemplate: t.uriTemplate,
			name: t.name,
			title: t.title,
			description: t.description,
			mimeType: t.mimeType,
			serverName,
			annotations: t.annotations,
			icons: t.icons
		}))
	);

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
		{@const ResourceIcon = getResourceIcon(resource.mimeType, resource.uri)}
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
				class={[
					'flex flex-1 items-center gap-2 rounded px-2 py-1 text-left text-sm transition-colors',
					'hover:bg-muted/50',
					isSelected && 'bg-muted'
				]}
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

		<span class="inline-flex flex-col items-start text-left">
			<span class="inline-flex items-center justify-start gap-1.5 font-medium">
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

				{displayName}
			</span>

			<span class="text-xs text-muted-foreground">
				({serverRes.resources.length} resource{serverRes.resources.length !== 1
					? 's'
					: ''}{#if hasTemplates}, {serverRes.templates.length} template{serverRes.templates
						.length !== 1
						? 's'
						: ''}{/if})
			</span>
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
			{:else if !hasContent}
				<div class="py-1 text-xs text-muted-foreground">No resources</div>
			{:else}
				{#if hasResources}
					{#each sortTreeChildren( [...resourceTree.children.values()] ) as child (child.resource?.uri || `${serverName}:${child.name}`)}
						{@render renderTreeNode(child, 1, '')}
					{/each}
				{/if}

				{#if hasTemplates && onTemplateSelect}
					{#if hasResources}
						<div class="my-1 border-t border-border/30"></div>
					{/if}

					<div
						class="py-0.5 text-[11px] font-medium tracking-wide text-muted-foreground/70 uppercase"
					>
						Templates
					</div>

					{#each templateInfos as template (template.uriTemplate)}
						<button
							class={[
								'flex w-full items-center gap-2 rounded px-2 py-1 text-left text-sm transition-colors',
								'hover:bg-muted/50',
								selectedTemplateUri === template.uriTemplate && 'bg-muted'
							]}
							onclick={() => onTemplateSelect(template)}
							title={template.uriTemplate}
						>
							<Braces class="h-3.5 w-3.5 shrink-0 text-muted-foreground" />

							<span class="min-w-0 flex-1 truncate text-left">
								{template.title || template.name}
							</span>
						</button>
					{/each}
				{/if}
			{/if}
		</div>
	</Collapsible.Content>
</Collapsible.Root>
