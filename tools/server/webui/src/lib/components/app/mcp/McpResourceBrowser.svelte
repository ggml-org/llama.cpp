<script lang="ts">
	import {
		FileText,
		FolderOpen,
		ChevronDown,
		ChevronRight,
		RefreshCw,
		Loader2,
		Database,
		File,
		Image,
		Code
	} from '@lucide/svelte';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import * as Collapsible from '$lib/components/ui/collapsible';
	import { Button } from '$lib/components/ui/button';
	import { cn } from '$lib/components/ui/utils';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { mcpResources, mcpResourcesLoading } from '$lib/stores/mcp-resources.svelte';
	import { getFaviconUrl } from '$lib/utils';
	import { TruncatedText } from '$lib/components/app';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import type { MCPResource, MCPResourceInfo } from '$lib/types';
	import { SvelteSet } from 'svelte/reactivity';

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
	let hasAutoExpanded = $state(false);

	const resources = $derived(mcpResources());
	const isLoading = $derived(mcpResourcesLoading());

	$effect(() => {
		if (expandToUri && resources.size > 0 && !hasAutoExpanded) {
			autoExpandToResource(expandToUri);
			hasAutoExpanded = true;
		}
	});

	function autoExpandToResource(uri: string) {
		for (const [serverName, serverRes] of resources.entries()) {
			const resource = serverRes.resources.find((r) => r.uri === uri);
			if (resource) {
				const newExpandedServers = new SvelteSet(expandedServers);
				newExpandedServers.add(serverName);
				expandedServers = newExpandedServers;

				const pathParts = parseResourcePath(uri);
				if (pathParts.length > 1) {
					const newExpandedFolders = new SvelteSet(expandedFolders);
					let currentPath = '';
					for (let i = 0; i < pathParts.length - 1; i++) {
						currentPath = `${currentPath}/${pathParts[i]}`;
						const folderId = `${serverName}:${currentPath}`;
						newExpandedFolders.add(folderId);
					}
					expandedFolders = newExpandedFolders;
				}
				break;
			}
		}
	}

	function toggleServer(serverName: string) {
		const newSet = new SvelteSet(expandedServers);
		if (newSet.has(serverName)) {
			newSet.delete(serverName);
		} else {
			newSet.add(serverName);
		}
		expandedServers = newSet;
	}

	function toggleFolder(folderId: string) {
		const newSet = new SvelteSet(expandedFolders);
		if (newSet.has(folderId)) {
			newSet.delete(folderId);
		} else {
			newSet.add(folderId);
		}
		expandedFolders = newSet;
	}

	interface ResourceTreeNode {
		name: string;
		resource?: MCPResourceInfo;
		children: Map<string, ResourceTreeNode>;
	}

	function parseResourcePath(uri: string): string[] {
		// Parse URI like "svelte://cli/overview.md" -> ["cli", "overview.md"]
		try {
			// Remove protocol (e.g., "svelte://")
			const withoutProtocol = uri.replace(/^[a-z]+:\/\//, '');
			// Split by / and filter empty parts
			return withoutProtocol.split('/').filter((p) => p.length > 0);
		} catch {
			return [uri];
		}
	}

	function getDisplayName(pathPart: string): string {
		// Convert filename to display name: "overview.md" -> "Overview"
		const withoutExt = pathPart.replace(/\.[^.]+$/, '');
		// Convert kebab-case or snake_case to Title Case
		return withoutExt
			.split(/[-_]/)
			.map((word) => word.charAt(0).toUpperCase() + word.slice(1))
			.join(' ');
	}

	function buildResourceTree(resourceList: MCPResource[], serverName: string): ResourceTreeNode {
		const root: ResourceTreeNode = { name: 'root', children: new Map() };

		for (const resource of resourceList) {
			const pathParts = parseResourcePath(resource.uri);
			let current = root;

			// Navigate/create folders for all but the last part
			for (let i = 0; i < pathParts.length - 1; i++) {
				const part = pathParts[i];
				if (!current.children.has(part)) {
					current.children.set(part, { name: part, children: new Map() });
				}
				current = current.children.get(part)!;
			}

			// Add the resource at the leaf
			const fileName = pathParts[pathParts.length - 1] || resource.name;
			current.children.set(resource.uri, {
				name: fileName,
				resource: { ...resource, serverName },
				children: new Map()
			});
		}

		return root;
	}

	function countTreeResources(node: ResourceTreeNode): number {
		if (node.resource) return 1;
		let count = 0;
		for (const child of node.children.values()) {
			count += countTreeResources(child);
		}
		return count;
	}

	function handleRefresh() {
		mcpStore.fetchAllResources();
	}

	function handleResourceClick(resource: MCPResourceInfo, event: MouseEvent) {
		onSelect?.(resource, event.shiftKey);
	}

	function handleCheckboxChange(resource: MCPResourceInfo, checked: boolean) {
		onToggle?.(resource, checked);
	}

	function handleAttachClick(e: Event, resource: MCPResourceInfo) {
		e.stopPropagation();
		onAttach?.(resource);
	}

	function getResourceIcon(resource: MCPResourceInfo) {
		const mimeType = resource.mimeType?.toLowerCase() || '';
		const uri = resource.uri.toLowerCase();

		if (mimeType.startsWith('image/') || /\.(png|jpg|jpeg|gif|svg|webp)$/.test(uri)) {
			return Image;
		}
		if (
			mimeType.includes('json') ||
			mimeType.includes('javascript') ||
			mimeType.includes('typescript') ||
			/\.(js|ts|json|yaml|yml|xml|html|css)$/.test(uri)
		) {
			return Code;
		}
		if (mimeType.includes('text') || /\.(txt|md|log)$/.test(uri)) {
			return FileText;
		}
		if (uri.includes('database') || uri.includes('db://')) {
			return Database;
		}
		return File;
	}

	function isResourceSelected(resource: MCPResourceInfo): boolean {
		return selectedUris.has(resource.uri);
	}

	function getServerDisplayName(serverId: string): string {
		const server = mcpStore.getServerById(serverId);
		return server ? mcpStore.getServerLabel(server) : serverId;
	}

	function getServerFavicon(serverId: string): string | null {
		const server = mcpStore.getServerById(serverId);
		return server ? getFaviconUrl(server.url) : null;
	}
</script>

{#snippet renderTreeNode(
	node: ResourceTreeNode,
	serverName: string,
	depth: number,
	parentPath: string
)}
	{@const isFolder = !node.resource && node.children.size > 0}
	{@const folderId = `${serverName}:${parentPath}/${node.name}`}
	{@const isFolderExpanded = expandedFolders.has(folderId)}

	{#if isFolder}
		{@const folderCount = countTreeResources(node)}
		<Collapsible.Root open={isFolderExpanded} onOpenChange={() => toggleFolder(folderId)}>
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
					{#each [...node.children.values()].sort((a, b) => {
						// Folders first, then files
						const aIsFolder = !a.resource && a.children.size > 0;
						const bIsFolder = !b.resource && b.children.size > 0;
						if (aIsFolder && !bIsFolder) return -1;
						if (!aIsFolder && bIsFolder) return 1;
						return a.name.localeCompare(b.name);
					}) as child (child.resource?.uri || `${serverName}:${parentPath}/${node.name}/${child.name}`)}
						{@render renderTreeNode(child, serverName, depth + 1, `${parentPath}/${node.name}`)}
					{/each}
				</div>
			</Collapsible.Content>
		</Collapsible.Root>
	{:else if node.resource}
		{@const resource = node.resource}
		{@const ResourceIcon = getResourceIcon(resource)}
		{@const isSelected = isResourceSelected(resource)}
		{@const displayName = resource.title || getDisplayName(node.name)}
		<div class="group flex w-full items-center gap-2">
			{#if onToggle}
				<Checkbox
					checked={isSelected}
					onCheckedChange={(checked) => handleCheckboxChange(resource, checked === true)}
					class="h-4 w-4"
				/>
			{/if}
			<button
				class={cn(
					'flex flex-1 items-center gap-2 rounded px-2 py-1 text-left text-sm transition-colors',
					'hover:bg-muted/50',
					isSelected && 'bg-muted'
				)}
				onclick={(e) => handleResourceClick(resource, e)}
			>
				<ResourceIcon class="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
				<Tooltip.Root>
					<Tooltip.Trigger class="min-w-0 flex-1 text-left">
						<TruncatedText text={displayName} />
					</Tooltip.Trigger>
					<Tooltip.Content class="z-[9999]">
						<p>{displayName}</p>
					</Tooltip.Content>
				</Tooltip.Root>
				{#if onAttach}
					<Button
						variant="ghost"
						size="sm"
						class="h-5 px-1.5 text-xs opacity-0 transition-opacity group-hover:opacity-100 hover:opacity-100"
						onclick={(e) => handleAttachClick(e, resource)}
					>
						Attach
					</Button>
				{/if}
			</button>
		</div>
	{/if}
{/snippet}

<div class={cn('flex flex-col gap-2', className)}>
	<div class="flex items-center justify-between">
		<h3 class="text-sm font-medium">Available resources</h3>

		<Button
			variant="ghost"
			size="sm"
			class="h-7 w-7 p-0"
			onclick={handleRefresh}
			disabled={isLoading}
			title="Refresh resources"
		>
			{#if isLoading}
				<Loader2 class="h-3.5 w-3.5 animate-spin" />
			{:else}
				<RefreshCw class="h-3.5 w-3.5" />
			{/if}
		</Button>
	</div>

	<div class="flex flex-col gap-1">
		{#if resources.size === 0}
			<div class="py-4 text-center text-sm text-muted-foreground">
				{#if isLoading}
					Loading resources...
				{:else}
					No resources available
				{/if}
			</div>
		{:else}
			{#each [...resources.entries()] as [serverName, serverRes] (serverName)}
				{@const isExpanded = expandedServers.has(serverName)}
				{@const hasResources = serverRes.resources.length > 0}
				{@const displayName = getServerDisplayName(serverName)}
				{@const favicon = getServerFavicon(serverName)}
				{@const resourceTree = buildResourceTree(serverRes.resources, serverName)}
				<Collapsible.Root open={isExpanded} onOpenChange={() => toggleServer(serverName)}>
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
								{#each [...resourceTree.children.values()].sort((a, b) => {
									const aIsFolder = !a.resource && a.children.size > 0;
									const bIsFolder = !b.resource && b.children.size > 0;

									if (aIsFolder && !bIsFolder) return -1;
									if (!aIsFolder && bIsFolder) return 1;

									return a.name.localeCompare(b.name);
								}) as child (child.resource?.uri || `${serverName}:${child.name}`)}
									{@render renderTreeNode(child, serverName, 1, '')}
								{/each}
							{/if}
						</div>
					</Collapsible.Content>
				</Collapsible.Root>
			{/each}
		{/if}
	</div>
</div>
