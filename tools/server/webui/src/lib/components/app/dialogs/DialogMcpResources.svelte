<script lang="ts">
	import { FolderOpen, Plus, Loader2 } from '@lucide/svelte';
	import { toast } from 'svelte-sonner';
	import * as Dialog from '$lib/components/ui/dialog';
	import { Button } from '$lib/components/ui/button';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { mcpResources, mcpTotalResourceCount } from '$lib/stores/mcp-resources.svelte';
	import { McpResourceBrowser, McpResourcePreview } from '$lib/components/app';
	import type { MCPResourceInfo } from '$lib/types';
	import { SvelteSet } from 'svelte/reactivity';

	interface Props {
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
		onAttach?: (resource: MCPResourceInfo) => void;
		preSelectedUri?: string;
	}

	let { open = $bindable(false), onOpenChange, onAttach, preSelectedUri }: Props = $props();

	let selectedResources = new SvelteSet<string>();
	let lastSelectedUri = $state<string | null>(null);
	let isAttaching = $state(false);

	const totalCount = $derived(mcpTotalResourceCount());

	$effect(() => {
		if (open) {
			loadResources();

			if (preSelectedUri) {
				selectedResources.clear();
				selectedResources.add(preSelectedUri);
				lastSelectedUri = preSelectedUri;
			}
		}
	});

	async function loadResources() {
		const perChatOverrides = conversationsStore.getAllMcpServerOverrides();
		const initialized = await mcpStore.ensureInitialized(perChatOverrides);

		if (initialized) {
			await mcpStore.fetchAllResources();
		}
	}

	function handleOpenChange(newOpen: boolean) {
		open = newOpen;
		onOpenChange?.(newOpen);

		if (!newOpen) {
			selectedResources.clear();
			lastSelectedUri = null;
		}
	}

	function handleResourceSelect(resource: MCPResourceInfo, shiftKey: boolean = false) {
		if (shiftKey && lastSelectedUri) {
			// Get all resources in tree order (matching the display order in McpResourceBrowser)
			const allResources = getAllResourcesFlatInTreeOrder();
			const lastIndex = allResources.findIndex((r) => r.uri === lastSelectedUri);
			const currentIndex = allResources.findIndex((r) => r.uri === resource.uri);

			if (lastIndex !== -1 && currentIndex !== -1) {
				const start = Math.min(lastIndex, currentIndex);
				const end = Math.max(lastIndex, currentIndex);

				for (let i = start; i <= end; i++) {
					selectedResources.add(allResources[i].uri);
				}
			}
		} else {
			selectedResources.clear();
			selectedResources.add(resource.uri);
			lastSelectedUri = resource.uri;
		}
	}

	function handleResourceToggle(resource: MCPResourceInfo, checked: boolean) {
		if (checked) {
			selectedResources.add(resource.uri);
		} else {
			selectedResources.delete(resource.uri);
		}

		lastSelectedUri = resource.uri;
	}

	function getResourceDisplayName(resource: MCPResourceInfo): string {
		// Extract the display name from the resource URI (last path segment)
		try {
			const uriWithoutProtocol = resource.uri.replace(/^[a-z]+:\/\//, '');
			const parts = uriWithoutProtocol.split('/');
			return parts[parts.length - 1] || resource.name || resource.uri;
		} catch {
			return resource.name || resource.uri;
		}
	}

	function getAllResourcesFlatInTreeOrder(): MCPResourceInfo[] {
		// Get resources in the same order as displayed in McpResourceBrowser tree
		// Sort by: folders first (if applicable), then alphabetically by display name
		const allResources: MCPResourceInfo[] = [];
		const resourcesMap = mcpResources();

		for (const [serverName, serverRes] of resourcesMap.entries()) {
			for (const resource of serverRes.resources) {
				allResources.push({ ...resource, serverName });
			}
		}

		// Sort to match tree display order: folders first, then alphabetically
		return allResources.sort((a, b) => {
			const aName = getResourceDisplayName(a);
			const bName = getResourceDisplayName(b);
			return aName.localeCompare(bName);
		});
	}

	function getAllResourcesFlat(): MCPResourceInfo[] {
		// Fallback for other uses (like attaching)
		return getAllResourcesFlatInTreeOrder();
	}

	async function handleAttach() {
		if (selectedResources.size === 0) return;

		isAttaching = true;

		try {
			const allResources = getAllResourcesFlat();
			const resourcesToAttach = allResources.filter((r) => selectedResources.has(r.uri));

			for (const resource of resourcesToAttach) {
				await mcpStore.attachResource(resource.uri);
				onAttach?.(resource);
			}

			const count = resourcesToAttach.length;

			toast.success(
				count === 1
					? `Resource attached: ${resourcesToAttach[0].name}`
					: `${count} resources attached`
			);

			handleOpenChange(false);
		} catch (error) {
			console.error('Failed to attach resources:', error);
		} finally {
			isAttaching = false;
		}
	}
</script>

<Dialog.Root {open} onOpenChange={handleOpenChange}>
	<Dialog.Content class="max-h-[80vh] !max-w-4xl overflow-hidden p-0">
		<Dialog.Header class="border-b border-border/30 px-6 py-4">
			<Dialog.Title class="flex items-center gap-2">
				<FolderOpen class="h-5 w-5" />

				<span>MCP Resources</span>

				{#if totalCount > 0}
					<span class="text-sm font-normal text-muted-foreground">({totalCount})</span>
				{/if}
			</Dialog.Title>

			<Dialog.Description>
				Browse and attach resources from connected MCP servers to your chat context.
			</Dialog.Description>
		</Dialog.Header>

		<div class="flex h-[500px] min-w-0">
			<div class="w-72 shrink-0 overflow-y-auto border-r border-border/30 p-4">
				<McpResourceBrowser
					onSelect={handleResourceSelect}
					onToggle={handleResourceToggle}
					selectedUris={selectedResources}
					expandToUri={preSelectedUri}
				/>
			</div>

			<div class="min-w-0 flex-1 overflow-auto p-4">
				{#if selectedResources.size === 1}
					{@const allResources = getAllResourcesFlat()}
					{@const selectedResource = allResources.find((r) => selectedResources.has(r.uri))}

					<McpResourcePreview resource={selectedResource ?? null} />
				{:else if selectedResources.size > 1}
					<div class="flex h-full items-center justify-center text-sm text-muted-foreground">
						{selectedResources.size} resources selected
					</div>
				{:else}
					<div class="flex h-full items-center justify-center text-sm text-muted-foreground">
						Select a resource to preview
					</div>
				{/if}
			</div>
		</div>

		<Dialog.Footer class="border-t border-border/30 px-6 py-4">
			<Button variant="outline" onclick={() => handleOpenChange(false)}>Cancel</Button>

			<Button onclick={handleAttach} disabled={selectedResources.size === 0 || isAttaching}>
				{#if isAttaching}
					<Loader2 class="mr-2 h-4 w-4 animate-spin" />
				{:else}
					<Plus class="mr-2 h-4 w-4" />
				{/if}

				Attach {selectedResources.size > 0 ? `(${selectedResources.size})` : 'Resource'}
			</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
