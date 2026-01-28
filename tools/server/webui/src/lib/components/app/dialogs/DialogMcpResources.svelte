<script lang="ts">
	import { FolderOpen, Plus, Loader2 } from '@lucide/svelte';
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
				selectedResources = new SvelteSet([preSelectedUri]);
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
			selectedResources = new SvelteSet();
			lastSelectedUri = null;
		}
	}

	function handleResourceSelect(resource: MCPResourceInfo, shiftKey: boolean = false) {
		if (shiftKey && lastSelectedUri) {
			const allResources = getAllResourcesFlat();
			const lastIndex = allResources.findIndex((r) => r.uri === lastSelectedUri);
			const currentIndex = allResources.findIndex((r) => r.uri === resource.uri);

			if (lastIndex !== -1 && currentIndex !== -1) {
				const start = Math.min(lastIndex, currentIndex);
				const end = Math.max(lastIndex, currentIndex);
				const newSelection = new SvelteSet(selectedResources);

				for (let i = start; i <= end; i++) {
					newSelection.add(allResources[i].uri);
				}

				selectedResources = newSelection;
			}
		} else {
			selectedResources = new SvelteSet([resource.uri]);
			lastSelectedUri = resource.uri;
		}
	}

	function handleResourceToggle(resource: MCPResourceInfo, checked: boolean) {
		const newSelection = new SvelteSet(selectedResources);
		if (checked) {
			newSelection.add(resource.uri);
		} else {
			newSelection.delete(resource.uri);
		}
		selectedResources = newSelection;
		lastSelectedUri = resource.uri;
	}

	function getAllResourcesFlat(): MCPResourceInfo[] {
		const allResources: MCPResourceInfo[] = [];
		const resourcesMap = mcpResources();

		for (const [serverName, serverRes] of resourcesMap.entries()) {
			for (const resource of serverRes.resources) {
				allResources.push({ ...resource, serverName });
			}
		}

		return allResources;
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

			handleOpenChange(false);
		} catch (error) {
			console.error('Failed to attach resources:', error);
		} finally {
			isAttaching = false;
		}
	}

	async function handleQuickAttach(resource: MCPResourceInfo) {
		isAttaching = true;
		try {
			await mcpStore.attachResource(resource.uri);
			onAttach?.(resource);
		} catch (error) {
			console.error('Failed to attach resource:', error);
		} finally {
			isAttaching = false;
		}
	}
</script>

<Dialog.Root {open} onOpenChange={handleOpenChange}>
	<Dialog.Content class="max-h-[80vh] !max-w-4xl overflow-hidden p-0">
		<Dialog.Header class="border-b px-6 py-4">
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

		<div class="flex h-[500px]">
			<div class="w-72 shrink-0 overflow-y-auto border-r p-4">
				<McpResourceBrowser
					onSelect={handleResourceSelect}
					onToggle={handleResourceToggle}
					onAttach={handleQuickAttach}
					selectedUris={selectedResources}
					expandToUri={preSelectedUri}
				/>
			</div>

			<div class="flex-1 overflow-y-auto p-4">
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

		<Dialog.Footer class="border-t px-6 py-4">
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
