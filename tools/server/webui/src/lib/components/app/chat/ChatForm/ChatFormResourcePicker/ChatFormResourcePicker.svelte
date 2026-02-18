<script lang="ts">
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { mcpResourceStore } from '$lib/stores/mcp-resources.svelte';
	import { KeyboardKey } from '$lib/enums';
	import type { MCPResourceInfo, MCPServerSettingsEntry } from '$lib/types';
	import { SvelteMap } from 'svelte/reactivity';
	import { ChatFormResourcePickerList, ChatFormPickerPopover } from '$lib/components/app/chat';

	interface Props {
		class?: string;
		isOpen?: boolean;
		searchQuery?: string;
		onClose?: () => void;
		onResourceSelect?: (resource: MCPResourceInfo) => void;
		onBrowse?: () => void;
	}

	let {
		class: className = '',
		isOpen = false,
		searchQuery = '',
		onClose,
		onResourceSelect,
		onBrowse
	}: Props = $props();

	let resources = $state<MCPResourceInfo[]>([]);
	let isLoading = $state(false);
	let selectedIndex = $state(0);
	let internalSearchQuery = $state('');

	let serverSettingsMap = $derived.by(() => {
		const servers = mcpStore.getServers();
		const map = new SvelteMap<string, MCPServerSettingsEntry>();

		for (const server of servers) {
			map.set(server.id, server);
		}

		return map;
	});

	$effect(() => {
		if (isOpen) {
			loadResources();
			selectedIndex = 0;
		}
	});

	$effect(() => {
		if (filteredResources.length > 0 && selectedIndex >= filteredResources.length) {
			selectedIndex = 0;
		}
	});

	async function loadResources() {
		isLoading = true;

		try {
			const perChatOverrides = conversationsStore.getAllMcpServerOverrides();

			const initialized = await mcpStore.ensureInitialized(perChatOverrides);

			if (!initialized) {
				resources = [];

				return;
			}

			await mcpStore.fetchAllResources();
			resources = mcpResourceStore.getAllResourceInfos();
		} catch (error) {
			console.error('[ChatFormResourcePicker] Failed to load resources:', error);
			resources = [];
		} finally {
			isLoading = false;
		}
	}

	function handleResourceClick(resource: MCPResourceInfo) {
		mcpStore.attachResource(resource.uri);
		onResourceSelect?.(resource);
		onClose?.();
	}

	function isResourceAttached(uri: string): boolean {
		return mcpResourceStore.isAttached(uri);
	}

	export function handleKeydown(event: KeyboardEvent): boolean {
		if (!isOpen) return false;

		if (event.key === KeyboardKey.ESCAPE) {
			event.preventDefault();
			onClose?.();

			return true;
		}

		if (event.key === KeyboardKey.ARROW_DOWN) {
			event.preventDefault();

			if (filteredResources.length > 0) {
				selectedIndex = (selectedIndex + 1) % filteredResources.length;
			}

			return true;
		}

		if (event.key === KeyboardKey.ARROW_UP) {
			event.preventDefault();
			if (filteredResources.length > 0) {
				selectedIndex = selectedIndex === 0 ? filteredResources.length - 1 : selectedIndex - 1;
			}

			return true;
		}

		if (event.key === KeyboardKey.ENTER) {
			event.preventDefault();
			if (filteredResources[selectedIndex]) {
				handleResourceClick(filteredResources[selectedIndex]);
			}

			return true;
		}

		return false;
	}

	let filteredResources = $derived.by(() => {
		const sortedServers = mcpStore.getServersSorted();
		const serverOrderMap = new Map(sortedServers.map((server, index) => [server.id, index]));

		const sortedResources = [...resources].sort((a, b) => {
			const orderA = serverOrderMap.get(a.serverName) ?? Number.MAX_SAFE_INTEGER;
			const orderB = serverOrderMap.get(b.serverName) ?? Number.MAX_SAFE_INTEGER;
			return orderA - orderB;
		});

		const query = (searchQuery || internalSearchQuery).toLowerCase();
		if (!query) return sortedResources;

		return sortedResources.filter(
			(resource) =>
				resource.name.toLowerCase().includes(query) ||
				resource.title?.toLowerCase().includes(query) ||
				resource.description?.toLowerCase().includes(query) ||
				resource.uri.toLowerCase().includes(query)
		);
	});

	let showSearchInput = $derived(resources.length > 3);
</script>

<ChatFormPickerPopover
	bind:isOpen
	class={className}
	srLabel="Open resource picker"
	{onClose}
	onKeydown={handleKeydown}
>
	<ChatFormResourcePickerList
		resources={filteredResources}
		{isLoading}
		{selectedIndex}
		bind:searchQuery={internalSearchQuery}
		{showSearchInput}
		{serverSettingsMap}
		getServerLabel={(server) => mcpStore.getServerLabel(server)}
		{isResourceAttached}
		onResourceClick={handleResourceClick}
		{onBrowse}
	/>
</ChatFormPickerPopover>
