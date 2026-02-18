<script lang="ts">
	import { FolderOpen } from '@lucide/svelte';
	import type { MCPResourceInfo, MCPServerSettingsEntry } from '$lib/types';
	import { SearchInput } from '$lib/components/app';
	import {
		ChatFormResourcePickerListItem,
		ChatFormResourcePickerListItemSkeleton
	} from '$lib/components/app/chat';
	import { SvelteMap } from 'svelte/reactivity';
	import ScrollArea from '$lib/components/ui/scroll-area/scroll-area.svelte';
	import { Button } from '$lib/components/ui/button';
	import { CHAT_FORM_POPOVER_MAX_HEIGHT } from '$lib/constants/css-classes';

	interface Props {
		resources: MCPResourceInfo[];
		isLoading: boolean;
		selectedIndex: number;
		searchQuery: string;
		showSearchInput: boolean;
		serverSettingsMap: SvelteMap<string, MCPServerSettingsEntry>;
		getServerLabel: (server: MCPServerSettingsEntry) => string;
		isResourceAttached: (uri: string) => boolean;
		onResourceClick: (resource: MCPResourceInfo) => void;
		onBrowse?: () => void;
	}

	let {
		resources,
		isLoading,
		selectedIndex,
		searchQuery = $bindable(),
		showSearchInput,
		serverSettingsMap,
		getServerLabel,
		isResourceAttached,
		onResourceClick,
		onBrowse
	}: Props = $props();

	let listContainer = $state<HTMLDivElement | null>(null);

	$effect(() => {
		if (listContainer && selectedIndex >= 0 && selectedIndex < resources.length) {
			const selectedElement = listContainer.querySelector(
				`[data-resource-index="${selectedIndex}"]`
			) as HTMLElement;

			if (selectedElement) {
				selectedElement.scrollIntoView({
					behavior: 'smooth',
					block: 'center',
					inline: 'nearest'
				});
			}
		}
	});
</script>

<ScrollArea>
	{#if showSearchInput}
		<div class="absolute top-0 right-0 left-0 z-10 p-2 pb-0">
			<SearchInput placeholder="Search resources..." bind:value={searchQuery} />
		</div>
	{/if}

	<div
		bind:this={listContainer}
		class="{CHAT_FORM_POPOVER_MAX_HEIGHT} p-2"
		class:pt-13={showSearchInput}
	>
		{#if isLoading}
			<ChatFormResourcePickerListItemSkeleton />
		{:else if resources.length === 0}
			<div class="py-6 text-center text-sm text-muted-foreground">No MCP resources available</div>
		{:else}
			{#each resources as resource, index (resource.serverName + ':' + resource.uri)}
				{@const server = serverSettingsMap.get(resource.serverName)}
				{@const serverLabel = server ? getServerLabel(server) : resource.serverName}

				<ChatFormResourcePickerListItem
					data-resource-index={index}
					{resource}
					{server}
					{serverLabel}
					isSelected={index === selectedIndex}
					isAttached={isResourceAttached(resource.uri)}
					onClick={() => onResourceClick(resource)}
				/>
			{/each}
		{/if}
	</div>

	{#if onBrowse && resources.length > 3}
		<Button
			class="fixed right-3 bottom-3"
			type="button"
			onclick={onBrowse}
			variant="secondary"
			size="sm"
		>
			<FolderOpen class="h-3 w-3" />

			Browse all
		</Button>
	{/if}
</ScrollArea>
