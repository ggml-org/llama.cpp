<script lang="ts">
	import type { MCPPromptInfo, MCPServerSettingsEntry } from '$lib/types';
	import { SearchInput } from '$lib/components/app';
	import ChatFormPromptPickerListItem from './ChatFormPromptPickerListItem.svelte';
	import ChatFormPromptPickerListItemSkeleton from './ChatFormPromptPickerListItemSkeleton.svelte';
	import { SvelteMap } from 'svelte/reactivity';
	import ScrollArea from '$lib/components/ui/scroll-area/scroll-area.svelte';

	interface Props {
		prompts: MCPPromptInfo[];
		isLoading: boolean;
		selectedIndex: number;
		searchQuery: string;
		showSearchInput: boolean;
		serverSettingsMap: SvelteMap<string, MCPServerSettingsEntry>;
		getServerLabel: (server: MCPServerSettingsEntry) => string;
		onPromptClick: (prompt: MCPPromptInfo) => void;
	}

	let {
		prompts,
		isLoading,
		selectedIndex,
		searchQuery = $bindable(),
		showSearchInput,
		serverSettingsMap,
		getServerLabel,
		onPromptClick
	}: Props = $props();
</script>

<ScrollArea>
	{#if showSearchInput}
		<div class="absolute top-0 right-0 left-0 z-10 p-2 pb-0">
			<SearchInput placeholder="Search prompts..." bind:value={searchQuery} />
		</div>
	{/if}

	<div class="max-h-64 p-2" class:pt-13={showSearchInput}>
		{#if isLoading}
			<ChatFormPromptPickerListItemSkeleton />
		{:else if prompts.length === 0}
			<div class="py-6 text-center text-sm text-muted-foreground">
				{prompts.length === 0 ? 'No MCP prompts available' : 'No prompts found'}
			</div>
		{:else}
			{#each prompts as prompt, index (prompt.serverName + ':' + prompt.name)}
				{@const server = serverSettingsMap.get(prompt.serverName)}
				{@const serverLabel = server ? getServerLabel(server) : prompt.serverName}

				<ChatFormPromptPickerListItem
					{prompt}
					{server}
					{serverLabel}
					isSelected={index === selectedIndex}
					onClick={() => onPromptClick(prompt)}
				/>
			{/each}
		{/if}
	</div>
</ScrollArea>
