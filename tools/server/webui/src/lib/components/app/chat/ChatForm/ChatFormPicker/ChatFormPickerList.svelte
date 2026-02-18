<script lang="ts" generics="T">
	import type { Snippet } from 'svelte';
	import { SearchInput } from '$lib/components/app';
	import ScrollArea from '$lib/components/ui/scroll-area/scroll-area.svelte';
	import { CHAT_FORM_POPOVER_MAX_HEIGHT } from '$lib/constants/css-classes';

	interface Props {
		items: T[];
		isLoading: boolean;
		selectedIndex: number;
		searchQuery: string;
		showSearchInput: boolean;
		searchPlaceholder?: string;
		emptyMessage?: string;
		itemKey: (item: T, index: number) => string;
		item: Snippet<[T, number, boolean]>;
		skeleton?: Snippet;
		footer?: Snippet;
	}

	let {
		items,
		isLoading,
		selectedIndex,
		searchQuery = $bindable(),
		showSearchInput,
		searchPlaceholder = 'Search...',
		emptyMessage = 'No items available',
		itemKey,
		item,
		skeleton,
		footer
	}: Props = $props();

	let listContainer = $state<HTMLDivElement | null>(null);

	$effect(() => {
		if (listContainer && selectedIndex >= 0 && selectedIndex < items.length) {
			const selectedElement = listContainer.querySelector(
				`[data-picker-index="${selectedIndex}"]`
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
			<SearchInput placeholder={searchPlaceholder} bind:value={searchQuery} />
		</div>
	{/if}

	<div
		bind:this={listContainer}
		class="{CHAT_FORM_POPOVER_MAX_HEIGHT} p-2"
		class:pt-13={showSearchInput}
	>
		{#if isLoading}
			{#if skeleton}
				{@render skeleton()}
			{/if}
		{:else if items.length === 0}
			<div class="py-6 text-center text-sm text-muted-foreground">{emptyMessage}</div>
		{:else}
			{#each items as itemData, index (itemKey(itemData, index))}
				{@render item(itemData, index, index === selectedIndex)}
			{/each}
		{/if}
	</div>

	{#if footer}
		{@render footer()}
	{/if}
</ScrollArea>
