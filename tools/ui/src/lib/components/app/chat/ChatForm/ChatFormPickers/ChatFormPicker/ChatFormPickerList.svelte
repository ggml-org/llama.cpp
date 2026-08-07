<script lang="ts" generics="T">
	import type { Snippet } from 'svelte';
	import { SearchInput } from '$lib/components/app';
	import ScrollArea from '$lib/components/ui/scroll-area/scroll-area.svelte';
	import { useScrollActiveRow } from '$lib/hooks/use-scroll-active-row.svelte';
	import { CHAT_FORM_POPOVER_MAX_HEIGHT } from '$lib/constants';

	interface Props {
		items: T[];
		isLoading: boolean;
		selectedIndex: number;
		searchQuery: string;
		showSearchInput: boolean;
		searchPlaceholder?: string;
		// Omit to distinguish "haven't searched yet" from "search returned nothing".
		emptyMessage?: string;
		autofocus?: boolean;
		inputRef?: HTMLInputElement | null;
		onSearchClose?: () => void;
		itemKey: (item: T, index: number) => string;
		item: Snippet<[T, number, boolean]>;
		skeleton?: Snippet;
		skeletonCount?: number;
		footer?: Snippet;
		// Counter bumped by the picker on keyboard nav; scrolls the selected
		// row into view without scrolling on hover or result replacement.
		scrollTrigger?: number;
	}

	let {
		items,
		isLoading,
		selectedIndex,
		searchQuery = $bindable(),
		showSearchInput,
		searchPlaceholder = 'Search...',
		emptyMessage,
		autofocus = false,
		inputRef = $bindable(null),
		onSearchClose,
		itemKey,
		item,
		skeleton,
		skeletonCount = 6,
		footer,
		scrollTrigger
	}: Props = $props();

	let listContainer = $state<HTMLDivElement | null>(null);

	let listPaddingTop = $derived(
		showSearchInput ? (isLoading || items.length > 0 ? 'pt-13' : 'pt-10') : ''
	);

	// selectedIndex/items.length are untracked so hover and result replacement
	// never re-fire the scroll; keyboard nav is the only path that bumps the trigger.
	useScrollActiveRow({
		getTrigger: () => scrollTrigger,
		getContainer: () => listContainer,
		getIndex: () => selectedIndex,
		getCount: () => items.length,
		dataIndex: 'picker'
	});
</script>

<ScrollArea>
	{#if showSearchInput}
		<div class="absolute top-0 right-0 left-0 z-10 p-2 pb-0">
			<SearchInput
				{autofocus}
				placeholder={searchPlaceholder}
				bind:value={searchQuery}
				bind:ref={inputRef}
				onClose={onSearchClose}
			/>
		</div>
	{/if}

	<div bind:this={listContainer} class={[`${CHAT_FORM_POPOVER_MAX_HEIGHT} p-2`, listPaddingTop]}>
		{#if isLoading}
			{#if skeleton}
				{@render skeleton()}
			{:else}
				<div aria-busy="true" aria-live="polite" class="flex flex-col">
					{#each { length: skeletonCount } as _, rowIndex (rowIndex)}
						<div class="flex items-start gap-3 rounded-lg px-3 py-2">
							<div class="mt-0.5 size-4 shrink-0 animate-pulse rounded-md bg-muted/60"></div>
							<div class="flex min-w-0 flex-1 flex-col">
								<div class="h-5 w-2/5 animate-pulse rounded-sm bg-muted/60"></div>
								<div class="h-4 w-1/3 animate-pulse rounded-sm bg-muted/40"></div>
							</div>
						</div>
					{/each}
				</div>
			{/if}
		{:else if items && items.length === 0}
			{#if emptyMessage}
				<div class="py-6 text-center text-sm text-muted-foreground">{emptyMessage}</div>
			{/if}
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
