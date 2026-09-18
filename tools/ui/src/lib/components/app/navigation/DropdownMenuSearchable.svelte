<script lang="ts">
	import { SearchInput } from '$lib/components/app';
	import type { Snippet } from 'svelte';

	interface Props {
		placeholder?: string;
		searchValue?: string;
		onSearchChange?: (value: string) => void;
		onSearchKeyDown?: (event: KeyboardEvent) => void;
		emptyMessage?: string;
		isEmpty?: boolean;
		/** Extra classes for the wrapper around the option list. */
		contentClass?: string;
		/** Extra classes for the search input. */
		searchClass?: string;
		children: Snippet;
		/**
		 * Optional sticky footer. It sticks to the bottom of the dropdown content's
		 * own scrollport, so it stays visible while the option list scrolls. For this
		 * to work, DropdownMenu.Content must be the scroll container (keep its
		 * overflow-y-auto and a max-height) and must not be `overflow-hidden`.
		 */
		footer?: Snippet;
	}

	let {
		children,
		contentClass = '',
		emptyMessage = 'No items found',
		footer,
		isEmpty = false,
		onSearchChange,
		onSearchKeyDown,
		placeholder = 'Search...',
		searchClass = '',
		searchValue = $bindable('')
	}: Props = $props();

	// the search sticks to the top; its height is published so
	// list section headers can stick right below it
	let stickyHeaderHeight = $state(0);
</script>

<div bind:clientHeight={stickyHeaderHeight} class="sticky top-0 z-20 bg-popover p-1.5">
	<SearchInput
		bind:value={searchValue}
		class={searchClass}
		onInput={onSearchChange}
		onKeyDown={onSearchKeyDown}
		{placeholder}
	/>
</div>

<div class={contentClass}>
	<!-- wrapper carries the sticky offset: bits-ui owns the style of the scrollport -->
	<div style="--dropdown-sticky-height: {stickyHeaderHeight}px">
		{@render children()}

		{#if isEmpty}
			<div class="px-2 py-3 text-center text-sm text-muted-foreground">{emptyMessage}</div>
		{/if}
	</div>
</div>

{#if footer}
	<div class="sticky bottom-0 z-20 bg-popover py-1.5">
		<div
			aria-orientation="horizontal"
			class="h-px bg-border/20 mb-1.5 mx-1.5"
			role="separator"
		></div>

		{@render footer()}
	</div>
{/if}
