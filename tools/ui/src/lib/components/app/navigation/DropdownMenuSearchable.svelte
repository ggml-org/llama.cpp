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
</script>

<!-- Sticks to the top of the dropdown content's scrollport. -->
<div class="sticky top-0 z-20 p-1.5">
	<SearchInput
		bind:value={searchValue}
		class={searchClass}
		onInput={onSearchChange}
		onKeyDown={onSearchKeyDown}
		{placeholder}
	/>
</div>

<div class={contentClass}>
	{@render children()}

	{#if isEmpty}
		<div class="px-2 py-3 text-center text-sm text-muted-foreground">{emptyMessage}</div>
	{/if}
</div>

{#if footer}
	<!-- Sticks to the bottom of the dropdown content's scrollport. -->
	<div class="sticky bottom-0 z-20 bg-popover py-1.5">
		<div class="h-px bg-border/20 mb-1.5 mx-1.5"></div>

		{@render footer()}
	</div>
{/if}
