<script lang="ts">
	import { SearchInput } from '$lib/components/app';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import type { Snippet } from 'svelte';

	interface Props {
		placeholder?: string;
		searchValue?: string;
		onSearchChange?: (value: string) => void;
		onSearchKeyDown?: (event: KeyboardEvent) => void;
		emptyMessage?: string;
		isEmpty?: boolean;
		children: Snippet;
		footer?: Snippet;
	}

	let {
		children,
		emptyMessage = 'No items found',
		footer,
		isEmpty = false,
		onSearchChange,
		onSearchKeyDown,
		placeholder = 'Search...',
		searchValue = $bindable('')
	}: Props = $props();
</script>

<div class="sticky top-0 z-10 mb-2 bg-popover p-1 pt-2">
	<SearchInput
		{placeholder}
		bind:value={searchValue}
		onInput={onSearchChange}
		onKeyDown={onSearchKeyDown}
	/>
</div>

<div class="overflow-y-auto">
	{@render children()}

	{#if isEmpty}
		<div class="px-2 py-3 text-center text-sm text-muted-foreground">{emptyMessage}</div>
	{/if}
</div>

{#if footer}
	<DropdownMenu.Separator />

	{@render footer()}
{/if}
