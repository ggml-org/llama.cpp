<script lang="ts">
	import type { Snippet } from 'svelte';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import { cn } from '$lib/components/ui/utils';
	import { SearchInput } from '$lib/components/app';

	interface Props {
		/** Controlled open state */
		open?: boolean;
		/** Callback when open state changes */
		onOpenChange?: (open: boolean) => void;
		/** Search input placeholder */
		placeholder?: string;
		/** Search input value (bindable) */
		searchValue?: string;
		/** Callback when search value changes */
		onSearchChange?: (value: string) => void;
		/** Callback for search input keydown events */
		onSearchKeyDown?: (event: KeyboardEvent) => void;
		/** Content alignment */
		align?: 'start' | 'center' | 'end';
		/** Content width class */
		contentClass?: string;
		/** Empty state message */
		emptyMessage?: string;
		/** Whether to show empty state */
		isEmpty?: boolean;
		/** Whether the trigger is disabled */
		disabled?: boolean;
		/** Trigger content */
		trigger: Snippet;
		/** List items content */
		children: Snippet;
		/** Optional footer content */
		footer?: Snippet;
	}

	let {
		open = $bindable(false),
		onOpenChange,
		placeholder = 'Search...',
		searchValue = $bindable(''),
		onSearchChange,
		onSearchKeyDown,
		align = 'start',
		contentClass = 'w-72',
		emptyMessage = 'No items found',
		isEmpty = false,
		disabled = false,
		trigger,
		children,
		footer
	}: Props = $props();

	function handleOpenChange(newOpen: boolean) {
		open = newOpen;

		if (!newOpen) {
			searchValue = '';
			onSearchChange?.('');
		}

		onOpenChange?.(newOpen);
	}
</script>

<DropdownMenu.Root bind:open onOpenChange={handleOpenChange}>
	<DropdownMenu.Trigger {disabled}>
		{@render trigger()}
	</DropdownMenu.Trigger>

	<DropdownMenu.Content {align} class={cn(contentClass)}>
		<div class="mb-2 p-1">
			<SearchInput
				{placeholder}
				bind:value={searchValue}
				onInput={onSearchChange}
				onKeyDown={onSearchKeyDown}
			/>
		</div>

		<div class={cn('overflow-y-auto', 'max-h-[--bits-dropdown-menu-content-available-height]')}>
			{@render children()}

			{#if isEmpty}
				<div class="px-2 py-3 text-center text-sm text-muted-foreground">{emptyMessage}</div>
			{/if}
		</div>

		{#if footer}
			<DropdownMenu.Separator />

			{@render footer()}
		{/if}
	</DropdownMenu.Content>
</DropdownMenu.Root>
