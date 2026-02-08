<script lang="ts">
	import type { Snippet } from 'svelte';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import { cn } from '$lib/components/ui/utils';
	import { SearchInput } from '$lib/components/app';

	interface Props {
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
		placeholder?: string;
		searchValue?: string;
		onSearchChange?: (value: string) => void;
		onSearchKeyDown?: (event: KeyboardEvent) => void;
		align?: 'start' | 'center' | 'end';
		contentClass?: string;
		emptyMessage?: string;
		isEmpty?: boolean;
		disabled?: boolean;
		trigger: Snippet;
		children: Snippet;
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
	<DropdownMenu.Trigger
		{disabled}
		onclick={(e) => {
			e.preventDefault();
			e.stopPropagation();
		}}
	>
		{@render trigger()}
	</DropdownMenu.Trigger>

	<DropdownMenu.Content {align} class={cn(contentClass, 'pt-0')}>
		<div class="sticky top-0 z-10 mb-2 bg-popover p-1 pt-2">
			<SearchInput
				{placeholder}
				bind:value={searchValue}
				onInput={onSearchChange}
				onKeyDown={onSearchKeyDown}
			/>
		</div>

		<div class={cn('overflow-y-auto')}>
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
