<script lang="ts">
	import { DropdownMenuSearchable } from '$lib/components/app';
	import { Drawer, DrawerContent } from '$lib/components/ui/drawer';
	import type { Snippet } from 'svelte';

	interface Props {
		children: Snippet;
		emptyMessage?: string;
		/** Sticky picker footer: it sticks to the bottom of the drawer's scrollport. */
		footer?: Snippet;
		isEmpty?: boolean;
		onOpenChange?: (open: boolean) => void;
		onSearchChange?: (value: string) => void;
		onSearchKeyDown?: (event: KeyboardEvent) => void;
		open?: boolean;
		placeholder?: string;
		searchTerm?: string;
	}

	let {
		children,
		emptyMessage = 'No items found',
		footer,
		isEmpty = false,
		onOpenChange,
		onSearchChange,
		onSearchKeyDown,
		open = $bindable(false),
		placeholder = 'Search models...',
		searchTerm = ''
	}: Props = $props();
</script>

<Drawer bind:open {onOpenChange}>
	<DrawerContent class="overflow-hidden">
		<!-- the wrapper is the scrollport: the search header and the footer stick to it -->
		<div class="max-h-[min(40rem,calc(100dvh-5rem))] min-h-0 overflow-y-auto">
			<DropdownMenuSearchable
				{emptyMessage}
				{footer}
				headerClass="p-2.5 pt-4"
				{isEmpty}
				{onSearchChange}
				{onSearchKeyDown}
				{placeholder}
				searchClass="bg-transparent"
				searchValue={searchTerm}
			>
				{@render children()}
			</DropdownMenuSearchable>
		</div>
	</DrawerContent>
</Drawer>
