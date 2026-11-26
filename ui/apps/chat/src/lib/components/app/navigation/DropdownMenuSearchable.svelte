<script lang="ts">
	import { cn } from '@ggml-org/llama-ui/utils';
	import { SearchInput } from '#lib/components/app/index.ts';
	import type { Snippet } from 'svelte';

	interface Props {
		/** Extra classes for the sticky search header. */
		headerClass?: string;
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
		 * Optional footer. By default it sticks to the bottom of the parent's scrollport
		 * (overflow-y-auto + max-height), see `scrollport`.
		 */
		footer?: Snippet;
		/**
		 * Scroll the header and the list in a box of their own, with the footer under it
		 * instead of sticking. For a parent that scales as it opens (a dropdown's zoom-in):
		 * WebKit pins a bottom-sticky element against the scaled box, so the footer would
		 * ride up the list until the animation ends. The parent is then a flex column that
		 * does not scroll itself (flex flex-col overflow-hidden + max-height).
		 */
		scrollport?: boolean;
	}

	let {
		children,
		contentClass = '',
		emptyMessage = 'No items found',
		footer,
		headerClass = '',
		isEmpty = false,
		onSearchChange,
		onSearchKeyDown,
		placeholder = 'Search...',
		scrollport = false,
		searchClass = '',
		searchValue = $bindable('')
	}: Props = $props();

	// the search height is published so list headers can stick right below it
	let stickyHeaderHeight = $state(0);
</script>

{#snippet body()}
	<div
		bind:clientHeight={stickyHeaderHeight}
		class={cn('sticky top-0 z-20 bg-popover p-1.5', headerClass)}
	>
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
				<div class="px-2 py-3 text-center text-body-sm text-muted-foreground">{emptyMessage}</div>
			{/if}
		</div>
	</div>
{/snippet}

{#if scrollport}
	<div class="min-h-0 flex-1 overflow-x-hidden overflow-y-auto">
		{@render body()}
	</div>
{:else}
	{@render body()}
{/if}

{#if footer}
	<div class={['z-20 bg-popover py-1.5', scrollport ? 'shrink-0' : 'sticky bottom-0']}>
		<div
			aria-orientation="horizontal"
			class="h-px bg-border/20 mb-1.5 mx-1.5"
			role="separator"
		></div>

		{@render footer()}
	</div>
{/if}
