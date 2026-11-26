<!--
	The scrolling body of a sheet with pages: it shows the page on top of a `useSheetPages` stack and
	slides between pages as they are pushed and popped.
-->
<script generics="P" lang="ts">
	import type { Snippet } from 'svelte';
	import { cn } from '../../ui/utils';
	import type { SheetPagesState } from './sheet-pages.svelte';

	interface Props {
		children: Snippet;
		/** Classes of each page: its padding and the gap between its groups. */
		class?: string;
		/** Keep the first page's height as the floor of the pages pushed over it, so a short sheet does not jump. */
		keepFirstPageHeight?: boolean;
		/** Identifies the page shown; a change of it slides to the new page. */
		pageKey: string;
		pages: SheetPagesState<P>;
		/** The scrolling element, to read or set its scroll position. */
		ref?: HTMLDivElement | null;
		scrollClass?: string;
	}

	let {
		children,
		class: className,
		keepFirstPageHeight = false,
		pageKey,
		pages,
		ref = $bindable(null),
		scrollClass
	}: Props = $props();

	let firstPageHeight = $state(0);

	let minHeight = $derived(
		keepFirstPageHeight && pages.canGoBack && firstPageHeight ? `${firstPageHeight}px` : undefined
	);

	function slide(node: Element, params: { incoming: boolean }) {
		return pages.transition(node, params);
	}
</script>

<div
	bind:this={ref}
	style:min-height={minHeight}
	class={cn(
		'grid min-h-0 flex-1 overflow-x-hidden overflow-y-auto overscroll-contain',
		scrollClass
	)}
>
	{#key pageKey}
		<div
			bind:clientHeight={
				() => firstPageHeight,
				(height) => {
					if (!pages.canGoBack && height) firstPageHeight = height;
				}
			}
			in:slide={{ incoming: true }}
			out:slide={{ incoming: false }}
			class={cn('col-start-1 row-start-1 flex min-w-0 flex-col', className)}
		>
			{@render children()}
		</div>
	{/key}
</div>
