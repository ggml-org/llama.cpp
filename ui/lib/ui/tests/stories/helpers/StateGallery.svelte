<!--
	Lays out one component in several visual states side by side, each under a small
	label. `columns` sets a fixed column count; leave it out to fill the row with
	cells of at least `minItemWidth`, or of exactly `itemWidth`. Either way a cell
	narrows to the gallery's width on a narrower screen, one to a row. `width` caps
	the gallery (72rem by default).
-->
<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		children: Snippet;
		columns?: number;
		/** a fixed cell width: as many cells to a row as fit, each this wide */
		itemWidth?: string;
		minItemWidth?: string;
		title?: string;
		width?: string;
	}

	let {
		children,
		columns,
		itemWidth,
		minItemWidth = '14rem',
		title,
		width = '72rem'
	}: Props = $props();

	let gridTemplate = $derived(
		columns
			? `repeat(${columns}, minmax(0, 1fr))`
			: itemWidth
				? `repeat(auto-fill, minmax(min(${itemWidth}, 100%), ${itemWidth}))`
				: `repeat(auto-fill, minmax(min(${minItemWidth}, 100%), 1fr))`
	);
</script>

<section style:width="min({width}, calc(100vw - 2rem))" class="p-4">
	{#if title}
		<h2 class="mb-4 text-sm font-medium text-muted-foreground">{title}</h2>
	{/if}

	<!-- a query container, so a cell can take the gallery's whole width (StateGalleryItem's `full`) -->
	<div
		style:container-type="inline-size"
		style:grid-template-columns={gridTemplate}
		class="grid items-start gap-x-6 gap-y-8"
	>
		{@render children()}
	</div>
</section>
