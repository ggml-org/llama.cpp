<!--
	Lays out one component in several visual states side by side, each under a small
	label. `columns` sets a fixed column count; leave it out to fill the row with
	cells of at least `minItemWidth`.
-->
<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		children: Snippet;
		columns?: number;
		minItemWidth?: string;
		title?: string;
	}

	let { children, columns, minItemWidth = '14rem', title }: Props = $props();

	let gridTemplate = $derived(
		columns
			? `repeat(${columns}, minmax(0, 1fr))`
			: `repeat(auto-fill, minmax(${minItemWidth}, 1fr))`
	);
</script>

<section class="w-[min(72rem,calc(100vw-2rem))] p-4">
	{#if title}
		<h2 class="mb-4 text-sm font-medium text-muted-foreground">{title}</h2>
	{/if}

	<div style:grid-template-columns={gridTemplate} class="grid items-start gap-x-6 gap-y-8">
		{@render children()}
	</div>
</section>
