<!--
	Bulletin: news, several items at once. The header with a pill, the headline with the lede beside it, then a lead
	item and a column of others, and the footer. On a square or tall format the lede and the column go under.
-->
<script lang="ts">
	import { FORMATS } from '../posts/post';
	import PostFooter from '../posts/PostFooter.svelte';
	import PostFrame from '../posts/PostFrame.svelte';
	import PostHeader from '../posts/PostHeader.svelte';
	import type { LayoutProps } from './layout';
	import type { Snippet } from 'svelte';

	interface Props extends LayoutProps {
		headline: string;
		lead: Snippet;
		lede?: string;
		side?: Snippet;
	}

	let { footer, format, header, headline, lead, lede, overlay, side, tone }: Props = $props();
</script>

<PostFrame
	format={FORMATS[format]}
	label="Bulletin, {FORMATS[format].name}, {tone}"
	{overlay}
	{tone}
>
	<PostHeader {...header} />

	<div class="grid">
		<h2 class="t-headline">{headline}</h2>

		{#if lede}<p class="t-lede">{lede}</p>{/if}

		<div class="lead">{@render lead()}</div>

		{#if side}<div class="side">{@render side()}</div>{/if}
	</div>

	{#if footer}<PostFooter left={footer[0]} right={footer[1]} />{/if}
</PostFrame>

<style>
	.grid {
		display: grid;
		flex: 1;
		grid-template-rows: auto auto minmax(0, 1fr) auto;
		grid-template-columns: minmax(0, 1fr);
		min-height: 0;
		gap: calc(var(--u) * 0.9) var(--gutter);
		margin-block: calc(var(--u) * 1.2) calc(var(--u) * 0.8);
	}
	.t-headline {
		margin-bottom: calc(var(--stack) - var(--u) * 0.9);
	}
	.lead,
	.side {
		display: grid;
		gap: var(--gutter);
	}
	.side {
		grid-template-columns: repeat(auto-fit, minmax(0, 1fr));
		grid-auto-flow: column;
	}

	/* on a square format the lead and the side column sit side by side under the headline and lede, the side's items
	   stacked: one under the other they would leave the lead no room */
	:global(.shape-square) .grid {
		grid-template-rows: auto auto minmax(0, 1fr);
		grid-template-columns: repeat(var(--cols), minmax(0, 1fr));
	}
	:global(.shape-square) :is(.t-headline, .t-lede) {
		grid-column: 1 / -1;
	}
	:global(.shape-square) .lead {
		grid-column: 1 / span 4;
	}
	:global(.shape-square) .side {
		grid-column: 5 / -1;
		grid-auto-flow: row;
		grid-template-columns: minmax(0, 1fr);
	}
	:global(.shape-square) .grid:not(:has(.side)) .lead {
		grid-column: 1 / -1;
	}

	/* a wide format is short: a unit over the grid, as on a Statement, and less between its rows */
	:global(.shape-wide) .grid {
		grid-template-rows: auto minmax(0, 1fr);
		grid-template-columns: repeat(var(--cols), minmax(0, 1fr));
		align-items: center;
		row-gap: calc(var(--u) * 0.8);
		margin-top: calc(var(--u) * 1);
	}
	:global(.shape-wide) .t-headline {
		grid-column: 1 / 8;
		margin: 0;
	}
	:global(.shape-wide) .t-lede {
		grid-column: 8 / -1;
	}
	:global(.shape-wide) .lead {
		grid-column: 1 / 8;
		align-self: stretch;
	}
	/* with no lede beside the headline, the side column runs from the headline's top to the lead's bottom */
	:global(.shape-wide) .grid:not(:has(.t-lede)) .side {
		grid-row: 1 / span 2;
	}
	:global(.shape-wide) .side {
		grid-column: 8 / -1;
		grid-auto-flow: row;
		grid-template-columns: minmax(0, 1fr);
		align-self: stretch;
	}
</style>
