<!--
	Roster: a headline, the lede with a motif beside it and the chips under them (on a wide format, the motif beside
	the whole block), then a labelled row of cards. The cards are the layout's content: models in a drop, features in
	a release. They go one, two or three across with the format.
-->
<script lang="ts">
	import HeadlineBlock from '../posts/HeadlineBlock.svelte';
	import { FORMATS } from '../posts/post';
	import PostFooter from '../posts/PostFooter.svelte';
	import PostFrame from '../posts/PostFrame.svelte';
	import PostHeader from '../posts/PostHeader.svelte';
	import type { LayoutProps } from './layout';
	import type { Snippet } from 'svelte';

	interface Props extends LayoutProps {
		cards: Snippet;
		chips?: string[];
		headline: string;
		/** the label over the cards */
		label?: string;
		lede?: string;
		motif?: Snippet;
	}

	let { cards, chips, footer, format, header, headline, label, lede, motif, overlay, tone }: Props =
		$props();
</script>

<PostFrame format={FORMATS[format]} label="Roster, {FORMATS[format].name}, {tone}" {overlay} {tone}>
	<PostHeader {...header} />

	<div class:plain={!motif} class="top">
		<HeadlineBlock {chips} {headline} {lede} />

		{#if motif}<div class="motif fill">{@render motif()}</div>{/if}
	</div>

	{#if label}<p class="t-label">{label}</p>{/if}

	<div class="cards">{@render cards()}</div>

	{#if footer}<PostFooter left={footer[0]} right={footer[1]} />{/if}
</PostFrame>

<style>
	.top {
		display: grid;
		grid-template-columns: repeat(var(--cols), minmax(0, 1fr));
		align-items: center;
		column-gap: var(--gutter);
		margin-top: calc(var(--u) * 1.2);
	}
	.top > :global(.block) {
		grid-column: 1 / span 4;
	}
	.plain > :global(.block) {
		grid-column: 1 / -1;
	}
	.motif {
		grid-column: 5 / -1;
		align-self: stretch;
	}
	/* on a square or tall format with a lede, the headline and the chips take the full width, so neither wraps more
	   than it must, and the lede goes between them beside the motif (4.5u high at the least): the block's parts are
	   laid out in this grid, at the text stack's rhythm. Without a lede, the motif stays beside the block */
	:global(:is(.shape-square, .shape-tall)) .top:has(:global(.t-lede)):not(.plain) {
		row-gap: var(--stack);
	}
	:global(:is(.shape-square, .shape-tall))
		.top:has(:global(.t-lede)):not(.plain)
		> :global(.block) {
		display: contents;
	}
	:global(:is(.shape-square, .shape-tall))
		.top:has(:global(.t-lede)):not(.plain)
		:global(:is(.t-headline, .chips)) {
		grid-column: 1 / -1;
	}
	:global(:is(.shape-square, .shape-tall)) .top:has(:global(.t-lede)):not(.plain) :global(.t-lede) {
		grid-column: 1 / span 4;
	}
	:global(:is(.shape-square, .shape-tall)) .top:has(:global(.t-lede)) .motif {
		grid-row: 2;
		min-height: calc(var(--u) * 4.5);
	}
	.t-label {
		margin-top: auto;
		padding-top: calc(var(--u) * 0.8);
		margin-bottom: calc(var(--u) * 0.45);
	}
	.cards {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: var(--gutter);
	}
	.cards:only-child,
	:global(.shape-tall) .cards {
		grid-template-columns: minmax(0, 1fr);
	}
	:global(.shape-wide) .top > :global(.block) {
		grid-column: 1 / span 8;
	}
	:global(.shape-wide) .motif {
		grid-column: 9 / -1;
	}
	:global(.shape-wide) .cards {
		grid-template-columns: repeat(3, minmax(0, 1fr));
	}
	:global(.shape-wide) .top {
		margin-top: calc(var(--u) * 1);
	}
</style>
