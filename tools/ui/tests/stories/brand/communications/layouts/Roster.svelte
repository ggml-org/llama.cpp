<!--
	Roster: a headline with a motif beside it, the lede and chips, then a labelled row of cards. The cards are
	the layout's content: models in a drop, features in a release. They go one, two or three across with the format.
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
