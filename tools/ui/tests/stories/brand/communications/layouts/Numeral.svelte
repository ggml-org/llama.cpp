<!--
	Numeral: one figure carries the piece. The number huge in cinder over a dot grid, its label, and a headline and lede
	that say what it means: beside it on a wide format, under it on a square or tall one.
-->
<script lang="ts">
	import DotGrid from '../posts/DotGrid.svelte';
	import Figure from '../posts/Figure.svelte';
	import { FORMATS, grid } from '../posts/post';
	import PostFooter from '../posts/PostFooter.svelte';
	import PostFrame from '../posts/PostFrame.svelte';
	import PostHeader from '../posts/PostHeader.svelte';
	import type { LayoutProps } from './layout';

	interface Props extends LayoutProps {
		/** what the figure counts */
		figureLabel: string;
		headline: string;
		lede?: string;
		value: string;
	}

	let { figureLabel, footer, format, header, headline, lede, overlay, tone, value }: Props =
		$props();

	const g = $derived(grid(FORMATS[format]));
	// the figure's size in units, less on a format too low for it (a billboard is 16 units high)
	const figure = $derived(
		Math.min({ square: 6, tall: 9, wide: 8 }[g.shape], FORMATS[format].height / g.unit - 10)
	);
</script>

<PostFrame
	format={FORMATS[format]}
	label="Numeral, {FORMATS[format].name}, {tone}"
	{overlay}
	{tone}
>
	<PostHeader {...header} />

	<div class="body">
		<div class="figure">
			<div class="field"><DotGrid at={null} /></div>

			<Figure accent label={figureLabel} size={figure} {value} />
		</div>

		<div class="text">
			<h2 class="t-headline">{headline}</h2>

			{#if lede}<p class="t-lede">{lede}</p>{/if}
		</div>
	</div>

	{#if footer}<PostFooter left={footer[0]} right={footer[1]} />{/if}
</PostFrame>

<style>
	.body {
		display: grid;
		flex: 1;
		grid-template-rows: 1fr auto;
		gap: calc(var(--u) * 1.2);
		margin-block: calc(var(--u) * 1.5);
	}
	.figure {
		position: relative;
		display: grid;
		align-items: end;
		padding: calc(var(--u) * 0.5) 0 0;
	}
	.field {
		position: absolute;
		inset: 0 0 calc(var(--u) * 1.6) 0;
		opacity: 0.6;
	}
	.figure > :global(.figure) {
		position: relative;
	}
	.text {
		display: grid;
		gap: var(--stack);
	}

	:global(.shape-wide) .body {
		grid-template-rows: none;
		grid-template-columns: repeat(var(--cols), minmax(0, 1fr));
		column-gap: var(--gutter);
	}
	:global(.shape-wide) .figure {
		grid-column: 1 / 7;
	}
	:global(.shape-wide) .text {
		grid-column: 8 / -1;
		align-self: end;
	}
</style>
