<!--
	Product: the headline and the app side by side. On a wide format the text takes the left half and the app the
	right; on a square or tall one the app sits under the text and runs off the bottom edge. A caption chip closes the
	text, where the platforms crop least.
-->
<script lang="ts">
	import { FORMATS, grid } from '../posts/post';
	import PostFrame from '../posts/PostFrame.svelte';
	import PostHeader from '../posts/PostHeader.svelte';
	import ProductShot from '../posts/ProductShot.svelte';
	import type { LayoutProps } from './layout';

	interface Props extends LayoutProps {
		caption?: string;
		headline: string;
		lede?: string;
	}

	let { caption, format, header, headline, lede, overlay, tone }: Props = $props();

	const g = $derived(grid(FORMATS[format]));
	// the app's width in units: on a wide format, as much as the height allows (the panel is 1.3 times as high as wide)
	const shot = $derived(
		g.shape === 'wide'
			? Math.min(15, (FORMATS[format].height / g.unit - 4) / 1.3)
			: g.shape === 'tall'
				? 17
				: 14
	);
</script>

<PostFrame
	format={FORMATS[format]}
	label="Product, {FORMATS[format].name}, {tone}"
	{overlay}
	{tone}
>
	<div class="body">
		<div class="text">
			<PostHeader {...header} size={g.shape === 'wide' ? 1.2 : 0.9} />

			<h2 class="t-display">{headline}</h2>

			{#if lede}<p class="t-lede">{lede}</p>{/if}

			{#if caption}<p class="caption">{caption}</p>{/if}
		</div>

		<div class="shot">
			<ProductShot size={shot} unit={g.unit} />
		</div>
	</div>
</PostFrame>

<style>
	.body {
		display: grid;
		flex: 1;
		grid-template-rows: auto 1fr;
		gap: calc(var(--u) * 2);
	}
	.text {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: calc(var(--u) * 0.9);
	}
	.text :global(.t-display) {
		margin-top: calc(var(--u) * 1.2);
	}
	.caption {
		margin: auto 0 0;
		padding: calc(var(--u) * 0.3) calc(var(--u) * 0.55);
		border-radius: calc(var(--u) * 0.3);
		background: var(--p-solid);
		color: var(--p-solid-ink);
		font-size: calc(var(--u) * 0.9);
		line-height: 1.2;
	}
	.shot {
		justify-self: end;
	}
	/* on a square or tall format the app runs off the bottom edge */
	:global(:is(.shape-square, .shape-tall)) .shot {
		align-self: end;
		margin-bottom: calc(-1 * (var(--m) + var(--safe-bottom)) - var(--u) * 4);
	}
	:global(:is(.shape-square, .shape-tall)) .caption {
		margin-top: calc(var(--u) * 0.5);
		font-size: calc(var(--u) * 0.6);
	}

	:global(.shape-wide) .body {
		grid-template-rows: none;
		grid-template-columns: repeat(var(--cols), minmax(0, 1fr));
		column-gap: var(--gutter);
	}
	:global(.shape-wide) .text {
		grid-column: 1 / 8;
	}
	:global(.shape-wide) .shot {
		grid-column: 8 / -1;
		align-self: start;
	}
</style>
