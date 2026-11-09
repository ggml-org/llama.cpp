<!--
	A · Field cards, after the references' project cards: each product a tall card whose whole face is one field of
	artwork (a sky with grain, or a quantised sky), the icon centred on it, and its name and one line at the bottom
	left, in the field's own ink. Above the cards, a small title and a two-line lede, set to the left.
-->
<script lang="ts">
	import {
		type FormatKey,
		FORMATS,
		type Ground,
		ICON
	} from '../../brand/communications/posts/post';
	import PostFrame from '../../brand/communications/posts/PostFrame.svelte';
	import PostHeader from '../../brand/communications/posts/PostHeader.svelte';
	import Grain from './Grain.svelte';
	import QuantSky from './QuantSky.svelte';

	interface Card {
		field: 'day' | 'night' | 'quant';
		line: string;
		name: string;
	}

	interface Props {
		cards?: Card[];
		format: FormatKey;
		lede?: string;
		title?: string;
		tone: Ground;
	}

	let {
		cards = [
			{ field: 'night', line: 'The app. It picks a model that fits your computer.', name: 'Llama' },
			{
				field: 'quant',
				line: 'The engine. Open models on the hardware you have.',
				name: 'llama.cpp'
			},
			{ field: 'day', line: 'The chat, in a browser on any of your devices.', name: 'Llama Chat' }
		],
		format,
		lede = 'An app, an engine and a chat,\nall running on your own computer.',
		title = 'The family',
		tone
	}: Props = $props();
</script>

<PostFrame format={FORMATS[format]} label="Field cards, {FORMATS[format].name}, {tone}" {tone}>
	<PostHeader brand="Llama" url="llama.app" />

	<div class="intro">
		<p class="title">{title}</p>

		<p class="t-lede">{lede}</p>
	</div>

	<div style:--n={cards.length} class="cards">
		{#each cards as card (card.name)}
			<figure class="card ground {card.field === 'day' ? 'ground-day' : 'ground-night'}">
				{#if card.field === 'quant'}
					<div class="art"><QuantSky columns={6} rows={9} sky="night" /></div>
				{:else}
					<div class="art"><Grain /></div>
				{/if}

				<span aria-hidden="true" class="mark">{ICON}</span>

				<figcaption>
					<span class="name">{card.name}</span>

					<span class="line">{card.line}</span>
				</figcaption>
			</figure>
		{/each}
	</div>
</PostFrame>

<style>
	.intro {
		display: grid;
		gap: calc(var(--u) * 0.6);
		margin-block: calc(var(--u) * 1.6) calc(var(--u) * 1.2);
	}
	.title {
		margin: 0;
		font: 600 calc(var(--u) * 0.84) / 1.1 var(--sans);
		letter-spacing: -0.01em;
	}
	.intro .t-lede {
		white-space: pre-line;
	}
	.cards {
		display: grid;
		flex: 1;
		grid-template-columns: repeat(var(--n), minmax(0, 1fr));
		gap: calc(var(--u) * 0.5);
		min-height: 0;
	}
	/* a tall format stacks the cards' row into one: the first card fills the piece */
	:global(.shape-tall) .cards,
	:global(.shape-square) .cards {
		grid-template-columns: minmax(0, 1fr);
	}
	:global(.shape-tall) .card:not(:first-child),
	:global(.shape-square) .card:not(:first-child) {
		display: none;
	}
	.card {
		position: relative;
		display: grid;
		place-items: center;
		margin: 0;
		padding-bottom: calc(var(--u) * 2.4);
		overflow: hidden;
		border-radius: calc(var(--u) * 0.8);
	}
	.art {
		position: absolute;
		inset: 0;
	}
	.mark {
		position: relative;
		font: 700 calc(var(--u) * 3.6) / 1 var(--sans);
	}
	:global(.shape-wide) .mark {
		font-size: calc(var(--u) * 2.4);
		color: var(--p-ink);
	}
	figcaption {
		position: absolute;
		inset: auto calc(var(--u) * 0.8) calc(var(--u) * 0.8);
		display: grid;
		gap: calc(var(--u) * 0.3);
	}
	.name {
		font: 600 calc(var(--u) * 0.84) / 1.1 var(--sans);
		letter-spacing: -0.01em;
	}
	.line {
		max-width: calc(var(--u) * 12);
		color: var(--p-muted);
		font-size: calc(var(--u) * 0.62);
		line-height: 1.35;
	}
</style>
