<!--
	E · Index card, after the references' web cards: a header bar like a site's (the lockup and the address, in a
	pill of the ground's inset), the headline at a reading size in three lines, an inset panel holding the motif, and
	one small paragraph at the bottom. A piece that reads like a page of llama.app.
-->
<script lang="ts">
	import {
		type FormatKey,
		FORMATS,
		type Ground,
		LOCKUP
	} from '../../brand/communications/posts/post';
	import PostFrame from '../../brand/communications/posts/PostFrame.svelte';
	import QuantBlocks from '../../brand/communications/posts/QuantBlocks.svelte';

	interface Props {
		body?: string;
		format: FormatKey;
		headline?: string;
		tone: Ground;
	}

	let {
		body = 'A 2-bit or 4-bit quant keeps a model’s weights in a few levels each, so it fits in the memory you have. Llama picks the one that fits your computer.',
		format,
		headline = 'The model that fits your computer, picked for you.',
		tone
	}: Props = $props();
</script>

<PostFrame format={FORMATS[format]} label="Index card, {FORMATS[format].name}, {tone}" {tone}>
	<div class="bar">
		<span class="lockup">{LOCKUP}Llama</span>

		<span class="url">llama.app</span>
	</div>

	<h2 class="headline">{headline}</h2>

	<div class="panel">
		<div class="blocks"><QuantBlocks columns={10} rows={7} /></div>
	</div>

	<p class="body">{body}</p>
</PostFrame>

<style>
	.bar {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: calc(var(--u) * 2);
		padding: calc(var(--u) * 0.5) calc(var(--u) * 0.7);
		border-radius: 999px;
		background: var(--p-inset);
	}
	.lockup {
		font: 700 calc(var(--u) * 0.72) / 1 var(--sans);
	}
	.url {
		font: 400 calc(var(--u) * 0.58) / 1 var(--mono);
	}
	.headline {
		max-width: calc(var(--u) * 19);
		margin: calc(var(--u) * 1.4) 0 0;
		font: 600 calc(var(--u) * 1.5) / 1.06 var(--sans);
		letter-spacing: -0.03em;
		text-wrap: balance;
	}
	.panel {
		display: grid;
		flex: 1;
		place-items: center;
		min-height: 0;
		margin-block: calc(var(--u) * 1.4);
		padding: calc(var(--u) * 1.2);
		border-radius: calc(var(--u) * 0.8);
		background: var(--p-inset);
	}
	.blocks {
		width: min(100%, calc(var(--u) * 16));
	}
	.body {
		max-width: calc(var(--u) * 18);
		margin: 0;
		font-size: calc(var(--u) * 0.62);
		line-height: 1.4;
		text-wrap: pretty;
	}
</style>
