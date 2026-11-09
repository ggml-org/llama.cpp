<!--
	B · A series of three, after the references' poster triplets: three portraits that read as one, each with its
	own job. 1 says it in a few stepped words on ink, with its place in the series as a bracketed numeral; 2 is the
	mark alone on a sky with grain; 3 explains it in one paragraph, on sand. 1 and 3 sign off with the icon over a
	hairline.
-->
<script lang="ts">
	import { type FormatKey, FORMATS, LOCKUP } from '../../brand/communications/posts/post';
	import PostFrame from '../../brand/communications/posts/PostFrame.svelte';
	import PostHeader from '../../brand/communications/posts/PostHeader.svelte';
	import Grain from './Grain.svelte';
	import Signature from './Signature.svelte';

	interface Props {
		format?: FormatKey;
		paragraph?: string;
		part: 1 | 2 | 3;
		/** the stepped words, each on its line and set in by its column */
		words?: [string, number][];
	}

	let {
		format = 'portrait',
		paragraph = 'Llama runs open models on the computer you already have. Your conversations stay on it, it works offline, and there is no account to make.',
		part,
		words = [
			['Private', 0],
			['Offline', 2],
			['Yours', 1]
		]
	}: Props = $props();

	const tone = $derived(part === 1 ? 'dark' : part === 2 ? 'night' : 'sand');
</script>

<PostFrame format={FORMATS[format]} label="Series, part {part}, {FORMATS[format].name}" {tone}>
	{#if part === 1}
		<PostHeader brand="Llama" />

		<div class="words">
			{#each words as [word, column] (word)}
				<p style:--col={column} class="word">{word}</p>
			{/each}
		</div>

		<p class="numeral end">[1]</p>

		<Signature right="llama.app" />
	{:else if part === 2}
		<div class="bleed"><Grain amount={0.45} /></div>

		<p class="lockup">{LOCKUP}Llama</p>

		<p class="url">llama.app</p>
	{:else}
		<p class="numeral">[2]</p>

		<p class="paragraph">{paragraph}</p>

		<Signature right="llama.app" />
	{/if}
</PostFrame>

<style>
	.words {
		display: grid;
		gap: calc(var(--u) * 1.2);
		margin-top: calc(var(--u) * 3);
	}
	/* a word set in by its column of the six */
	.word {
		margin: 0 0 0 calc((100% + var(--gutter)) / 6 * var(--col));
		font: 500 calc(var(--u) * 3) / 0.95 var(--sans);
		letter-spacing: -0.035em;
	}
	.numeral {
		margin: 0;
		color: var(--p-muted);
		font: 400 calc(var(--u) * 4) / 1 var(--mono);
		letter-spacing: -0.04em;
	}
	.numeral.end {
		align-self: end;
		margin-top: auto;
		margin-bottom: calc(var(--u) * 1.2);
	}
	.paragraph {
		max-width: calc(var(--u) * 19);
		margin: calc(var(--u) * 1.6) 0 0;
		font: 500 calc(var(--u) * 1.4) / 1.18 var(--sans);
		letter-spacing: -0.025em;
		text-wrap: pretty;
	}
	.bleed {
		position: absolute;
		inset: calc(var(--m) * -1);
	}
	.lockup {
		position: relative;
		margin: auto;
		font: 700 calc(var(--u) * 1.8) / 1 var(--sans);
		letter-spacing: -0.01em;
	}
	.url {
		position: relative;
		margin: 0;
		color: var(--p-muted);
		font: 400 calc(var(--u) * 0.58) / 1 var(--mono);
		text-align: center;
	}
</style>
