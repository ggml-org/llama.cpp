<!--
	The Typography overview's page: full-bleed, on the specimen's colours and fonts. Its sections hold their own
	1180px column, so the hero can run edge to edge.
-->
<script lang="ts">
	import '../../fonts/fonts.css';
	import '../lib/styles/specimen.css';
	import type { Snippet } from 'svelte';

	let { children }: { children: Snippet } = $props();
</script>

<div class="specimen overview">{@render children()}</div>

<style>
	.overview {
		--ui: 'Llama Sans UI', var(--body);
		--headline: 'Llama Sans Display', var(--body);
		--column: min(1180px, 100% - 2 * max(16px, 4vw));
		display: grid;
		gap: clamp(64px, 9vw, 128px);
		padding-bottom: 96px;
		font-family: var(--ui);
	}
	.overview > :global(*) {
		min-width: 0;
	}
	/* the specimen's metrics and notes step aside on its busy phone page; here they are the page */
	@media (max-width: 600px) {
		.overview :global(.metrics) {
			display: flex;
		}
		.overview :global(.notes) {
			display: grid;
		}
	}
	/* Markdown between the sections (headings and paragraphs) sits in the same column */
	.overview > :global(:is(p, h2)) {
		width: var(--column);
		margin: 0 auto;
	}
</style>
