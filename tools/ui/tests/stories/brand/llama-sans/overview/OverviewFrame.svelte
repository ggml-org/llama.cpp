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
		--code: 'Llama Mono', ui-monospace, monospace;
		--column: min(1180px, 100% - 2 * max(16px, 4vw));
		/* the overview is neutral: the design system's greys everywhere, and vermilion as the one accent, only where it
		   does a job (marking a detail, an angle, a joint). The brand orange is kept for the llama.cpp logotype */
		--vermilion: #d9553b;
		--vermilion-soft: rgb(217 85 59 / 0.12);
		--mark: var(--vermilion);
		display: grid;
		gap: clamp(64px, 9vw, 128px);
		padding-bottom: 96px;
		font-family: var(--ui);
	}
	:global(.dark) .overview {
		--vermilion: #e8735a;
		--vermilion-soft: rgb(232 115 90 / 0.16);
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
