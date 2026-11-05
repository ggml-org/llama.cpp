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
		/* the overview's palette: the design system's greys, llama.app's soft blue, a warm beige and a muted clay.
		   The brand orange is kept for the llama.cpp logotype, so here clay takes over as the one accent (--mark) */
		--beige: #efe9df;
		--beige-deep: #e3dbcc;
		--beige-ink: #2b2620;
		--clay: #a8704f;
		--clay-soft: rgb(168 112 79 / 0.14);
		--mist:
			radial-gradient(ellipse at 15% 10%, #dcecfb 0%, transparent 55%),
			radial-gradient(ellipse at 90% 95%, #e8f2fc 0%, transparent 50%),
			linear-gradient(160deg, #c4ddf6 0%, #d0e4f7 55%, #dfecf9 100%); /* llama.app's light hero, in either theme */
		--mist-ink: #0f1f33;
		--mark: var(--clay);
		display: grid;
		gap: clamp(64px, 9vw, 128px);
		padding-bottom: 96px;
		font-family: var(--ui);
	}
	:global(.dark) .overview {
		--clay: #c99572;
		--clay-soft: rgb(201 149 114 / 0.16);
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
