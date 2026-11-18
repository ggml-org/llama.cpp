<!-- A figure on the editorial page: a quiet panel, wider than the text, with a numbered caption under it -->
<script lang="ts">
	import type { Snippet } from 'svelte';

	let {
		caption,
		children,
		n,
		plain = false
	}: { caption: string; children: Snippet; n: number; plain?: boolean } = $props();
</script>

<figure class="wide">
	<div class={['panel', { plain }]}>{@render children()}</div>

	<figcaption>
		<span class="n">Fig. {n}</span>
		{caption}
	</figcaption>
</figure>

<style>
	figure {
		margin-block: clamp(32px, 5vw, 56px) !important;
	}

	/* muted text a step darker than the page's, so it keeps 4.5:1 on the panel's tint */
	.panel {
		--figure-muted: color-mix(in oklab, var(--fg-muted) 75%, var(--fg-default));
		display: grid;
		padding: clamp(20px, 4vw, 48px);
		border: 1px solid var(--line-default);
		border-radius: 14px;
		background: color-mix(in oklab, var(--fg-default) 2.5%, var(--surface-page));
		font-size: 16px;
		line-height: 1.5;
	}

	.panel.plain {
		padding: 0;
		border: 0;
		background: none;
	}

	figcaption {
		max-width: 40rem;
		margin-top: 14px;
		color: var(--fg-muted);
		font-size: 14px;
		line-height: 1.5;
	}

	.n {
		margin-right: 8px;
		color: var(--fg-default);
		font: 500 11px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}
</style>
