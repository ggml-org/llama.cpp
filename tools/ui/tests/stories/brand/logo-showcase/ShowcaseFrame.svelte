<!--
	The logo showcase's frame: the grey card the page sits on, with the showcase's colours and fonts. A copy of the
	page template in ggml-org/llama.brand (logo/showcase/src/lib/templates/Doc.svelte); the showcase in lib/ and
	LogoShowcase.svx are copied from there too, so change them there first.
-->
<script lang="ts">
	import './lib/styles/logo.css';
	import '../fonts/fonts.css';
	import type { Snippet } from 'svelte';

	let { children }: { children: Snippet } = $props();
</script>

<div class="logo">{@render children()}</div>

<style>
	.logo {
		--body: 'Llama Sans', var(--font-sans);
		--mono: ui-monospace, 'SF Mono', SFMono-Regular, Menlo, Consolas, monospace;
		display: grid;
		gap: 40px;
		max-width: 1180px;
		margin: 0 auto;
		padding: clamp(16px, 4vw, 48px);
		background: var(--l-bg);
		color: var(--l-ink);
		font: 16px/1.55 var(--body);
	}
	.logo > :global(*) {
		margin: 0;
		min-width: 0;
	}
	/* a kicker, its heading and the lead read as one block, 10px apart */
	.logo > :global(.kicker) {
		margin-bottom: -30px;
		font: 500 12px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--l-muted);
	}
	.logo > :global(h2) {
		font:
			700 clamp(32px, 5vw, 56px) / 1.15 'Llama Sans',
			var(--body);
	}
	.logo > :global(h2 + p) {
		max-width: 62ch;
		margin-top: -30px;
		color: var(--l-muted);
		font:
			400 17px / 1.5 'Llama Sans',
			var(--body);
	}
	/* the origin's tiles sit closer under the animation they follow */
	.logo > :global(.steps) {
		margin-top: -20px;
	}
</style>
