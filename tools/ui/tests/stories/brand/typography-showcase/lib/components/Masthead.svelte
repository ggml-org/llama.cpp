<!-- The specimen's title: the build line, the family's name over the version on show, and the lede as children. -->
<script lang="ts">
	import { specimen } from '../specimen';
	import type { Snippet } from 'svelte';

	let { children, variant }: { variant: string; children?: Snippet } = $props();
	const { axis, glyphCount } = specimen;
</script>

<header>
	<div class="top">
		<span class="label">Draft 0.6 · variable {axis.min}–{axis.max} · {glyphCount} glyphs</span>

		<span class="label">Llama family · Sans</span>
	</div>

	<h1>Llama Sans <span class="version">{variant}</span></h1>

	{#if children}<div class="lede">{@render children()}</div>{/if}
</header>

<style>
	header {
		display: grid;
		gap: 20px;
	}
	.top {
		display: flex;
		flex-wrap: wrap;
		justify-content: space-between;
		gap: 12px;
		align-items: baseline;
	}
	h1 {
		margin: 0;
		font: var(--w, 700) clamp(56px, 12vw, 148px) / 0.9 var(--display);
		letter-spacing: -0.01em;
		text-wrap: balance;
	}
	h1 .version {
		display: block; /* the version always sits on its own line */
		color: var(--muted);
	}
	.lede {
		max-width: 62ch;
		color: var(--muted);
		font-size: 17px;
	}
	.lede :global(p) {
		margin: 0;
	}
	@media (max-width: 600px) {
		.top {
			display: none;
		}
	}
</style>
