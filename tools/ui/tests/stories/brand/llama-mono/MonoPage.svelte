<!--
	The page a Llama Mono usage story sits on, inside OverviewFrame: a kicker, a title and a short intro, then the
	examples in the overview's column.
-->
<script lang="ts">
	import type { Snippet } from 'svelte';

	let {
		children,
		intro,
		kicker,
		title
	}: { children: Snippet; intro: string; kicker: string; title: string } = $props();
</script>

<div class="page">
	<header>
		<p class="kicker">{kicker}</p>

		<h1>{title}</h1>

		<p class="intro">{intro}</p>
	</header>

	{@render children()}
</div>

<style>
	.page {
		display: grid;
		gap: 40px;
		width: var(--column);
		margin: 0 auto;
		padding-top: clamp(32px, 5vw, 64px);
	}
	/* the examples fill this column instead of narrowing it again */
	.page > :global(*) {
		--column: 100%;
		min-width: 0;
	}
	header {
		display: grid;
		gap: 8px;
	}
	header > * {
		margin: 0;
	}
	.kicker {
		font: 500 11px var(--code);
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--muted);
	}
	h1 {
		font: 700 clamp(36px, 5vw, 56px) / 1.1 var(--headline);
		letter-spacing: -0.01em;
	}
	.intro {
		max-width: 62ch;
		font: 400 16px / 1.5 var(--ui);
		color: var(--muted);
	}
</style>
