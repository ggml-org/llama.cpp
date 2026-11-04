<!--
	One version of Llama Sans, after github.com/mona-sans: a rule with the version's name and the icon, a big statement
	set in that version, then its sample (children) and specs. side="end" puts the text in the right half.
-->
<script lang="ts">
	import type { Snippet } from 'svelte';

	let {
		children,
		name,
		side = 'start',
		specs,
		statement
	}: {
		children?: Snippet;
		name: string;
		side?: 'end' | 'start';
		specs: string;
		statement: string;
	} = $props();
</script>

<section
	class:end={side === 'end'}
	style:--face={`'Llama Sans ${name}', var(--body)`}
	class="version"
>
	<header>
		<h2>Llama Sans {name}</h2>

		<span aria-hidden="true"></span>
	</header>

	<div class="text">
		<p class="statement">{statement}</p>

		{#if children}<div class="sample">{@render children()}</div>{/if}

		<p class="specs">{specs}</p>
	</div>
</section>

<style>
	.version {
		display: grid;
		gap: clamp(24px, 3vw, 40px);
		width: var(--column);
		margin-inline: auto;
		font-family: var(--face);
	}
	header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding-top: 20px;
		border-top: 1px solid var(--rule);
	}
	h2 {
		margin: 0;
		font: 700 clamp(28px, 3.4vw, 44px) / 1 var(--headline);
		letter-spacing: -0.01em;
		text-transform: uppercase;
	}
	header span {
		font: 700 28px / 1 var(--headline);
		color: var(--muted);
	}
	.text {
		display: grid;
		gap: 28px;
		width: min(100%, 680px);
	}
	.end .text {
		justify-self: end;
	}
	.text > * {
		margin: 0;
	}
	.statement {
		font: 600 clamp(36px, 5.4vw, 72px) / 1.05 var(--face);
		letter-spacing: -0.01em;
	}
	.sample {
		max-width: 560px;
		font-family: var(--face);
	}
	.specs {
		font: 500 12px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--muted);
	}
</style>
