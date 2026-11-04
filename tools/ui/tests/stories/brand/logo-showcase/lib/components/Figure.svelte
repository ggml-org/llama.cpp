<!-- One small drawing on a card with its caption: a step of the origin (with its number) or a rule of the icon.
     The drawing is a snippet in the icon's own coordinates; the caption is the children. dim fades it back;
     with onclick, the whole card is a button labelled by action. -->
<script lang="ts">
	import type { Snippet } from 'svelte';

	let {
		action,
		children,
		dim = false,
		drawing,
		n,
		onclick,
		title,
		view = '-60 -60 720 720'
	}: {
		title: string;
		n?: string;
		view?: string;
		dim?: boolean;
		onclick?: () => void;
		action?: string;
		drawing: Snippet;
		children?: Snippet;
	} = $props();
</script>

<figure class:faded={dim} class="panel">
	{#if onclick}<button aria-label={action ?? title} class="go" {onclick} type="button"
		></button>{/if}

	{#if n}<span class="step">{n}</span>{/if}

	<svg aria-label={title} role="img" viewBox={view}>{@render drawing()}</svg>

	<figcaption><strong>{title}</strong>{@render children?.()}</figcaption>
</figure>

<style>
	.panel {
		position: relative;
		margin: 0;
		display: grid;
		gap: 10px;
		align-content: start;
		min-width: 0;
		padding: 14px;
		border-radius: 12px;
		background: var(--l-panel);
		transition:
			opacity 400ms ease-out,
			box-shadow 160ms ease-out;
	}
	.faded {
		opacity: 0.25;
	}
	.go {
		position: absolute;
		inset: 0;
		padding: 0;
		border: 0;
		border-radius: inherit;
		background: none;
		cursor: pointer;
	}
	.go:focus-visible {
		outline: 2px solid var(--l-ink);
		outline-offset: 2px;
	}
	.panel:has(> .go:hover) {
		box-shadow: inset 0 0 0 1px var(--l-rule);
		opacity: 1;
	}
	.panel svg {
		display: block;
		width: 100%;
		height: auto;
	}
	figcaption {
		display: grid;
		gap: 2px;
		font:
			400 14px / 1.5 'Llama Sans UI',
			var(--body);
		color: var(--l-muted);
	}
	strong {
		color: var(--l-ink);
		font:
			600 15px / normal 'Llama Sans UI',
			var(--body);
	}
	.step {
		font: 500 11px var(--mono);
		letter-spacing: 0.08em;
		color: var(--l-muted);
	}

	/* the drawings' marks, for every figure's drawing snippet */
	svg :global(.guide) {
		stroke: var(--l-line);
		stroke-width: 1;
		stroke-dasharray: 6 4;
		fill: none;
		vector-effect: non-scaling-stroke;
	}
	svg :global(.joint) {
		fill: var(--l-line);
	}
	svg :global(.cell) {
		fill: var(--l-faint);
		stroke: var(--l-ink);
		stroke-width: 2.5;
		vector-effect: non-scaling-stroke;
	}
	svg :global(.cell-outline) {
		fill: none;
		stroke: var(--l-ink);
		stroke-width: 2;
		vector-effect: non-scaling-stroke;
	}
	svg :global(.ghost) {
		fill: none;
		stroke: var(--l-line);
		stroke-width: 1.5;
		vector-effect: non-scaling-stroke;
	}
	svg :global(.lifted) {
		fill: none;
		stroke: var(--l-ink);
		stroke-width: 1.5;
		stroke-dasharray: 5 5;
		vector-effect: non-scaling-stroke;
	}
	svg :global(.strike) {
		fill: var(--l-bolt-core);
		stroke: var(--l-bolt-glow);
		stroke-width: 2;
		stroke-linejoin: round;
		vector-effect: non-scaling-stroke;
		filter: drop-shadow(0 0 2px var(--l-bolt-glow)) drop-shadow(0 0 6px var(--l-bolt-halo));
	}
	svg :global(.hi) {
		stroke: var(--l-ink);
		stroke-width: 2.5;
		fill: none;
		vector-effect: non-scaling-stroke;
	}
	svg :global(.dim) {
		stroke: var(--l-ink);
		stroke-width: 1.5;
		stroke-dasharray: 4 4;
		vector-effect: non-scaling-stroke;
	}
	svg :global(.arc) {
		stroke: var(--l-ink);
		stroke-width: 2;
		fill: none;
		vector-effect: non-scaling-stroke;
	}
	svg :global(.note) {
		font:
			500 30px 'Llama Sans UI',
			var(--body);
		fill: var(--l-ink);
	}
	svg :global(.note.on-shape) {
		fill: var(--l-bg);
	}
</style>
