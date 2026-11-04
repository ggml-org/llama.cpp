<!-- One design note: a term and what it says about the letters. -->
<script lang="ts">
	import type { Snippet } from 'svelte';
	let { children, term }: { term: string; children: Snippet } = $props();
</script>

<div>
	<dt>{term}</dt>

	<dd>{@render children()}</dd>
</div>

<style>
	div {
		display: grid;
		gap: 4px;
		align-content: start;
		min-width: 0;
		padding-top: 10px;
		border-top: 1px solid var(--rule);
	}
	dt {
		font: 500 11px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--muted);
	}
	dd {
		margin: 0;
		font-size: 15px;
		line-height: 1.5;
	}
</style>
