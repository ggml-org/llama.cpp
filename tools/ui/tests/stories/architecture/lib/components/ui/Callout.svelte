<!-- A boxed finding or note with a title, set off from the prose -->
<script lang="ts">
	import type { Snippet } from 'svelte';
	let { children, title }: { title?: string; children: Snippet } = $props();
</script>

<aside class="callout">
	{#if title}<strong>{title}</strong>{/if}

	<div class="body">{@render children()}</div>
</aside>

<style>
	.callout {
		border-left: 3px solid var(--app);
		background: var(--surface);
		border-radius: 0 12px 12px 0;
		padding: 16px 18px;
		display: grid;
		gap: 8px;
	}
	strong {
		color: var(--app);
	}
	/* normal flow inside, so inline text, links and code stay on one line */
	.body > :global(*) {
		margin: 0;
	}
	.body > :global(* + *) {
		margin-top: 8px;
	}
</style>
