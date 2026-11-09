<!-- A small footnote under a rule, closing a panel -->
<script lang="ts">
	import type { Snippet } from 'svelte';
	let { children }: { children: Snippet } = $props();
</script>

<div class="note">{@render children()}</div>

<style>
	.note {
		font-size: 13px;
		color: var(--muted);
		border-top: 1px solid var(--rule);
		padding-top: 10px;
	}
	.note :global(*) {
		margin: 0;
	}
	.note.note :global(a) {
		color: inherit;
	}
</style>
