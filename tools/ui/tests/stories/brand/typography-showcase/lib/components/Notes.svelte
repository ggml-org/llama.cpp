<!-- The design notes: a grid of <Note>s, then the next pass and the licence as Markdown after it. -->
<script lang="ts">
	import type { Snippet } from 'svelte';
	let { children }: { children: Snippet } = $props();
</script>

<dl class="notes">{@render children()}</dl>

<style>
	.notes {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
		gap: 18px 28px;
		margin: 0;
	}
	/* design notes are for the desktop page; on phones the specimen keeps to the type itself */
	@media (max-width: 600px) {
		.notes {
			display: none;
		}
	}
</style>
