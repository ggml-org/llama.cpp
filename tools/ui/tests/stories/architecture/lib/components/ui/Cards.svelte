<!-- A grid of cards: 2, 3 or 4 columns that drop to one on narrow screens -->
<script lang="ts">
	import type { Snippet } from 'svelte';
	let { children, cols = 3 }: { cols?: 2 | 3 | 4; children: Snippet } = $props();
</script>

<div style:--cols={cols} class="cards">{@render children()}</div>

<style>
	.cards {
		display: grid;
		grid-template-columns: repeat(var(--cols), minmax(0, 1fr));
		gap: 16px;
	}
	@media (max-width: 900px) {
		.cards {
			grid-template-columns: repeat(min(var(--cols), 2), minmax(0, 1fr));
		}
	}
	@media (max-width: 560px) {
		.cards {
			grid-template-columns: 1fr;
		}
	}
</style>
