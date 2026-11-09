<!-- A stack of <Layer> rows -->
<script lang="ts">
	import type { Snippet } from 'svelte';
	let { children }: { children: Snippet } = $props();
</script>

<div class="layers">{@render children()}</div>

<style>
	.layers {
		display: grid;
		gap: 8px;
	}
</style>
