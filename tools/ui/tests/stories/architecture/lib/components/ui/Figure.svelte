<!-- A diagram in a card that scrolls sideways on narrow screens. `label` describes it for screen readers -->
<script lang="ts">
	import type { Snippet } from 'svelte';
	let {
		children,
		label,
		minWidth = 720
	}: { label: string; minWidth?: number; children: Snippet } = $props();
</script>

<div
	style:--min-width="{minWidth}px"
	aria-label={label}
	class="figure card-surface rounded-card"
	role="img"
>
	{@render children()}
</div>

<style>
	.figure {
		padding: 12px;
		overflow-x: auto;
	}
	.figure :global(svg) {
		display: block;
		width: 100%;
		min-width: var(--min-width);
		height: auto;
	}
</style>
