<!--
	A Mono pill: an outline chip for a use case, a solid one for a tag, and the cinder one, used once per post.
-->
<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		children: Snippet;
		/** the pill's font size */
		size?: number;
		variant?: 'accent' | 'outline' | 'solid';
	}

	let { children, size = 24, variant = 'outline' }: Props = $props();
</script>

<span style:--s="{size}px" class="chip {variant}">{@render children()}</span>

<style>
	.chip {
		display: inline-flex;
		align-items: center;
		padding: calc(var(--s) * 0.5) calc(var(--s) * 0.9);
		border: 2px solid var(--p-line);
		border-radius: 999px;
		font: 500 var(--s) / 1 var(--mono);
		white-space: nowrap;
	}
	.solid {
		padding-block: calc(var(--s) * 0.42);
		border-color: transparent;
		background: var(--p-solid);
		color: var(--p-solid-ink);
	}
	.accent {
		padding-block: calc(var(--s) * 0.42);
		border-color: transparent;
		background: var(--art-cinder);
		color: #111111;
	}
</style>
