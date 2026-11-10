<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { HTMLAttributes } from 'svelte/elements';

	interface Props extends HTMLAttributes<HTMLDivElement> {
		children?: Snippet;
		class?: string;
		ref?: HTMLDivElement;
	}

	let { children, class: className = '', ref = $bindable(), ...rest }: Props = $props();
</script>

<!-- The one element that carries the markdown styles, shared by MarkdownContent and
     MarkdownEditor, so the bundle holds a single copy of markdown-content.css -->
<div bind:this={ref} class="markdown-content {className}" {...rest}>
	{@render children?.()}
</div>

<style>
	@import './markdown-content.css';
</style>
