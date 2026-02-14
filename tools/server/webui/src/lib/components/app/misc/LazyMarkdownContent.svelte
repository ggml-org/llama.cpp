<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import MarkdownContent from './MarkdownContent.svelte';

	interface Props {
		content: string;
		class?: string;
		/** If true, render immediately without waiting for intersection */
		eager?: boolean;
		/** Placeholder height estimate for layout stability */
		placeholderHeight?: string;
	}

	let { content, class: className = '', eager = false, placeholderHeight = 'auto' }: Props = $props();

	let containerRef = $state<HTMLDivElement>();
	let isVisible = $state(false);
	let hasBeenVisible = $state(false);
	let observer: IntersectionObserver | null = null;
	
	// Initialize based on eager prop
	$effect(() => {
		if (eager) {
			isVisible = true;
			hasBeenVisible = true;
		}
	});

	onMount(() => {
		if (eager) {
			hasBeenVisible = true;
			isVisible = true;
			return;
		}
		
		if (!containerRef) {
			return;
		}

		observer = new IntersectionObserver(
			(entries) => {
				for (const entry of entries) {
					if (entry.isIntersecting) {
						isVisible = true;
						hasBeenVisible = true;
						// Once visible, no need to observe anymore
						observer?.disconnect();
					}
				}
			},
			{
				// Start loading slightly before element comes into view
				rootMargin: '200px 0px',
				threshold: 0
			}
		);

		observer.observe(containerRef);
	});

	onDestroy(() => {
		observer?.disconnect();
	});

	// If content is very short, render eagerly (no point lazy loading)
	$effect(() => {
		if (content.length < 200 && !hasBeenVisible) {
			hasBeenVisible = true;
			isVisible = true;
		}
	});
</script>

<div 
	bind:this={containerRef} 
	class={className}
	style:min-height={!hasBeenVisible ? placeholderHeight : 'auto'}
>
	{#if hasBeenVisible}
		<MarkdownContent {content} class={className} />
	{:else}
		<!-- Lightweight placeholder: show raw text truncated -->
		<div class="placeholder-content">
			{content.slice(0, 500)}{content.length > 500 ? '...' : ''}
		</div>
	{/if}
</div>

<style>
	.placeholder-content {
		white-space: pre-wrap;
		word-break: break-word;
		color: var(--muted-foreground);
		font-size: 0.875rem;
		line-height: 1.5;
		opacity: 0.7;
	}
</style>
