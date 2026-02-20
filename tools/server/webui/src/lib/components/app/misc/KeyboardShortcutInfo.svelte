<script lang="ts">
	import { ArrowBigUp } from '@lucide/svelte';

	interface Props {
		keys: string[];
		variant?: 'default' | 'destructive';
		class?: string;
	}

	let { keys, variant = 'default', class: className = '' }: Props = $props();

	let baseClasses =
		'px-1 pointer-events-none inline-flex select-none items-center gap-0.5 font-sans text-md font-medium opacity-0 transition-opacity -my-1';
	let variantClasses = $derived(
		variant === 'destructive' ? 'text-destructive' : 'text-muted-foreground'
	);
</script>

<kbd class="{baseClasses} {variantClasses} {className}">
	{#each keys as key, index (index)}
		{#if key === 'shift'}
			<ArrowBigUp class="h-1 w-1 {variant === 'destructive' ? 'text-destructive' : ''} -mr-1" />
		{:else if key === 'cmd'}
			<span class={variant === 'destructive' ? 'text-destructive' : ''}>⌘</span>
		{:else}
			{key.toUpperCase()}
		{/if}

		{#if index < keys.length - 1}
			<span> </span>
		{/if}
	{/each}
</kbd>
