<script lang="ts">
	import { ArrowBigUp } from '@lucide/svelte';

	interface Props {
		keys: string[];
		variant?: 'default' | 'destructive';
		class?: string;
	}

	let {
		keys,
		variant = 'default',
		class: className = ''
	}: Props = $props();

	let baseClasses = 'px-1 pointer-events-none inline-flex select-none items-center gap-0.5 font-mono text-md font-medium opacity-0 transition-opacity';
	let variantClasses = variant === 'destructive' ? 'text-destructive' : 'text-muted-foreground';
</script>

<kbd class="{baseClasses} {variantClasses} {className}">
	{#each keys as key, index}
		{#if key === 'shift'}
			<ArrowBigUp class="h-2 w-2 {variant === 'destructive' ? 'text-destructive' : ''} -mr-1" />
		{:else if key === 'cmd'}
			<span class="text-lg {variant === 'destructive' ? 'text-destructive' : ''}">⌘</span>
		{:else}
			{key.toUpperCase()}
		{/if}
		{#if index < keys.length - 1}
			<span> </span>
		{/if}
	{/each}
</kbd>
