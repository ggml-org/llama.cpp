<script lang="ts">
	import { cn } from '$lib/components/ui/utils';
	import type { Snippet } from 'svelte';

	interface Props {
		children: Snippet;
		class?: string;
		icon?: Snippet;
		onclick?: () => void;
	}

	let { children, class: className = '', icon, onclick }: Props = $props();
</script>

<button
	class={cn(
		'inline-flex cursor-pointer items-center gap-1 rounded-sm bg-muted-foreground/15 px-1.5 py-0.75',
		className
	)}
	{onclick}
>
	{#if icon}
		{@render icon()}
	{/if}

	{@render children()}
</button>
