<script lang="ts">
	import { cn } from '$lib/components/ui/utils';
	import type { Snippet } from 'svelte';
	import type { HTMLAttributes } from 'svelte/elements';

	interface Props extends HTMLAttributes<HTMLDivElement> {
		children: Snippet;
	}

	let { class: className, children, ...restProps }: Props = $props();
</script>

<div
	class={cn(
		'flex items-center [&>*:first-child]:rounded-r-none [&>*:last-child]:rounded-l-none [&>*:not(:first-child):not(:last-child)]:rounded-none',
		className
	)}
	{...restProps}
>
	{@render children()}
</div>
