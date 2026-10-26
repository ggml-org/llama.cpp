<script lang="ts">
	import { cn, type WithElementRef, type WithoutChildren } from '$lib/components/ui/utils.js';
	import type { HTMLAttributes } from 'svelte/elements';

	let {
		class: className,
		ref = $bindable(null),
		...restProps
	}: WithoutChildren<WithElementRef<HTMLAttributes<HTMLDivElement>>> = $props();
</script>

<div
	bind:this={ref}
	class={cn('animate-pulse rounded-md bg-accent', className)}
	data-slot="skeleton"
	{...restProps}
></div>
