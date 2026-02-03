<script lang="ts">
	import { cn, type WithElementRef, type WithoutChildren } from '$lib/components/ui/utils.js';
	import type { HTMLAttributes } from 'svelte/elements';

	let {
		ref = $bindable(null),
		class: className,
		...restProps
	}: WithoutChildren<WithElementRef<HTMLAttributes<HTMLDivElement>>> = $props();
</script>

<div
	bind:this={ref}
	data-slot="skeleton"
	class={cn('animate-pulse rounded-md bg-accent', className)}
	{...restProps}
></div>
