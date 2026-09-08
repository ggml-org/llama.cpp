<script lang="ts">
	import { cn, type WithElementRef } from '$lib/components/ui/utils.js';
	import type { HTMLAttributes } from 'svelte/elements';

	let {
		children,
		class: className,
		ref = $bindable(null),
		...restProps
	}: WithElementRef<HTMLAttributes<HTMLDivElement>> = $props();
</script>

<div
	bind:this={ref}
	class={cn('col-start-2 line-clamp-1 min-h-4 font-medium tracking-tight', className)}
	data-slot="alert-title"
	{...restProps}
>
	{@render children?.()}
</div>
