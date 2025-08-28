<script lang="ts">
	import type { HTMLAttributes } from 'svelte/elements';
	import { cn, type WithElementRef } from '$lib/components/ui/utils.js';

	let {
		ref = $bindable(null),
		class: className,
		children,
		...restProps
	}: WithElementRef<HTMLAttributes<HTMLElement>> = $props();
</script>

<div
	bind:this={ref}
	data-slot="sidebar-group"
	data-sidebar="group"
	class={cn('relative flex w-full min-w-0 flex-col p-2', className)}
	{...restProps}
>
	{@render children?.()}
</div>
