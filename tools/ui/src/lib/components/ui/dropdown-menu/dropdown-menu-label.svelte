<script lang="ts">
	import { cn, type WithElementRef } from '$lib/components/ui/utils.js';
	import type { HTMLAttributes } from 'svelte/elements';

	let {
		children,
		class: className,
		inset,
		ref = $bindable(null),
		...restProps
	}: WithElementRef<HTMLAttributes<HTMLDivElement>> & {
		inset?: boolean;
	} = $props();
</script>

<div
	bind:this={ref}
	class={cn('px-2 py-1.5 text-sm font-semibold data-[inset]:pl-8', className)}
	data-inset={inset}
	data-slot="dropdown-menu-label"
	{...restProps}
>
	{@render children?.()}
</div>
