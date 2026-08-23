<script lang="ts">
	import { cn, type WithoutChildrenOrChild } from '$lib/components/ui/utils.js';
	import * as ResizablePrimitive from 'paneforge';

	let {
		class: className,
		ref = $bindable(null),
		withHandle = false,
		...restProps
	}: WithoutChildrenOrChild<ResizablePrimitive.PaneResizerProps> & {
		withHandle?: boolean;
	} = $props();
</script>

<ResizablePrimitive.PaneResizer
	bind:ref
	data-slot="resizable-handle"
	class={cn(
		'cn-resizable-handle group relative flex w-3 shrink-0 items-center justify-center bg-transparent before:absolute before:inset-y-0 before:left-1/2 before:w-px before:-translate-x-1/2 before:bg-border before:transition-[background-color,width] before:duration-150 hover:before:bg-ring/70 focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 data-[active=keyboard]:before:bg-ring data-[active=pointer]:before:w-0.5 data-[active=pointer]:before:bg-ring data-[direction=vertical]:h-3 data-[direction=vertical]:w-full data-[direction=vertical]:before:inset-x-0 data-[direction=vertical]:before:inset-y-auto data-[direction=vertical]:before:top-1/2 data-[direction=vertical]:before:h-px data-[direction=vertical]:before:w-auto data-[direction=vertical]:before:-translate-y-1/2 data-[direction=vertical]:before:translate-x-0 [&[data-direction=vertical]>div]:rotate-90 motion-reduce:before:transition-none',
		className
	)}
	{...restProps}
>
	{#if withHandle}
		<div class="pointer-events-none z-10 flex h-6 w-1 shrink-0 rounded-lg bg-border transition-colors duration-150 group-hover:bg-ring/70 group-data-[active=keyboard]:bg-ring group-data-[active=pointer]:bg-ring motion-reduce:transition-none"></div>
	{/if}
</ResizablePrimitive.PaneResizer>
