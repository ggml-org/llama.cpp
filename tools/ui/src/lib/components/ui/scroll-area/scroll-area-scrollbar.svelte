<script lang="ts">
	import { cn, type WithoutChild } from '$lib/components/ui/utils';
	import { ScrollArea as ScrollAreaPrimitive } from 'bits-ui';

	let {
		children,
		class: className,
		orientation = 'vertical',
		ref = $bindable(null),
		...restProps
	}: WithoutChild<ScrollAreaPrimitive.ScrollbarProps> = $props();
</script>

<ScrollAreaPrimitive.Scrollbar
	bind:ref
	class={cn(
		'flex touch-none p-px transition-colors select-none',
		orientation === 'vertical' && 'h-full w-2.5 border-l border-l-transparent',
		orientation === 'horizontal' && 'h-2.5 flex-col border-t border-t-transparent',
		className
	)}
	data-slot="scroll-area-scrollbar"
	{orientation}
	{...restProps}
>
	{@render children?.()}

	<ScrollAreaPrimitive.Thumb
		class="relative flex-1 rounded-full bg-border"
		data-slot="scroll-area-thumb"
	/>
</ScrollAreaPrimitive.Scrollbar>
