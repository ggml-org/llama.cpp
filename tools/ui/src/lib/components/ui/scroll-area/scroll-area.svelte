<script lang="ts">
	import { Scrollbar } from './index.js';
	import { cn, type WithoutChild } from '$lib/components/ui/utils';
	import { ScrollArea as ScrollAreaPrimitive } from 'bits-ui';

	let {
		children,
		class: className,
		orientation = 'vertical',
		ref = $bindable(null),
		scrollbarXClasses = '',
		scrollbarYClasses = '',
		...restProps
	}: WithoutChild<ScrollAreaPrimitive.RootProps> & {
		orientation?: 'vertical' | 'horizontal' | 'both' | undefined;
		scrollbarXClasses?: string | undefined;
		scrollbarYClasses?: string | undefined;
	} = $props();
</script>

<ScrollAreaPrimitive.Root
	bind:ref
	class={cn('relative', className)}
	data-slot="scroll-area"
	{...restProps}
>
	<ScrollAreaPrimitive.Viewport
		class="size-full rounded-[inherit] ring-ring/10 outline-ring/50 transition-[color,box-shadow] focus-visible:ring-4 focus-visible:outline-1 dark:ring-ring/20 dark:outline-ring/40"
		data-slot="scroll-area-viewport"
	>
		{@render children?.()}
	</ScrollAreaPrimitive.Viewport>

	{#if orientation === 'vertical' || orientation === 'both'}
		<Scrollbar class={scrollbarYClasses} orientation="vertical" />
	{/if}

	{#if orientation === 'horizontal' || orientation === 'both'}
		<Scrollbar class={scrollbarXClasses} orientation="horizontal" />
	{/if}

	<ScrollAreaPrimitive.Corner />
</ScrollAreaPrimitive.Root>
