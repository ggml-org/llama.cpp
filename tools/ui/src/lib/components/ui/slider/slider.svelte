<script lang="ts">
	import { cn, type WithoutChildrenOrChild } from '$lib/components/ui/utils';
	import { Slider as SliderPrimitive } from 'bits-ui';

	let {
		class: className,
		orientation = 'horizontal',
		ref = $bindable(null),
		value = $bindable(),
		...restProps
	}: WithoutChildrenOrChild<SliderPrimitive.RootProps> = $props();
</script>

<!--
Discriminated Unions + Destructing (required for bindable) do not
get along, so we shut typescript up by casting `value` to `never`.
-->
<SliderPrimitive.Root
	bind:ref
	bind:value={value as never}
	class={cn(
		'data-[orientation=vertical]:min-h-40 relative flex w-full touch-none items-center select-none data-disabled:opacity-50 data-[orientation=vertical]:h-full data-[orientation=vertical]:w-auto data-[orientation=vertical]:flex-col',
		className
	)}
	data-slot="slider"
	{orientation}
	{...restProps}
>
	{#snippet children({ thumbItems })}
		<span
			class={cn(
				'bg-muted rounded-full data-[orientation=horizontal]:h-1 data-[orientation=horizontal]:w-full data-[orientation=vertical]:h-full data-[orientation=vertical]:w-1 relative grow overflow-hidden data-[orientation=horizontal]:w-full data-[orientation=vertical]:h-full'
			)}
			data-orientation={orientation}
			data-slot="slider-track"
		>
			<SliderPrimitive.Range
				class={cn(
					'bg-primary absolute select-none data-[orientation=horizontal]:h-full data-[orientation=vertical]:w-full'
				)}
				data-slot="slider-range"
			/>
		</span>

		{#each thumbItems as thumb (thumb.index)}
			<SliderPrimitive.Thumb
				class="border-ring ring-ring/50 relative size-3 rounded-full border bg-white transition-[color,box-shadow] after:absolute after:-inset-2 hover:ring-3 focus-visible:ring-3 focus-visible:outline-hidden active:ring-3 block shrink-0 select-none disabled:pointer-events-none disabled:opacity-50"
				data-slot="slider-thumb"
				index={thumb.index}
			/>
		{/each}
	{/snippet}
</SliderPrimitive.Root>
