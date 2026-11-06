<script lang="ts">
	import CheckIcon from '@lucide/svelte/icons/check';
	import { cn, type WithoutChild } from '$lib/components/ui/utils.js';
	import { Select as SelectPrimitive } from 'bits-ui';

	let {
		children: childrenProp,
		class: className,
		label,
		ref = $bindable(null),
		value,
		...restProps
	}: WithoutChild<SelectPrimitive.ItemProps> = $props();
</script>

<SelectPrimitive.Item
	bind:ref
	class={cn(
		"relative flex w-full cursor-default items-center gap-2 rounded-sm py-1.5 pr-8 pl-2 text-sm outline-hidden select-none data-[disabled]:pointer-events-none data-[disabled]:opacity-50 data-[highlighted]:bg-accent data-[highlighted]:text-accent-foreground [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4 [&_svg:not([class*='text-'])]:text-muted-foreground *:[span]:last:flex *:[span]:last:items-center *:[span]:last:gap-2",
		className
	)}
	data-slot="select-item"
	{value}
	{...restProps}
>
	{#snippet children({ highlighted, selected })}
		<span class="absolute right-2 flex size-3.5 items-center justify-center">
			{#if selected}
				<CheckIcon class="size-4" />
			{/if}
		</span>

		{#if childrenProp}
			{@render childrenProp({ highlighted, selected })}
		{:else}
			{label || value}
		{/if}
	{/snippet}
</SelectPrimitive.Item>
