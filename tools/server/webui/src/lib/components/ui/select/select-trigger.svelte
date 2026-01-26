<script lang="ts">
	import { Select as SelectPrimitive } from 'bits-ui';
	import ChevronDownIcon from '@lucide/svelte/icons/chevron-down';
	import { cn, type WithoutChild } from '$lib/components/ui/utils.js';

	let {
		ref = $bindable(null),
		class: className,
		children,
		size = 'default',
		variant = 'default',
		...restProps
	}: WithoutChild<SelectPrimitive.TriggerProps> & {
		size?: 'sm' | 'default';
		variant?: 'default' | 'plain';
	} = $props();

	const baseClasses = $derived(
		variant === 'plain'
			? "group inline-flex w-full items-center justify-end gap-2 whitespace-nowrap px-0 py-0 text-sm font-medium text-muted-foreground transition-colors focus-visible:outline-none focus-visible:ring-0 focus-visible:ring-offset-0 disabled:cursor-not-allowed disabled:opacity-50 data-[placeholder]:text-muted-foreground data-[size=default]:h-9 data-[size=sm]:h-8 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-3 [&_svg:not([class*='text-'])]:text-muted-foreground"
			: "flex w-fit items-center justify-between gap-2 rounded-md border border-input bg-transparent px-3 py-2 text-sm whitespace-nowrap shadow-xs transition-[color,box-shadow] outline-none select-none focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:cursor-not-allowed disabled:opacity-50 aria-invalid:border-destructive aria-invalid:ring-destructive/20 data-[placeholder]:text-muted-foreground data-[size=default]:h-9 data-[size=sm]:h-8 *:data-[slot=select-value]:line-clamp-1 *:data-[slot=select-value]:flex *:data-[slot=select-value]:items-center *:data-[slot=select-value]:gap-2 dark:bg-input/30 dark:hover:bg-input/50 dark:aria-invalid:ring-destructive/40 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4 [&_svg:not([class*='text-'])]:text-muted-foreground"
	);

	const chevronClasses = $derived(
		variant === 'plain'
			? 'size-3 opacity-60 transition-transform group-data-[state=open]:-rotate-180'
			: 'size-4 opacity-50'
	);
</script>

<SelectPrimitive.Trigger
	bind:ref
	data-slot="select-trigger"
	data-size={size}
	class={cn(baseClasses, className)}
	{...restProps}
>
	{@render children?.()}
	<ChevronDownIcon class={chevronClasses} />
</SelectPrimitive.Trigger>
