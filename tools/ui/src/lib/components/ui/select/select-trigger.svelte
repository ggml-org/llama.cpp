<script lang="ts">
	import ChevronDownIcon from '@lucide/svelte/icons/chevron-down';
	import { cn, type WithoutChild } from '$lib/components/ui/utils.js';
	import { Select as SelectPrimitive } from 'bits-ui';

	let {
		children,
		class: className,
		ref = $bindable(null),
		size = 'default',
		variant = 'default',
		...restProps
	}: WithoutChild<SelectPrimitive.TriggerProps> & {
		size?: 'xs' | 'sm' | 'default';
		variant?: 'default' | 'plain';
	} = $props();

	// Super small trigger: fits its selected value, for dense inline use.
	const xsClasses =
		"group flex h-6 w-fit items-center justify-between gap-1 rounded-md border border-border/30 bg-muted/60 px-2 py-0 text-xs whitespace-nowrap outline-none select-none transition-colors hover:bg-muted/80 focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:cursor-not-allowed disabled:opacity-50 data-[placeholder]:text-muted-foreground dark:border-border/20 dark:bg-muted/75 dark:hover:bg-muted [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='text-'])]:text-muted-foreground";

	const baseClasses = $derived(
		variant === 'plain'
			? "group inline-flex w-full items-center justify-end gap-2 whitespace-nowrap px-0 py-0 text-sm font-medium text-muted-foreground transition-colors focus-visible:outline-none focus-visible:ring-0 focus-visible:ring-offset-0 disabled:cursor-not-allowed disabled:opacity-50 data-[placeholder]:text-muted-foreground data-[size=default]:h-9 data-[size=sm]:h-8 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-3 [&_svg:not([class*='text-'])]:text-muted-foreground"
			: size === 'xs'
				? xsClasses
				: "flex w-fit items-center justify-between gap-2 rounded-md border border-border/30 bg-muted/60 px-3 py-2 text-sm whitespace-nowrap shadow-sm transition-colors outline-none select-none hover:bg-muted/80 focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:cursor-not-allowed disabled:opacity-50 aria-invalid:border-destructive aria-invalid:ring-destructive/20 data-[placeholder]:text-muted-foreground data-[size=default]:h-9 data-[size=sm]:h-8 *:data-[slot=select-value]:line-clamp-1 *:data-[slot=select-value]:flex *:data-[slot=select-value]:items-center *:data-[slot=select-value]:gap-2 dark:border-border/20 dark:bg-muted/75 dark:hover:bg-muted dark:aria-invalid:ring-destructive/40 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-3.5 [&_svg:not([class*='text-'])]:text-muted-foreground"
	);

	const chevronClasses = $derived(
		variant === 'plain' || size === 'xs'
			? 'size-3 opacity-60 transition-transform group-data-[state=open]:-rotate-180'
			: 'size-3.5 opacity-60 transition-transform group-data-[state=open]:-rotate-180'
	);
</script>

<SelectPrimitive.Trigger
	bind:ref
	class={cn(baseClasses, className)}
	data-size={size}
	data-slot="select-trigger"
	{...restProps}
>
	{@render children?.()}

	<ChevronDownIcon class={chevronClasses} />
</SelectPrimitive.Trigger>
