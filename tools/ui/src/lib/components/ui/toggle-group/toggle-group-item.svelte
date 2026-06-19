<script lang="ts">
	import { ToggleGroup as ToggleGroupPrimitive } from 'bits-ui';
	import { getToggleGroupCtx } from './toggle-group.svelte';
	import { cn } from '$lib/components/ui/utils.js';
	import { type ToggleVariants, toggleVariants } from '$lib/components/ui/toggle/index.js';

	let {
		ref = $bindable(null),
		value = $bindable(),
		class: className,
		size,
		variant,
		...restProps
	}: ToggleGroupPrimitive.ItemProps & ToggleVariants = $props();

	const ctx = getToggleGroupCtx();
</script>

<ToggleGroupPrimitive.Item
	bind:ref
	data-slot="toggle-group-item"
	data-variant={ctx.variant || variant}
	data-size={ctx.size || size}
	data-spacing={ctx.spacing}
	class={cn(
		'group-data-[spacing=0]/toggle-group:rounded-none group-data-[spacing=0]/toggle-group:px-2 group-data-[spacing=0]/toggle-group:has-data-[icon=inline-end]:pr-1.5 group-data-[spacing=0]/toggle-group:has-data-[icon=inline-start]:pl-1.5 group-data-horizontal/toggle-group:data-[spacing=0]:first:rounded-l-lg group-data-vertical/toggle-group:data-[spacing=0]:first:rounded-t-lg group-data-horizontal/toggle-group:data-[spacing=0]:last:rounded-r-lg group-data-vertical/toggle-group:data-[spacing=0]:last:rounded-b-lg shrink-0 focus:z-10 focus-visible:z-10 group-data-horizontal/toggle-group:data-[spacing=0]:data-[variant=outline]:border-l-0 group-data-vertical/toggle-group:data-[spacing=0]:data-[variant=outline]:border-t-0 group-data-horizontal/toggle-group:data-[spacing=0]:data-[variant=outline]:first:border-l group-data-vertical/toggle-group:data-[spacing=0]:data-[variant=outline]:first:border-t',
		toggleVariants({
			variant: ctx.variant || variant,
			size: ctx.size || size
		}),
		className
	)}
	{value}
	{...restProps}
/>
