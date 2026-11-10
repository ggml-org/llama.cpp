<script lang="ts">
	import { getToggleGroupCtx } from './toggle-group.svelte';
	import { type ToggleSize, toggleVariants } from '$lib/components/ui/toggle/index.js';
	import { cn } from '$lib/components/ui/utils.js';
	import { ToggleGroup as ToggleGroupPrimitive } from 'bits-ui';

	let {
		class: className,
		ref = $bindable(null),
		size,
		value = $bindable(),
		...restProps
	}: ToggleGroupPrimitive.ItemProps & { size?: ToggleSize } = $props();

	const ctx = getToggleGroupCtx();
</script>

<!--
	a key: pressed, it turns white and sinks in, its label a pixel lower; while any key is
	pressed, the others turn grey and mute their labels
-->
<ToggleGroupPrimitive.Item
	bind:ref
	class={cn(
		toggleVariants({ size: ctx.size || size, variant: 'default' }),
		'h-auto shrink-0 rounded-none text-(--toggle-group-key-text) backdrop-blur-none hover:bg-(--toggle-group-key-background-hover) hover:text-(--toggle-group-key-text) focus:z-10 focus-visible:z-10',
		'border-(--toggle-group-seam) group-data-[orientation=horizontal]/toggle-group:not-first:border-l group-data-[orientation=vertical]/toggle-group:not-first:border-t',
		'group-has-[[data-state=on]]/toggle-group:data-[state=off]:bg-(--toggle-group-key-background-idle) group-has-[[data-state=on]]/toggle-group:data-[state=off]:text-(--toggle-group-key-text-muted) group-has-[[data-state=on]]/toggle-group:data-[state=off]:hover:bg-(--toggle-group-key-background-idle-hover) group-has-[[data-state=on]]/toggle-group:data-[state=off]:hover:text-(--toggle-group-key-text)',
		'data-[state=on]:bg-(--toggle-group-key-background-on) data-[state=on]:text-(--toggle-group-key-text) data-[state=on]:shadow-(--toggle-group-key-shadow-on) data-[state=on]:hover:bg-(--toggle-group-key-background-on) data-[state=on]:*:translate-y-px',
		className
	)}
	data-size={ctx.size || size}
	data-slot="toggle-group-item"
	{value}
	{...restProps}
/>
