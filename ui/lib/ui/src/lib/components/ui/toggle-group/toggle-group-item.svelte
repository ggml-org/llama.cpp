<script lang="ts">
	import { getToggleGroupCtx } from './toggle-group.svelte';
	import { type ToggleSize, toggleVariants } from '../toggle/index.js';
	import { cn } from '../utils.js';
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
		'h-auto shrink-0 rounded-none text-(--fg-default) backdrop-blur-none hover:bg-(--action-ghost-hover) hover:text-(--fg-default) focus:z-10 focus-visible:z-10',
		'border-(--line-control) group-data-[orientation=horizontal]/toggle-group:not-first:border-l group-data-[orientation=vertical]/toggle-group:not-first:border-t',
		'group-has-[[data-state=on]]/toggle-group:data-[state=off]:bg-(--toggle-group-key-background-idle) group-has-[[data-state=on]]/toggle-group:data-[state=off]:text-(--fg-soft) group-has-[[data-state=on]]/toggle-group:data-[state=off]:hover:bg-(--toggle-group-key-background-idle-hover) group-has-[[data-state=on]]/toggle-group:data-[state=off]:hover:text-(--fg-default)',
		'data-[state=on]:bg-(--surface-pressed) data-[state=on]:text-(--fg-default) data-[state=on]:shadow-(--elevation-pressed) data-[state=on]:hover:bg-(--surface-pressed) data-[state=on]:*:translate-y-px',
		className
	)}
	data-size={ctx.size || size}
	data-slot="toggle-group-item"
	{value}
	{...restProps}
/>
