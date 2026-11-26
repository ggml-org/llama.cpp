<script lang="ts">
	import { cn, type WithoutChildrenOrChild } from '../utils.js';
	import { Switch as SwitchPrimitive } from 'bits-ui';

	let {
		checked = $bindable(false),
		class: className,
		ref = $bindable(null),
		size = 'default',
		...restProps
	}: WithoutChildrenOrChild<SwitchPrimitive.RootProps> & {
		/** `lg` is sized for a finger, for touch screens' settings lists. */
		size?: 'default' | 'lg';
	} = $props();
</script>

<SwitchPrimitive.Root
	bind:checked
	bind:ref
	class={cn(
		'peer group/switch inline-flex h-[1.15rem] w-8 data-[size=lg]:h-7 data-[size=lg]:w-12 shrink-0 cursor-pointer items-center rounded-full border border-transparent bg-(--surface-muted) shadow-(--elevation-inset) transition-all outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 disabled:cursor-not-allowed disabled:opacity-50 data-[state=checked]:bg-(--action-primary)',
		className
	)}
	data-size={size}
	data-slot="switch"
	{...restProps}
>
	<SwitchPrimitive.Thumb
		class={cn(
			'pointer-events-none block size-4 group-data-[size=lg]/switch:size-6 rounded-full bg-(--surface-lifted) shadow-(--elevation-control) ring-0 transition-transform data-[state=checked]:translate-x-[calc(100%-2px)] data-[state=checked]:bg-(--switch-thumb-on) data-[state=unchecked]:translate-x-0'
		)}
		data-slot="switch-thumb"
	/>
</SwitchPrimitive.Root>
