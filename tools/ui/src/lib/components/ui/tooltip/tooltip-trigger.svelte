<script lang="ts">
	import { cn } from '$lib/components/ui/utils';
	import { Tooltip as TooltipPrimitive } from 'bits-ui';

	let {
		class: className,
		ref = $bindable(null),
		...restProps
	}: TooltipPrimitive.TriggerProps = $props();
</script>

<TooltipPrimitive.Trigger
	bind:ref
	class={cn('cursor-pointer', className)}
	data-slot="tooltip-trigger"
	{...restProps}
/>
