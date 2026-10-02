<script lang="ts">
	import { cn } from '$lib/components/ui/utils.js';
	import { Popover as PopoverPrimitive } from 'bits-ui';

	let {
		class: className,
		ref = $bindable(null),
		...restProps
	}: PopoverPrimitive.TriggerProps = $props();
</script>

<PopoverPrimitive.Trigger
	bind:ref
	class={cn('', className)}
	data-slot="popover-trigger"
	{...restProps}
/>
