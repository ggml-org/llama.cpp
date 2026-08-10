<script lang="ts">
	import { cn } from '$lib/components/ui/utils.js';
	import { RadioGroup as RadioGroupPrimitive } from 'bits-ui';

	let {
		class: className,
		ref = $bindable(null),
		value = $bindable(''),
		...restProps
	}: RadioGroupPrimitive.RootProps = $props();
</script>

<RadioGroupPrimitive.Root
	bind:ref
	bind:value
	data-slot="radio-group"
	class={cn('grid gap-2 w-full', className)}
	{...restProps}
/>
