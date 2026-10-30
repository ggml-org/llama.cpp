<script lang="ts">
	import { cn } from '$lib/components/ui/utils.js';
	import { Tabs as TabsPrimitive } from 'bits-ui';

	let {
		class: className,
		ref = $bindable(null),
		...restProps
	}: TabsPrimitive.ContentProps = $props();
</script>

<TabsPrimitive.Content
	bind:ref
	class={cn('flex-1 text-body-sm outline-none', className)}
	data-slot="tabs-content"
	{...restProps}
/>
