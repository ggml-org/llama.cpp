<script lang="ts">
	import { cn } from '$lib/components/ui/utils';
	import { Tabs as TabsPrimitive } from 'bits-ui';

	let {
		class: className,
		ref = $bindable(null),
		...restProps
	}: TabsPrimitive.ContentProps = $props();
</script>

<TabsPrimitive.Content
	bind:ref
	class={cn('text-sm flex-1 outline-none', className)}
	data-slot="tabs-content"
	{...restProps}
/>
