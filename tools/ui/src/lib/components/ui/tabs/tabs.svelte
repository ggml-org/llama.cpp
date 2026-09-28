<script lang="ts">
	import { cn } from '$lib/components/ui/utils';
	import { Tabs as TabsPrimitive } from 'bits-ui';

	let {
		class: className,
		ref = $bindable(null),
		value = $bindable(''),
		...restProps
	}: TabsPrimitive.RootProps = $props();
</script>

<TabsPrimitive.Root
	bind:ref
	bind:value
	class={cn('gap-2 group/tabs flex data-[orientation=horizontal]:flex-col', className)}
	data-slot="tabs"
	{...restProps}
/>
