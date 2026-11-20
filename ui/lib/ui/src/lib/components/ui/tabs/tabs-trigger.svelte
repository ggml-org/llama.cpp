<script lang="ts">
	import { cn } from '../utils.js';
	import { Tabs as TabsPrimitive } from 'bits-ui';

	let {
		class: className,
		ref = $bindable(null),
		...restProps
	}: TabsPrimitive.TriggerProps = $props();
</script>

<!-- muted until active; the list draws the raised chip or underline and slides it between triggers -->
<TabsPrimitive.Trigger
	bind:ref
	class={cn(
		"relative inline-flex h-full flex-1 items-center justify-center gap-1.5 rounded-md px-3 text-label font-medium whitespace-nowrap transition-colors duration-250 ease-out-quint outline-none hover:text-(--fg-default) focus-visible:ring-3 focus-visible:ring-ring/50 disabled:pointer-events-none disabled:opacity-50 group-data-[orientation=vertical]/tabs:w-full group-data-[orientation=vertical]/tabs:justify-start group-data-[orientation=vertical]/tabs:py-2 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-icon-md",
		'data-[state=active]:text-(--fg-default)',
		className
	)}
	data-slot="tabs-trigger"
	{...restProps}
/>
