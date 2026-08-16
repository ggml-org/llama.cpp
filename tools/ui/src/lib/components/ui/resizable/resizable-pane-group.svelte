<script lang="ts">
	import { cn } from '$lib/components/ui/utils.js';
	import * as ResizablePrimitive from 'paneforge';

	let {
		class: className,
		ref = $bindable(null),
		this: paneGroup = $bindable(),
		...restProps
	}: ResizablePrimitive.PaneGroupProps & {
		this?: ResizablePrimitive.PaneGroup;
	} = $props();
</script>

<ResizablePrimitive.PaneGroup
	bind:ref
	bind:this={paneGroup}
	data-slot="resizable-pane-group"
	class={cn(
		'cn-resizable-panel-group flex h-full w-full data-[direction=vertical]:flex-col',
		className
	)}
	{...restProps}
/>
