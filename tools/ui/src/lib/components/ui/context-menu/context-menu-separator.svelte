<script lang="ts">
	import { cn } from '$lib/components/ui/utils.js';
	import { ContextMenu as ContextMenuPrimitive } from 'bits-ui';

	let {
		class: className,
		ref = $bindable(null),
		...restProps
	}: ContextMenuPrimitive.SeparatorProps = $props();
</script>

<ContextMenuPrimitive.Separator
	bind:ref
	class={cn('-mx-1 my-1 h-px bg-border/20', className)}
	data-slot="context-menu-separator"
	{...restProps}
/>
