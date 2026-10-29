<script lang="ts">
	import { cn } from '$lib/components/ui/utils.js';
	import { Drawer as DrawerPrimitive } from 'vaul-svelte';

	let {
		class: className,
		ref = $bindable(null),
		...restProps
	}: DrawerPrimitive.OverlayProps = $props();
</script>

<DrawerPrimitive.Overlay
	bind:ref
	class={cn(
		'bg-black/10 supports-backdrop-filter:backdrop-blur-xs fixed inset-0 z-50 data-open:animate-in data-open:fade-in-0 data-closed:animate-out data-closed:fade-out-0',
		className
	)}
	data-slot="drawer-overlay"
	{...restProps}
/>
