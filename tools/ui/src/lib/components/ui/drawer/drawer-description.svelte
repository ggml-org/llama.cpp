<script lang="ts">
	import { cn } from '$lib/components/ui/utils.js';
	import { Drawer as DrawerPrimitive } from 'vaul-svelte';

	let {
		class: className,
		ref = $bindable(null),
		...restProps
	}: DrawerPrimitive.DescriptionProps = $props();
</script>

<DrawerPrimitive.Description
	bind:ref
	class={cn('text-muted-foreground text-sm', className)}
	data-slot="drawer-description"
	{...restProps}
/>
