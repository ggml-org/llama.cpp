<script lang="ts">
	import { cn } from '$lib/components/ui/utils.js';
	import { Dialog as SheetPrimitive } from 'bits-ui';

	let {
		class: className,
		ref = $bindable(null),
		...restProps
	}: SheetPrimitive.DescriptionProps = $props();
</script>

<SheetPrimitive.Description
	bind:ref
	data-slot="sheet-description"
	class={cn('text-sm text-muted-foreground', className)}
	{...restProps}
/>
