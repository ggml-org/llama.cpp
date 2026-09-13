<script lang="ts">
	import { cn } from '$lib/components/ui/utils.js';
	import { Dialog as SheetPrimitive } from 'bits-ui';

	let {
		class: className,
		ref = $bindable(null),
		...restProps
	}: SheetPrimitive.TitleProps = $props();
</script>

<SheetPrimitive.Title
	bind:ref
	class={cn('font-semibold text-foreground', className)}
	data-slot="sheet-title"
	{...restProps}
/>
