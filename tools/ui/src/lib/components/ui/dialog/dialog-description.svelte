<script lang="ts">
	import { cn } from '$lib/components/ui/utils';
	import { Dialog as DialogPrimitive } from 'bits-ui';

	let {
		class: className,
		ref = $bindable(null),
		...restProps
	}: DialogPrimitive.DescriptionProps = $props();
</script>

<DialogPrimitive.Description
	bind:ref
	class={cn('text-sm text-muted-foreground', className)}
	data-slot="dialog-description"
	{...restProps}
/>
