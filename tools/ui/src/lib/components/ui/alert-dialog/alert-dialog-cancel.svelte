<script lang="ts">
	import { AlertDialog as AlertDialogPrimitive } from 'bits-ui';
	import { buttonVariants } from '$lib/components/ui/button/index.js';
	import { cn } from '$lib/components/ui/utils.js';

	let {
		ref = $bindable(null),
		class: className,
		...restProps
	}: AlertDialogPrimitive.CancelProps = $props();
</script>

<AlertDialogPrimitive.Cancel
	bind:ref
	data-slot="alert-dialog-cancel"
	class={cn(buttonVariants({ variant: 'outline' }), className)}
	{...restProps}
/>
