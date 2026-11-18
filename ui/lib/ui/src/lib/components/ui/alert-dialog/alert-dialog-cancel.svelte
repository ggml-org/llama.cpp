<script lang="ts">
	import { buttonVariants } from '../button/index.js';
	import { cn } from '../utils.js';
	import { AlertDialog as AlertDialogPrimitive } from 'bits-ui';

	let {
		class: className,
		ref = $bindable(null),
		...restProps
	}: AlertDialogPrimitive.CancelProps = $props();
</script>

<AlertDialogPrimitive.Cancel
	bind:ref
	class={cn(buttonVariants({ variant: 'outline' }), className)}
	data-slot="alert-dialog-cancel"
	{...restProps}
/>
