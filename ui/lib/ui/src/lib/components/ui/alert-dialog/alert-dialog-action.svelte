<script lang="ts">
	import { type ButtonVariant, buttonVariants } from '../button/index.js';
	import { cn } from '../utils.js';
	import { AlertDialog as AlertDialogPrimitive } from 'bits-ui';

	let {
		class: className,
		ref = $bindable(null),
		variant = 'default',
		...restProps
	}: AlertDialogPrimitive.ActionProps & { variant?: ButtonVariant } = $props();
</script>

<AlertDialogPrimitive.Action
	bind:ref
	class={cn(buttonVariants({ variant }), className)}
	data-slot="alert-dialog-action"
	{...restProps}
/>
