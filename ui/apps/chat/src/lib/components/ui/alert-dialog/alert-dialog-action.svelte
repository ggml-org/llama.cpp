<script lang="ts">
	import { buttonVariants } from '#lib/components/ui/button/index.ts';
	import { cn } from '#lib/components/ui/utils.ts';
	import { AlertDialog as AlertDialogPrimitive } from 'bits-ui';

	let {
		class: className,
		ref = $bindable(null),
		...restProps
	}: AlertDialogPrimitive.ActionProps = $props();
</script>

<AlertDialogPrimitive.Action
	bind:ref
	class={cn(buttonVariants(), className)}
	data-slot="alert-dialog-action"
	{...restProps}
/>
