<script lang="ts">
	import { AlertDialog as AlertDialogPrimitive } from 'bits-ui';
	import AlertDialogOverlay from './alert-dialog-overlay.svelte';
	import { cn, type WithoutChild, type WithoutChildrenOrChild } from '$lib/components/ui/utils.js';

	let {
		ref = $bindable(null),
		class: className,
		portalProps,
		...restProps
	}: WithoutChild<AlertDialogPrimitive.ContentProps> & {
		portalProps?: WithoutChildrenOrChild<AlertDialogPrimitive.PortalProps>;
	} = $props();
</script>

<AlertDialogPrimitive.Portal {...portalProps}>
	<AlertDialogOverlay />
	<AlertDialogPrimitive.Content
		bind:ref
		data-slot="alert-dialog-content"
		class={cn(
			'fixed z-[999999] grid w-full gap-4 border bg-background p-6 shadow-lg duration-200',
			// Mobile: Bottom sheet behavior
			'right-0 bottom-0 left-0 max-h-[100dvh] translate-x-0 translate-y-0 overflow-y-auto rounded-t-lg',
			'data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:slide-out-to-bottom-full',
			'data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:slide-in-from-bottom-full',
			// Desktop: Centered dialog behavior
			'sm:top-[50%] sm:right-auto sm:bottom-auto sm:left-[50%] sm:max-h-[100vh] sm:max-w-lg sm:translate-x-[-50%] sm:translate-y-[-50%] sm:rounded-lg',
			'sm:data-[state=closed]:slide-out-to-bottom-0 sm:data-[state=closed]:zoom-out-95',
			'sm:data-[state=open]:slide-in-from-bottom-0 sm:data-[state=open]:zoom-in-95',
			className
		)}
		{...restProps}
	/>
</AlertDialogPrimitive.Portal>
