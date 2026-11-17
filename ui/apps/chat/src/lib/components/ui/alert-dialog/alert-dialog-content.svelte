<script lang="ts">
	import AlertDialogOverlay from './alert-dialog-overlay.svelte';
	import { cn, type WithoutChild, type WithoutChildrenOrChild } from '$lib/components/ui/utils.js';
	import { AlertDialog as AlertDialogPrimitive } from 'bits-ui';

	let {
		class: className,
		portalProps,
		ref = $bindable(null),
		...restProps
	}: WithoutChild<AlertDialogPrimitive.ContentProps> & {
		portalProps?: WithoutChildrenOrChild<AlertDialogPrimitive.PortalProps>;
	} = $props();
</script>

<AlertDialogPrimitive.Portal {...portalProps}>
	<AlertDialogOverlay />

	<AlertDialogPrimitive.Content
		bind:ref
		class={cn(
			'fixed z-[999999] grid gap-4 border border-border bg-popover p-6 text-popover-foreground shadow-md duration-200 dark:border-border/20',
			// centered on every screen: an alert is short, so a phone shows it as a card of its
			// own height rather than as a drawer from the bottom, which it is not. The phone's
			// 90% is a width, so a caller's max-w-* still caps it on wider screens
			'top-[50%] left-[50%] max-h-[calc(100dvh-2rem)] w-[90%] translate-x-[-50%] translate-y-[-50%] overflow-y-auto rounded-xl sm:w-full sm:max-w-lg',
			'data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[state=closed]:fill-mode-forwards',
			'data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95',
			className
		)}
		data-slot="alert-dialog-content"
		{...restProps}
	/>
</AlertDialogPrimitive.Portal>
