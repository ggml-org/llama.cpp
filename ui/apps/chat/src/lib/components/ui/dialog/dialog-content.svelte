<script lang="ts">
	import * as Dialog from './index.js';
	import XIcon from '@lucide/svelte/icons/x';
	import { cn, type WithoutChildrenOrChild } from '$lib/components/ui/utils';
	import { Dialog as DialogPrimitive } from 'bits-ui';
	import type { Snippet } from 'svelte';

	let {
		children,
		class: className,
		overlayClass,
		portalProps,
		ref = $bindable(null),
		showCloseButton = false,
		...restProps
	}: WithoutChildrenOrChild<DialogPrimitive.ContentProps> & {
		overlayClass?: string;
		portalProps?: DialogPrimitive.PortalProps;
		children: Snippet;
		showCloseButton?: boolean;
	} = $props();
</script>

<Dialog.Portal {...portalProps}>
	<Dialog.Overlay class={overlayClass} />

	<DialogPrimitive.Content
		bind:ref
		class={cn(
			`fixed top-[50%] left-[50%] z-50 grid translate-x-[-50%] translate-y-[-50%] gap-4 overflow-y-auto rounded-lg border border-border/30 bg-background p-6 shadow-lg duration-200 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:fill-mode-forwards data-[state=closed]:zoom-out-95 data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95`,
			'max-h-[100dvh] max-w-full size-auto sm:max-w-lg md:w-auto md:size-auto md:max-w-[calc(100%-2rem)]',
			// a phone shows every dialog full screen, whatever size it asks for on a desktop. The
			// status bar and the home indicator get transparent borders of their own height, so
			// the content keeps clear of them with no change to its padding (see safe-area.css)
			'max-md:top-0! max-md:left-0! max-md:translate-x-0! max-md:translate-y-0! max-md:h-dvh! max-md:max-h-none! max-md:w-screen! max-md:max-w-none! max-md:rounded-none! max-md:border-x-0! max-md:border-t-(length:--safe-area-top)! max-md:border-b-(length:--safe-area-bottom)! max-md:border-transparent!',
			className
		)}
		data-slot="dialog-content"
		{...restProps}
	>
		{@render children?.()}

		{#if showCloseButton}
			<DialogPrimitive.Close
				class="absolute top-4 right-4 rounded-xs opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:ring-2 focus:ring-ring focus:ring-offset-2 focus:outline-hidden disabled:pointer-events-none [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4"
			>
				<XIcon />

				<span class="sr-only">Close</span>
			</DialogPrimitive.Close>
		{/if}
	</DialogPrimitive.Content>
</Dialog.Portal>
