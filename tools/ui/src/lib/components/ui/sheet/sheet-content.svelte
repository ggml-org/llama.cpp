<script lang="ts" module>
	import { tv, type VariantProps } from 'tailwind-variants';
	export const sheetVariants = tv({
		base: `border-border/30 dark:border-border/20 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fill-mode-forwards fixed z-50 flex flex-col gap-4 shadow-sm transition ease-in-out data-[state=closed]:duration-300 data-[state=open]:duration-500 ${PANEL_CLASSES}`,
		defaultVariants: {
			side: 'right'
		},
		variants: {
			side: {
				bottom:
					'data-[state=closed]:slide-out-to-bottom data-[state=open]:slide-in-from-bottom inset-x-0 bottom-0 h-auto border-t',
				left: 'data-[state=closed]:slide-out-to-left data-[state=open]:slide-in-from-left inset-y-0 left-0 h-full w-3/4 border-r sm:max-w-sm',
				right:
					'data-[state=closed]:slide-out-to-right data-[state=open]:slide-in-from-right inset-y-0 right-0 h-full w-3/4 border-l sm:max-w-sm',
				top: 'data-[state=closed]:slide-out-to-top data-[state=open]:slide-in-from-top inset-x-0 top-0 h-auto border-b'
			}
		}
	});

	export type Side = VariantProps<typeof sheetVariants>['side'];
</script>

<script lang="ts">
	import SheetOverlay from './sheet-overlay.svelte';
	import XIcon from '@lucide/svelte/icons/x';
	import { cn, type WithoutChildrenOrChild } from '$lib/components/ui/utils.js';
	import { PANEL_CLASSES } from '$lib/constants';
	import { Dialog as SheetPrimitive } from 'bits-ui';
	import type { Snippet } from 'svelte';

	let {
		children,
		class: className,
		portalProps,
		ref = $bindable(null),
		side = 'right',
		...restProps
	}: WithoutChildrenOrChild<SheetPrimitive.ContentProps> & {
		portalProps?: SheetPrimitive.PortalProps;
		side?: Side;
		children: Snippet;
	} = $props();
</script>

<SheetPrimitive.Portal {...portalProps}>
	<SheetOverlay />

	<SheetPrimitive.Content
		bind:ref
		class={cn(sheetVariants({ side }), className)}
		data-slot="sheet-content"
		{...restProps}
	>
		{@render children?.()}

		<SheetPrimitive.Close
			class="absolute top-4 right-4 rounded-xs opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:outline-hidden disabled:pointer-events-none"
		>
			<XIcon class="size-4" />

			<span class="sr-only">Close</span>
		</SheetPrimitive.Close>
	</SheetPrimitive.Content>
</SheetPrimitive.Portal>
