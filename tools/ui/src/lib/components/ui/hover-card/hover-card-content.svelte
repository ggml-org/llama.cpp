<script lang="ts">
	import HoverCardPortal from './hover-card-portal.svelte';
	import { cn, type WithoutChildrenOrChild } from '$lib/components/ui/utils.js';
	import { LinkPreview as HoverCardPrimitive } from 'bits-ui';
	import type { ComponentProps } from 'svelte';

	let {
		align = 'center',
		class: className,
		portalProps,
		ref = $bindable(null),
		sideOffset = 4,
		...restProps
	}: HoverCardPrimitive.ContentProps & {
		portalProps?: WithoutChildrenOrChild<ComponentProps<typeof HoverCardPortal>>;
	} = $props();
</script>

<HoverCardPortal {...portalProps}>
	<HoverCardPrimitive.Content
		bind:ref
		{align}
		class={cn(
			'z-50 w-64 origin-(--bits-link-preview-content-transform-origin) rounded-xl border matte-float p-1.5 text-sm text-popover-foreground outline-hidden data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:fill-mode-forwards data-[state=closed]:zoom-out-95 data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95',
			className
		)}
		data-slot="hover-card-content"
		{sideOffset}
		{...restProps}
	/>
</HoverCardPortal>
