<script lang="ts">
	import { cn } from '$lib/components/ui/utils.js';
	import { ContextMenu as ContextMenuPrimitive } from 'bits-ui';

	let {
		class: className,
		portalProps,
		ref = $bindable(null),
		...restProps
	}: ContextMenuPrimitive.ContentProps & {
		portalProps?: ContextMenuPrimitive.PortalProps;
	} = $props();
</script>

<ContextMenuPrimitive.Portal {...portalProps}>
	<ContextMenuPrimitive.Content
		bind:ref
		class={cn(
			'z-50 max-h-[calc(var(--bits-context-menu-content-available-height)-1rem)] min-w-[8rem] origin-(--bits-context-menu-content-transform-origin) overflow-x-hidden overflow-y-auto rounded-xl border border-border bg-popover p-1.5 text-popover-foreground shadow-md outline-none data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:fill-mode-forwards data-[state=closed]:zoom-out-95 data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95 dark:border-border/20',
			className
		)}
		data-slot="context-menu-content"
		{...restProps}
	/>
</ContextMenuPrimitive.Portal>
