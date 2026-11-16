<script lang="ts">
	import { cn } from '../utils.js';
	import { DropdownMenu as DropdownMenuPrimitive } from 'bits-ui';

	// rows sit 7px inside a menu (its 1px border and p-1.5): these offsets line the submenu's first row
	// up with the row it opens from, and set it 4px past the menu's edge, as a menu is 4px off its trigger
	const ROW_INSET = 7;
	const GAP = 4;

	let {
		align = 'start',
		alignOffset = -ROW_INSET,
		class: className,
		portalProps,
		ref = $bindable(null),
		sideOffset = ROW_INSET + GAP,
		...restProps
	}: DropdownMenuPrimitive.SubContentProps & {
		portalProps?: DropdownMenuPrimitive.PortalProps;
	} = $props();
</script>

<!-- A submenu leaves the menu it opens from: inside it, the menu's scroll box (and the filter and
     transform of its open animation) would clip the submenu wherever it sits outside that box.
     bits-ui sets its shared --bits-menu-* variables on a submenu, not the menu's own names -->
<DropdownMenuPrimitive.Portal {...portalProps}>
	<DropdownMenuPrimitive.SubContent
		bind:ref
		{align}
		{alignOffset}
		class={cn(
			'z-50 max-h-[calc(var(--bits-menu-content-available-height)-1rem)] min-w-[8rem] origin-(--bits-menu-content-transform-origin) overflow-x-hidden overflow-y-auto rounded-xl border matte-float p-1.5 text-popover-foreground outline-none data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:fill-mode-forwards data-[state=closed]:zoom-out-95 data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95',
			className
		)}
		data-slot="dropdown-menu-sub-content"
		{sideOffset}
		{...restProps}
	/>
</DropdownMenuPrimitive.Portal>
