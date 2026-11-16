<script lang="ts">
	import { cn } from '../utils.js';
	import { DropdownMenu as DropdownMenuPrimitive } from 'bits-ui';

	let {
		class: className,
		portalProps,
		ref = $bindable(null),
		...restProps
	}: DropdownMenuPrimitive.SubContentProps & {
		portalProps?: DropdownMenuPrimitive.PortalProps;
	} = $props();
</script>

<!-- A submenu leaves the menu it opens from: inside it, the menu's scroll box (and the filter and
     transform of its open animation) would clip the submenu wherever it sits outside that box -->
<DropdownMenuPrimitive.Portal {...portalProps}>
	<DropdownMenuPrimitive.SubContent
		bind:ref
		class={cn(
			'z-50 max-h-(--bits-dropdown-menu-content-available-height) min-w-[8rem] origin-(--bits-dropdown-menu-content-transform-origin) overflow-x-hidden overflow-y-auto rounded-md border matte-float p-1.5 text-popover-foreground outline-none data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:fill-mode-forwards data-[state=closed]:zoom-out-95 data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95',
			className
		)}
		data-slot="dropdown-menu-sub-content"
		{...restProps}
	/>
</DropdownMenuPrimitive.Portal>
