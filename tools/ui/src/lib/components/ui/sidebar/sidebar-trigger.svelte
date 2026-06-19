<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import PanelLeftIcon from '@lucide/svelte/icons/panel-left';
	import type { ComponentProps } from 'svelte';
	import { useSidebar } from './context.svelte.js';
	import { PanelLeftClose } from '@lucide/svelte';
	import { Logo } from '$lib/components/app/misc';
	import { TooltipSide } from '$lib/enums/ui.enums.js';
	import { ActionIcon } from '$lib/components/app';

	let {
		ref = $bindable(null),
		class: className,
		onclick,
		...restProps
	}: ComponentProps<typeof Button> & {
		onclick?: (e: MouseEvent) => void;
	} = $props();

	const sidebar = useSidebar();
	let isHovered = $state(false);
</script>

<Button
	data-sidebar="trigger"
	data-slot="sidebar-trigger"
	variant="ghost"
	size="icon-lg"
	class="rounded-full backdrop-blur-lg {className} {sidebar.open
		? 'top-1.5'
		: 'top-0'} md:left-[calc(var(--sidebar-width)-3.25rem)] {sidebar.isResizing
		? '!duration-0'
		: ''}"
	type="button"
	onclick={(e) => {
		onclick?.(e);
		sidebar.toggle();
	}}
	onmouseenter={() => (isHovered = true)}
	onmouseleave={() => (isHovered = false)}
	{...restProps}
>
    <ActionIcon
		icon={sidebar.open ? PanelLeftClose : isHovered ? PanelLeftIcon : Logo}
		tooltip={!sidebar.open ? 'Open sidebar' : 'Close sidebar'}
		tooltipSide={TooltipSide.RIGHT}
		size="lg"
		iconSize="h-4 w-4"
		class="h-9 w-9 rounded-full hover:bg-accent!"
	/>
</Button>
