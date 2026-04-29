<script lang="ts">
	import { Button } from '$lib/components/ui/button/index.js';
	import PanelLeftIcon from '@lucide/svelte/icons/panel-left';
	import type { ComponentProps } from 'svelte';
	import { useSidebar } from './context.svelte.js';
	import { PanelLeftClose } from '@lucide/svelte';

	let {
		ref = $bindable(null),
		class: className,
		onclick,
		...restProps
	}: ComponentProps<typeof Button> & {
		onclick?: (e: MouseEvent) => void;
	} = $props();

	const sidebar = useSidebar();
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
	{...restProps}
>
	{#if sidebar.open}
		<PanelLeftClose />
	{:else}
		<PanelLeftIcon />
	{/if}
	<span class="sr-only">Toggle Sidebar</span>
</Button>
