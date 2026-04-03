<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { ActionIcon } from '$lib/components/app/actions';
	import { McpLogo } from '$lib/components/app/mcp';
	import { Database, Settings, Search, SquarePen } from '@lucide/svelte';
	import { fade } from 'svelte/transition';
	import { onMount, type Component } from 'svelte';

	interface Props {
		sidebarOpen: boolean;
		onSearchClick: () => void;
	}

	let { sidebarOpen, onSearchClick }: Props = $props();

	const TRANSITION_DURATION = 200;

	let isMcpActive = $derived(page.route.id === '/settings/mcp');
	let isImportExportActive = $derived(page.route.id === '/settings/import-export');
	let isSettingsActive = $derived(page.route.id === '/settings/chat');

	interface IconItem {
		icon: Component;
		tooltip: string;
		onclick: () => void;
		activeClass?: string;
		group: 'top' | 'bottom';
	}

	let icons = $derived<IconItem[]>([
		{
			icon: SquarePen,
			tooltip: 'New Chat',
			onclick: () => goto('?new_chat=true#/'),
			group: 'top'
		},
		{
			icon: Search,
			tooltip: 'Search',
			onclick: onSearchClick,
			group: 'top'
		},
		{
			icon: McpLogo,
			tooltip: 'MCP Servers',
			onclick: () => goto('#/settings/mcp'),
			activeClass: isMcpActive ? 'bg-accent text-accent-foreground' : '',
			group: 'bottom'
		},
		{
			icon: Database,
			tooltip: 'Import / Export',
			onclick: () => goto('#/settings/import-export'),
			activeClass: isImportExportActive ? 'bg-accent text-accent-foreground' : '',
			group: 'bottom'
		},
		{
			icon: Settings,
			tooltip: 'Settings',
			onclick: () => goto('#/settings/chat'),
			activeClass: isSettingsActive ? 'bg-accent text-accent-foreground' : '',
			group: 'bottom'
		}
	]);

	let topIcons = $derived(icons.filter((i) => i.group === 'top'));
	let bottomIcons = $derived(icons.filter((i) => i.group === 'bottom'));

	let mounted = $state(false);
	onMount(() => (mounted = true));
	let showIcons = $derived(mounted && !sidebarOpen);
</script>

<!-- Spacer to reserve space for icon strip in the flex layout -->
<div
	class="hidden shrink-0 transition-[width] duration-200 ease-linear md:block {sidebarOpen
		? 'w-0'
		: 'w-[calc(var(--sidebar-width-icon)+1.5rem)]'}"
></div>
<!-- Desktop: icon strip, fixed position so it stays stationary and only fades -->
<aside
	class="fixed top-0 bottom-0 left-0 z-10 hidden w-[calc(var(--sidebar-width-icon)+1.5rem)] flex-col items-center justify-between py-3 transition-opacity duration-200 ease-linear md:flex {sidebarOpen
		? 'pointer-events-none opacity-0'
		: 'opacity-100'}"
>
	<div class="mt-12 flex flex-col items-center gap-1">
		{#each topIcons as item (item.tooltip)}
			{#if showIcons}
				<div in:fade={{ duration: TRANSITION_DURATION, delay: TRANSITION_DURATION }}>
					<ActionIcon
						icon={item.icon}
						tooltip={item.tooltip}
						size="lg"
						iconSize="h-4 w-4"
						class="h-9 w-9 rounded-full hover:bg-accent! {item.activeClass ?? ''}"
						onclick={item.onclick}
					/>
				</div>
			{/if}
		{/each}
	</div>
	<div class="flex flex-col items-center gap-1">
		{#each bottomIcons as item (item.tooltip)}
			{#if showIcons}
				<div in:fade={{ duration: TRANSITION_DURATION, delay: TRANSITION_DURATION }}>
					<ActionIcon
						icon={item.icon}
						tooltip={item.tooltip}
						size="lg"
						iconSize="h-4 w-4"
						class="h-9 w-9 rounded-full hover:bg-accent! {item.activeClass ?? ''}"
						onclick={item.onclick}
					/>
				</div>
			{/if}
		{/each}
	</div>
</aside>
