<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { ActionIcon } from '$lib/components/app/actions';
	import { McpLogo } from '$lib/components/app/mcp';
	import { Database, Settings, Search, SquarePen } from '@lucide/svelte';
	import {
		ICON_STRIP_TRANSITION_DURATION,
		ICON_STRIP_TRANSITION_DELAY_MULTIPLIER
	} from '$lib/constants';
	import { TooltipSide } from '$lib/enums';
	import { fade } from 'svelte/transition';
	import { circIn } from 'svelte/easing';
	import { onMount, type Component } from 'svelte';

	interface Props {
		sidebarOpen: boolean;
		onSearchClick: () => void;
	}

	let { sidebarOpen, onSearchClick }: Props = $props();

	let mounted = $state(false);
	onMount(() => (mounted = true));

	let isMcpActive = $derived(page.route.id === '/settings/mcp');
	let isImportExportActive = $derived(page.route.id === '/settings/import-export');
	let isSettingsActive = $derived(!!page.route.id?.startsWith('/settings/chat'));
	let showIcons = $derived(mounted && !sidebarOpen);

	interface IconItem {
		icon: Component;
		tooltip: string;
		onclick: () => void;
		activeClass?: string;
	}

	let icons = $derived<IconItem[]>([
		{
			icon: SquarePen,
			tooltip: 'New Chat',
			onclick: () => goto('?new_chat=true#/')
		},
		{
			icon: Search,
			tooltip: 'Search',
			onclick: onSearchClick
		},
		{
			icon: McpLogo,
			tooltip: 'MCP Servers',
			onclick: () => goto('#/settings/mcp'),
			activeClass: isMcpActive ? 'bg-accent text-accent-foreground' : ''
		},
		{
			icon: Database,
			tooltip: 'Import / Export',
			onclick: () => goto('#/settings/import-export'),
			activeClass: isImportExportActive ? 'bg-accent text-accent-foreground' : ''
		},
		{
			icon: Settings,
			tooltip: 'Settings',
			onclick: () => goto('#/settings/chat/general'),
			activeClass: isSettingsActive ? 'bg-accent text-accent-foreground' : ''
		}
	]);
</script>

<div
	class="hidden shrink-0 transition-[width] duration-200 ease-linear md:block {sidebarOpen
		? 'w-0'
		: 'w-[calc(var(--sidebar-width-icon)+1.5rem)]'}"
></div>

<aside
	class="fixed top-0 bottom-0 left-0 z-10 hidden w-[calc(var(--sidebar-width-icon)+1.5rem)] flex-col items-center justify-between py-3 transition-opacity duration-200 ease-linear md:flex {sidebarOpen
		? 'pointer-events-none opacity-0'
		: 'opacity-100'}"
>
	<div class="mt-12 flex flex-col items-center gap-1">
		{#each icons as item, i (item.tooltip)}
			{#if showIcons}
				<div
					in:fade={{
						duration: ICON_STRIP_TRANSITION_DURATION,
						delay:
							ICON_STRIP_TRANSITION_DELAY_MULTIPLIER + i * ICON_STRIP_TRANSITION_DELAY_MULTIPLIER,
						easing: circIn
					}}
				>
					<ActionIcon
						icon={item.icon}
						tooltip={item.tooltip}
						tooltipSide={TooltipSide.RIGHT}
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
