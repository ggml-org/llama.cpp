<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { ActionIcon } from '$lib/components/app/actions';
	import {
		ICON_STRIP_TRANSITION_DURATION,
		ICON_STRIP_TRANSITION_DELAY_MULTIPLIER,
		DESKTOP_ICON_STRIP_ICONS
	} from '$lib/constants';
	import { TooltipSide } from '$lib/enums';
	import { fade } from 'svelte/transition';
	import { circIn } from 'svelte/easing';
	import { onMount } from 'svelte';

	interface Props {
		sidebarOpen: boolean;
		onSearchClick: () => void;
	}

	let { sidebarOpen, onSearchClick }: Props = $props();

	let mounted = $state(false);
	onMount(() => (mounted = true));

	let showIcons = $derived(mounted && !sidebarOpen);
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
		{#each DESKTOP_ICON_STRIP_ICONS as item, i (item.tooltip)}
			{@const onclick = item.route ? () => goto(item.route!) : onSearchClick}
			{@const isActive = item.activeRouteId
				? page.route.id === item.activeRouteId
				: item.activeRoutePrefix
					? !!page.route.id?.startsWith(item.activeRoutePrefix)
					: false}
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
						class="h-9 w-9 rounded-full hover:bg-accent! {isActive
							? 'bg-accent text-accent-foreground'
							: ''}"
						{onclick}
					/>
				</div>
			{/if}
		{/each}
	</div>
</aside>
