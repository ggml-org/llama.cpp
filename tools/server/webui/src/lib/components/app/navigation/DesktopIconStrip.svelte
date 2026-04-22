<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { ActionIcon } from '$lib/components/app/actions';
	import {
		ICON_STRIP_TRANSITION_DURATION,
		ICON_STRIP_TRANSITION_DELAY_MULTIPLIER,
		SIDEBAR_ACTIONS_ITEMS
	} from '$lib/constants';
	import { TooltipSide } from '$lib/enums';
	import { fade } from 'svelte/transition';
	import { circIn } from 'svelte/easing';
	import { onMount } from 'svelte';
	import { useKeyboardShortcuts } from '$lib/hooks/use-keyboard-shortcuts.svelte';

	interface Props {
		sidebarOpen: boolean;
		onSearchClick: () => void;
	}

	let { sidebarOpen = false, onSearchClick }: Props = $props();

	const { handleKeydown } = useKeyboardShortcuts({ activateSearchMode: () => onSearchClick() });

	let initialized = $state(false);
	let showIcons = $derived(!sidebarOpen);

	showIcons = false;

	onMount(() => {
		showIcons = !sidebarOpen;

		setTimeout(() => {
			initialized = true;
		}, ICON_STRIP_TRANSITION_DELAY_MULTIPLIER * SIDEBAR_ACTIONS_ITEMS.length);
	});
</script>

<svelte:window onkeydown={handleKeydown} />

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
		{#each SIDEBAR_ACTIONS_ITEMS as item, i (item.tooltip)}
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
						delay: !initialized
							? ICON_STRIP_TRANSITION_DELAY_MULTIPLIER + i * ICON_STRIP_TRANSITION_DELAY_MULTIPLIER
							: 0,
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
