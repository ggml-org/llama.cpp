<script lang="ts">
	import { ICON_CLASS_DEFAULT } from '$lib/constants/css-classes';
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { Search } from '@lucide/svelte';
	import { ActionIcon, KeyboardShortcutInfo, SearchInput } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import {
		ICON_STRIP_TRANSITION_DURATION,
		ICON_STRIP_TRANSITION_DELAY_MULTIPLIER,
		ROUTES,
		SIDEBAR_ACTIONS_ITEMS
	} from '$lib/constants';
	import { isMobile } from '$lib/stores/viewport.svelte';
	import { TooltipSide } from '$lib/enums';
	import { fade } from 'svelte/transition';
	import { circIn } from 'svelte/easing';
	import { onMount } from 'svelte';
	import type { Component } from 'svelte';

	interface Props {
		class: string;
		isExpandedMode: boolean;
		isSearchModeActive: boolean;
		searchQuery: string;
		onSearchDeactivated?: () => void;
		onSearchClick?: () => void;
		onNewChat?: () => void;
	}

	let {
		class: className,
		isExpandedMode = false,
		isSearchModeActive = $bindable(false),
		searchQuery = $bindable(''),
		onSearchDeactivated,
		onSearchClick,
		onNewChat
	}: Props = $props();

	let initialized = $state(false);
	let showIcons = $state(false);
	let searchInputRef = $state<HTMLInputElement | null>(null);

	const isOnMobile = $derived(isMobile.current);

	$effect(() => {
		if (isSearchModeActive && searchInputRef) {
			searchInputRef.focus();
		}
	});

	onMount(() => {
		showIcons = true;

		setTimeout(() => {
			initialized = true;
		}, ICON_STRIP_TRANSITION_DELAY_MULTIPLIER * SIDEBAR_ACTIONS_ITEMS.length);
	});

	function handleSearchModeDeactivate() {
		isSearchModeActive = false;
		searchQuery = '';
		onSearchDeactivated?.();
	}

	function isItemActive(item: {
		activeRouteId?: string;
		activeRoutePrefix?: string;
		activeUrlIncludes?: string;
	}): boolean {
		if (item.activeRouteId) {
			return page.route.id === item.activeRouteId;
		}

		if (item.activeRoutePrefix) {
			return !!page.route.id?.startsWith(item.activeRoutePrefix);
		}

		if (item.activeUrlIncludes) {
			return page.url?.hash?.includes(item.activeUrlIncludes) ?? false;
		}

		return false;
	}
</script>

{#snippet itemIcon(IconComponent: Component)}
	<IconComponent class={ICON_CLASS_DEFAULT} />
{/snippet}

{#if isSearchModeActive}
	<div class="px-4 my-2">
		<SearchInput
			bind:value={searchQuery}
			bind:ref={searchInputRef}
			onClose={handleSearchModeDeactivate}
			onKeyDown={(e) => e.key === 'Escape' && handleSearchModeDeactivate()}
			placeholder="Search conversations..."
		/>
	</div>
{:else if isExpandedMode || isOnMobile}
	<div
		class="{className} flex flex-col gap-5 md:gap-1 mt-2 md:mt-0 {!isExpandedMode && isOnMobile
			? 'hidden pointer-events-none'
			: ''}"
	>
		{#each SIDEBAR_ACTIONS_ITEMS as item, i (item.tooltip)}
			{@const isActive = isItemActive(item)}
			{@const isSearchOnMobile = item.icon === Search && isMobile.current}
			{@const itemHref = isSearchOnMobile ? ROUTES.SEARCH : item.route}
			{@const itemOnClick = item.route
				? () => {
						onNewChat?.();
						goto(item.route!);
					}
				: isSearchOnMobile
					? undefined
					: onSearchClick}
			{@const itemTransition = {
				duration: ICON_STRIP_TRANSITION_DURATION,
				delay: !initialized ? i * ICON_STRIP_TRANSITION_DELAY_MULTIPLIER : 0,
				easing: circIn
			}}

			{#if showIcons}
				<div transition:fade={itemTransition}>
					<Button
						class="w-full min-w-9 justify-between px-2 backdrop-blur-none! hover:[&>kbd]:opacity-100 {isActive
							? 'bg-accent text-accent-foreground'
							: ''}"
						href={itemHref}
						onclick={itemOnClick}
						variant="ghost"
						size="default"
					>
						<span class="flex min-w-0 items-center px-0.5 gap-2">
							{@render itemIcon(item.icon)}

							{#if showIcons}
								<span in:fade={itemTransition} out:fade={itemTransition} class="min-w-0 truncate"
									>{item.tooltip}</span
								>
							{/if}
						</span>

						{#if item.keys}
							<KeyboardShortcutInfo keys={item.keys} />
						{/if}
					</Button>
				</div>
			{/if}
		{/each}
	</div>
{:else}
	<div class="{className} flex-col gap-1 hidden md:flex">
		{#each SIDEBAR_ACTIONS_ITEMS as item, i (item.tooltip)}
			{@const isActive = isItemActive(item)}
			{@const isSearchOnMobile = item.icon === Search && isMobile.current}
			{@const itemOnClick = item.route
				? () => {
						onNewChat?.();
						goto(item.route!);
					}
				: isSearchOnMobile
					? undefined
					: onSearchClick}
			{@const itemTransition = {
				duration: ICON_STRIP_TRANSITION_DURATION,
				delay: !initialized ? i * ICON_STRIP_TRANSITION_DELAY_MULTIPLIER : 0,
				easing: circIn
			}}

			{#if showIcons}
				<div transition:fade={itemTransition}>
					<ActionIcon
						icon={item.icon}
						tooltip={item.tooltip}
						tooltipSide={TooltipSide.RIGHT}
						size="lg"
						iconSize={ICON_CLASS_DEFAULT}
						class="h-9 w-9 rounded-full hover:bg-accent! {isActive
							? 'bg-accent text-accent-foreground'
							: ''}"
						onclick={itemOnClick}
					/>
				</div>
			{/if}
		{/each}
	</div>
{/if}
