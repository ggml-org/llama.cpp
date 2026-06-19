<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { ActionIcon, KeyboardShortcutInfo, Logo } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import {
		ICON_STRIP_TRANSITION_DURATION,
		ICON_STRIP_TRANSITION_DELAY_MULTIPLIER,
		SIDEBAR_ACTIONS_ITEMS
	} from '$lib/constants';
	import { TooltipSide } from '$lib/enums';
	import { fade } from 'svelte/transition';
	import { circIn } from 'svelte/easing';
	import { onMount } from 'svelte';
	import type { Component } from 'svelte';

	interface Props {
     	isExpandedMode: boolean;
		onSearchClick?: () => void;
	}

	let { isExpandedMode = false, onSearchClick = () => {} }: Props = $props();


	let initialized = $state(false);
	let showIcons = $state(false);

	onMount(() => {
		showIcons = true;

		setTimeout(() => {
			initialized = true;
		}, ICON_STRIP_TRANSITION_DELAY_MULTIPLIER * SIDEBAR_ACTIONS_ITEMS.length);
	});

	function isItemActive(item: { activeRouteId?: string; activeRoutePrefix?: string }): boolean {
		if (item.activeRouteId) {
			return page.route.id === item.activeRouteId;
		}
		if (item.activeRoutePrefix) {
			return !!page.route.id?.startsWith(item.activeRoutePrefix);
		}
		return false;
	}

	function getItemOnClick(item: { route?: string }) {
		return item.route ? () => goto(item.route!) : onSearchClick;
	}
</script>

{#snippet itemIcon(IconComponent: Component)}
	<IconComponent class="h-4 w-4" />
{/snippet}

<div class="flex min-h-0 flex-col gap-0.75">

	{#each SIDEBAR_ACTIONS_ITEMS as item, i (item.tooltip)}
		{@const isActive = isItemActive(item)}
		{@const itemOnClick = getItemOnClick(item)}
		{@const itemTransition = {
			duration: ICON_STRIP_TRANSITION_DURATION,
			delay: !initialized
				? ICON_STRIP_TRANSITION_DELAY_MULTIPLIER +
					i * ICON_STRIP_TRANSITION_DELAY_MULTIPLIER
				: 0,
			easing: circIn
		}}

		<!-- {#if showIcons && isExpandedMode} -->
			<div transition:fade={itemTransition}>
				<Button
					class="ml-0.5 w-full justify-between px-2 backdrop-blur-none! hover:[&>kbd]:opacity-100 {isExpandedMode ? 'rounded-full' : ''} {isActive
						? 'bg-accent text-accent-foreground'
						: ''}"
					href={item.route}
					onclick={itemOnClick}
					variant="ghost"
					size={isExpandedMode ? 'default' : 'icon'}
				>
					<span class="flex min-w-0 items-center gap-2">
						{@render itemIcon(item.icon)}

						{#if showIcons}
    						<span class="min-w-0 truncate">{item.tooltip}</span>
                        {/if}
					</span>

					{#if isExpandedMode && item.keys}
						<KeyboardShortcutInfo keys={item.keys} />
					{/if}
				</Button>
			</div>
		<!-- {:else if showIcons}
			<div transition:fade={itemTransition}>
				<ActionIcon
					icon={item.icon}
					tooltip={item.tooltip}
					tooltipSide={TooltipSide.RIGHT}
					size="lg"
					iconSize="h-4 w-4"
					class="h-9 w-9 rounded-full hover:bg-accent! {isActive
						? 'bg-accent text-accent-foreground'
						: ''}"
					onclick={itemOnClick}
				/>
			</div>
		{/if} -->
	{/each}
</div>
