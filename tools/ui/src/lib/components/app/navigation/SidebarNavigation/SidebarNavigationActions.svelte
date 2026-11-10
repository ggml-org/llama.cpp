<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { KeyboardShortcutInfo } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import {
		ICON_CLASS_DEFAULT,
		ICON_STRIP_TRANSITION_DELAY_MULTIPLIER,
		ICON_STRIP_TRANSITION_DURATION,
		SIDEBAR_ACTIONS_ITEMS
	} from '$lib/constants';
	import { SidebarAction } from '$lib/enums';
	import { deviceStore } from '$lib/stores';
	import type { DesktopIconStripItem } from '$lib/types';
	import { hapticTap } from '$lib/utils';
	import type { Component } from 'svelte';
	import { onMount } from 'svelte';
	import { circIn } from 'svelte/easing';
	import { fade } from 'svelte/transition';

	interface Props {
		class: string;
		/** The items to show: the sidebar's top actions unless given, e.g. its footer's. */
		items?: DesktopIconStripItem[];
		onAction?: (action: SidebarAction) => void;
		/** Follows a route item, so a phone can open its page behind the sidebar and then close it. */
		onNavigate?: (go: () => Promise<unknown>) => void;
	}

	let { class: className, items = SIDEBAR_ACTIONS_ITEMS, onAction, onNavigate }: Props = $props();

	// the models manager stays desktop-only: a phone picks its models in the selector; a phone
	// reaches search from the round key at the top of the sidebar instead
	const actions = $derived(
		items.filter(
			(item) =>
				!deviceStore.isMobile ||
				(item.action !== SidebarAction.MANAGE_MODELS && item.action !== SidebarAction.SEARCH)
		)
	);
	let initialized = $state(false);
	let showIcons = $state(false);

	onMount(() => {
		showIcons = true;

		setTimeout(() => {
			initialized = true;
		}, ICON_STRIP_TRANSITION_DELAY_MULTIPLIER * actions.length);
	});

	/** Click handler of one icon strip item: a sidebar action or a route. */
	function itemClickHandler(item: DesktopIconStripItem): (() => void) | undefined {
		const { action, route } = item;

		if (action) {
			return () => onAction?.(action);
		}

		if (route) {
			return () => (onNavigate ? onNavigate(() => goto(route)) : void goto(route));
		}

		return undefined;
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
	<IconComponent class="{ICON_CLASS_DEFAULT} max-md:size-5.5" />
{/snippet}

<!-- a phone sizes its rows for a thumb, flush one under the other like a list's -->
<div class="{className} flex flex-col md:gap-1">
	{#each actions as item, i (item.tooltip)}
		{@const isActive = isItemActive(item)}
		{@const itemOnClick = itemClickHandler(item)}
		{@const itemTransition = {
			delay: !initialized ? i * ICON_STRIP_TRANSITION_DELAY_MULTIPLIER : 0,
			duration: ICON_STRIP_TRANSITION_DURATION,
			easing: circIn
		}}

		{#if showIcons}
			<div transition:fade={itemTransition}>
				<Button
					class="w-full min-w-9 justify-between px-2 backdrop-blur-none! hover:[&>kbd]:opacity-100 rounded-(--surface-row-radius) max-md:h-12 max-md:rounded-2xl max-md:px-3 max-md:text-[17px] {isActive
						? // raised and edged, where a hover only tints: bg-accent matched the hover in light
							// and sank below an inset group's fill in dark
							'bg-background text-accent-foreground shadow-sm ring-1 ring-border/50 hover:bg-background dark:bg-foreground/15 dark:hover:bg-foreground/15'
						: ''}"
					href={item.route}
					onclick={() => {
						// a phone confirms the tap by feel, as a native list does
						if (deviceStore.isMobile) hapticTap();

						itemOnClick?.();
					}}
					size="default"
					variant="ghost"
				>
					<span class="flex min-w-0 items-center gap-2 px-0.5 max-md:gap-3.5">
						{@render itemIcon(item.icon)}

						{#if showIcons}
							<span
								in:fade={itemTransition}
								out:fade={itemTransition}
								class="min-w-0 truncate opacity-(--sidebar-fade)">{item.tooltip}</span
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
