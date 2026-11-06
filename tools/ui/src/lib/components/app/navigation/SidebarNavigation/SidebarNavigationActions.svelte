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
	import type { Component } from 'svelte';
	import { onMount } from 'svelte';
	import { circIn } from 'svelte/easing';
	import { fade } from 'svelte/transition';

	interface Props {
		class: string;
		onAction?: (action: SidebarAction) => void;
		/** Called before following a route item, so a phone can collapse the sidebar. */
		onNavigate?: () => void;
	}

	let { class: className, onAction, onNavigate }: Props = $props();

	// the models manager stays desktop-only: a phone picks its models in the selector
	const actions = $derived(
		SIDEBAR_ACTIONS_ITEMS.filter(
			(item) => item.action !== SidebarAction.MANAGE_MODELS || !deviceStore.isMobile
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
			return () => {
				onNavigate?.();
				goto(route);
			};
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
	<IconComponent class={ICON_CLASS_DEFAULT} />
{/snippet}

<div class="{className} flex flex-col gap-5 md:gap-1 mt-2 md:mt-0">
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
					class="w-full min-w-9 justify-between px-2 backdrop-blur-none! hover:[&>kbd]:opacity-100 {isActive
						? 'bg-accent text-accent-foreground'
						: ''}"
					href={item.route}
					onclick={itemOnClick}
					size="default"
					variant="ghost"
				>
					<span class="flex min-w-0 items-center px-0.5 gap-2">
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
