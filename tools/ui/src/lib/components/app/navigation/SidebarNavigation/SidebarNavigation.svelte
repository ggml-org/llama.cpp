<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { PanelLeftClose, Settings } from '@lucide/svelte';
	import { ActionIcon, KeyboardShortcutInfo, Logo } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import {
		ICON_STRIP_TRANSITION_DURATION,
		ICON_STRIP_TRANSITION_DELAY_MULTIPLIER,
		SIDEBAR_ACTIONS_ITEMS,
		ROUTES
	} from '$lib/constants';
	import { TooltipSide } from '$lib/enums';
	import { fade, scale } from 'svelte/transition';
	import { circIn } from 'svelte/easing';
	import { onMount } from 'svelte';
	import type { Component } from 'svelte';
	import { useKeyboardShortcuts } from '$lib/hooks/use-keyboard-shortcuts.svelte';
	import { conversationsStore, conversations } from '$lib/stores/conversations.svelte';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { RouterService } from '$lib/services/router.service';
	import SidebarNavigationConversationList from './SidebarNavigationConversationList.svelte';
	import SidebarNavigationActions from './SidebarNavigationActions.svelte';

	interface Props {
		onSearchClick?: () => void;
	}

	let { onSearchClick = () => {} }: Props = $props();

	const { handleKeydown } = useKeyboardShortcuts({ activateSearchMode: () => onSearchClick() });

	let initialized = $state(false);
	let showIcons = $state(false);

	onMount(() => {
		showIcons = true;

		setTimeout(() => {
			initialized = true;
		}, ICON_STRIP_TRANSITION_DELAY_MULTIPLIER * SIDEBAR_ACTIONS_ITEMS.length);
	});

	let isExpandedMode = $state(false);
	let hoveredTooltip = $state<string | null>(null);

	const isStripExpanded = $derived(isExpandedMode || hoveredTooltip !== null);

	function toggleExpandedMode() {
		isExpandedMode = !isExpandedMode;
		if (!isExpandedMode) {
			hoveredTooltip = null;
		}
	}

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

	// --- Conversation list state ---

	let currentChatId = $derived(page.params.id);
	let isSearchModeActive = $state(false);
	let searchQuery = $state('');

	let filteredConversations = $derived.by(() => {
		if (isSearchModeActive) {
			if (searchQuery.trim().length > 0) {
				return conversations().filter((conversation: { name: string }) =>
					conversation.name.toLowerCase().includes(searchQuery.toLowerCase())
				);
			}

			return [];
		}

		return conversations();
	});

	async function selectConversation(id: string) {
		await goto(RouterService.chat(id));
	}

	async function handleEditConversation(id: string) {
		const conversation = conversations().find((conv) => conv.id === id);
		if (!conversation) return;

		const newName = window.prompt('Rename conversation', conversation.name);
		if (newName && newName.trim()) {
			await conversationsStore.updateConversationName(id, newName.trim());
		}
	}

	async function handleDeleteConversation(id: string) {
		const conversation = conversations().find((conv) => conv.id === id);
		if (!conversation) return;

		const confirmed = window.confirm(
			`Delete "${conversation.name}"? This action cannot be undone.`
		);
		if (!confirmed) return;

		await conversationsStore.deleteConversation(id, { deleteWithForks: false });
	}

	function handleStopGeneration(id: string) {
		chatStore.stopGenerationForChat(id);
	}
</script>

{#snippet itemIcon(IconComponent: Component)}
	<IconComponent class="h-4 w-4" />
{/snippet}

<svelte:window onkeydown={handleKeydown} />

<aside
	class="fixed top-2 bottom-2 left-2 py-2 rounded-xl z-10 hidden flex-col justify-between transition-[width,padding] duration-200 ease-out md:flex {isStripExpanded
		? 'w-72 bg-muted/50 backdrop-blur-xl'
		: 'w-12'}"
>
	<div class="px-2 flex items-center justify-between">
		<ActionIcon
			icon={Logo}
			size="lg"
			iconSize="h-4 w-4"
			class="h-9 w-9 rounded-full hover:bg-accent!"
			href={isExpandedMode ? ROUTES.START : undefined}
			onclick={isExpandedMode ? undefined : toggleExpandedMode}
			ariaLabel={isExpandedMode ? 'Go to start' : 'Expand navigation'}
		/>

		{#if isExpandedMode}
			<div transition:fade={{ duration: 150, easing: circIn, delay: 50 }}>
				<ActionIcon
					icon={PanelLeftClose}
					size="lg"
					iconSize="h-4 w-4"
					class="h-9 w-9 rounded-full mr-1 hover:bg-accent!"
					onclick={toggleExpandedMode}
					ariaLabel="Collapse navigation"
				/>
			</div>
		{/if}
	</div>

	<div class="mt-2 flex min-h-0 flex-1 flex-col justify-between gap-1 px-2">
		<SidebarNavigationActions {isExpandedMode} />

		{#if isExpandedMode}
			<div
				class="flex min-h-0 flex-1 flex-col"
				transition:fade={{ duration: 150, easing: circIn }}
			>
				<SidebarNavigationConversationList
					{filteredConversations}
					{currentChatId}
					{isSearchModeActive}
					{searchQuery}
					onSelect={selectConversation}
					onEdit={handleEditConversation}
					onDelete={handleDeleteConversation}
					onStop={handleStopGeneration}
				/>
			</div>
		{/if}

		<div class="flex flex-col gap-0.5">
			<div role="presentation">
				{#if isExpandedMode || hoveredTooltip === 'Settings'}
					<div transition:scale={{ duration: 150, start: 0.9, opacity: 0 }} class="origin-left">
						<Button
							class="w-full justify-between px-2 backdrop-blur-none! hover:[&>kbd]:opacity-100"
							href={ROUTES.SETTINGS}
							variant="ghost"
						>
							<div class="flex items-center gap-2">
								<Settings class="h-4 w-4" />

								Settings
							</div>
						</Button>
					</div>
				{:else}
					<ActionIcon
						icon={Settings}
						size="lg"
						iconSize="h-4 w-4"
						class="h-9 w-9 rounded-full hover:bg-accent!"
						href={ROUTES.SETTINGS}
						tooltip="Settings"
						tooltipSide={TooltipSide.RIGHT}
					/>
				{/if}
			</div>
		</div>
	</div>
</aside>
