<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { PanelLeftClose, PanelLeftOpen, X } from '@lucide/svelte';
	import {
		ActionIcon,
		Logo,
		SidebarNavigationConversationList,
		SidebarNavigationActions
	} from '$lib/components/app';
	import { ROUTES } from '$lib/constants';
	import { fade } from 'svelte/transition';
	import { circIn } from 'svelte/easing';
	import { useKeyboardShortcuts } from '$lib/hooks/use-keyboard-shortcuts.svelte';
	import { conversationsStore, conversations } from '$lib/stores/conversations.svelte';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { RouterService } from '$lib/services/router.service';
	import { isMobile } from '$lib/stores/viewport.svelte';
	import { TooltipSide } from '$lib/enums';
	import { device } from '$lib/stores/device.svelte';

	interface Props {
		onSearchClick?: () => void;
	}

	let { onSearchClick = () => {} }: Props = $props();

	const { handleKeydown } = useKeyboardShortcuts({ activateSearchMode: () => onSearchClick() });

	let isExpandedMode = $state(false);
	let hoveredTooltip = $state<string | null>(null);
	let logoHovered = $state(false);

	const isStripExpanded = $derived(isExpandedMode || hoveredTooltip !== null);
	const isOnMobile = $derived(isMobile.current);

	function toggleExpandedMode() {
		isExpandedMode = !isExpandedMode;
		if (!isExpandedMode) {
			hoveredTooltip = null;
		}
	}

	$effect(() => {
		if (!isExpandedMode) {
			isSearchModeActive = false;
			searchQuery = '';
		}
	});

	// On mobile the dedicated /search route hides the sidebar (see the aside
	// render guard below). Collapse it as we enter /search so it doesn't
	// reappear expanded when the user navigates back via the back button.
	$effect(() => {
		if (isMobile.current && page.url.hash.includes(ROUTES.SEARCH)) {
			isExpandedMode = false;
		}
	});

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
		if (isMobile.current) {
			isExpandedMode = false;
		}
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

	let innerWidth = $state(0);
</script>

<svelte:window onkeydown={handleKeydown} bind:innerWidth />

{#if innerWidth > 768 || (!page.url.hash.includes(ROUTES.SETTINGS) && !page.url.hash.includes(ROUTES.MCP_SERVERS) && !page.url.hash.includes(ROUTES.SEARCH))}
	<aside
		class="fixed md:sticky top-2 left-2 md:left-0 md:ml-2 md:mt-2 {isExpandedMode &&
		device.isStandalone
			? 'h-[calc(100dvh-2rem)]'
			: isExpandedMode && device.isIOSDevice
				? 'h-[calc(100dvh-0.5rem)]'
				: isExpandedMode
					? 'h-[calc(100dvh-1rem)]'
					: ''} md:h-[calc(100dvh-1.125rem)] pt-2 rounded-3xl md:rounded-2xl z-10 flex flex-col justify-between md:transition-[width,padding] w-[calc(100dvw-1rem)] duration-200 ease-out {isStripExpanded
			? 'md:w-72 md:bg-muted/60 md:backdrop-blur-xl border-border shadow-md'
			: 'md:w-12'} {isExpandedMode ? 'is-expanded' : ''}"
	>
		<div class="px-2 flex items-center justify-between">
			<div
				role="button"
				tabindex="0"
				class="relative"
				onmouseenter={() => (logoHovered = true)}
				onmouseleave={() => (logoHovered = false)}
			>
				<ActionIcon
					icon={!isExpandedMode && logoHovered && innerWidth > 768 ? PanelLeftOpen : Logo}
					size="lg"
					iconSize="h-4.5 w-4.5 md:h-4 md:w-4"
					class="{isExpandedMode
						? 'md:bg-muted!'
						: 'bg-transparent!'} md:h-9 md:w-9 h-10 w-10 rounded-full md:hover:bg-accent!"
					href={isExpandedMode ? ROUTES.START : undefined}
					onclick={isExpandedMode ? undefined : toggleExpandedMode}
					tooltip={isExpandedMode ? undefined : 'Open Sidebar'}
					tooltipSide={TooltipSide.RIGHT}
					ariaLabel={isExpandedMode ? 'Go to start' : 'Expand navigation'}
				/>
			</div>

			{#if isExpandedMode || isOnMobile}
				<div
					class={!isExpandedMode && isOnMobile ? 'hidden' : ''}
					in:fade={{ duration: 150, easing: circIn, delay: 50 }}
					out:fade={{ duration: 100 }}
				>
					<ActionIcon
						icon={isMobile.current ? X : PanelLeftClose}
						size="lg"
						iconSize="h-4.5 w-4.5 md:h-4 md:w-4"
						class="backdrop-blur-none md:h-9 md:w-9 h-10 w-10 rounded-full mr-1 hover:bg-accent!"
						onclick={toggleExpandedMode}
						tooltip="Close Sidebar"
						tooltipSide={TooltipSide.LEFT}
						ariaLabel="Collapse navigation"
					/>
				</div>
			{/if}
		</div>

		<div class="mt-2 flex min-h-0 flex-1 flex-col gap-1">
			<SidebarNavigationActions
				{isExpandedMode}
				class="px-2"
				bind:isSearchModeActive
				bind:searchQuery
				onSearchDeactivated={() => {
					isSearchModeActive = false;
					searchQuery = '';
				}}
				onSearchClick={() => {
					isExpandedMode = true;
					isSearchModeActive = true;
				}}
				onNewChat={() => {
					if (isMobile.current) {
						isExpandedMode = false;
					}
				}}
			/>

			{#if isExpandedMode || isOnMobile}
				<div
					class="flex min-h-0 flex-1 flex-col {!isExpandedMode && isOnMobile ? 'hidden' : ''}"
					in:fade={{ duration: 150, easing: circIn, delay: 50 }}
					out:fade={{ duration: 100 }}
				>
					<SidebarNavigationConversationList
						class="px-2"
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
		</div>
	</aside>
{/if}

<style>
	aside {
		@media (max-width: 768px) {
			--size: 1.125rem;
		}
	}

	@media (max-width: 768px) {
		aside.is-expanded::before {
			content: '';
			position: fixed;
			top: -0.5rem;
			bottom: -0.25rem;
			left: -0.5rem;
			right: -0.5rem;
			z-index: -1;
			background: var(--background);
			backdrop-filter: blur(1rem);
			pointer-events: none;
		}
	}
</style>
