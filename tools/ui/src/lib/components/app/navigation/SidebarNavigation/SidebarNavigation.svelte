<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { PanelLeftClose, PanelLeftOpen, X } from '@lucide/svelte';
	import {
		ActionIcon,
		DialogConversationRename,
		Logo,
		SidebarNavigationConversationList,
		SidebarNavigationActions
	} from '$lib/components/app';
	import { ROUTES } from '$lib/constants';
	import { fade } from 'svelte/transition';
	import { SvelteSet } from 'svelte/reactivity';
	import { useMarqueeSelection } from '$lib/hooks/use-marquee-selection.svelte';

	import { useKeyboardShortcuts } from '$lib/hooks/use-keyboard-shortcuts.svelte';
	import {
		buildConversationTree,
		conversationsStore,
		conversations
	} from '$lib/stores/conversations.svelte';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { config } from '$lib/stores/settings.svelte';
	import { RouterService } from '$lib/services/router.service';
	import { isMobile } from '$lib/stores/viewport.svelte';
	import { TooltipSide } from '$lib/enums';
	import { device } from '$lib/stores/device.svelte';
	import { circIn } from 'svelte/easing';

	interface Props {
		onSearchClick?: () => void;
	}

	let { onSearchClick = () => {} }: Props = $props();

	const { handleKeydown } = useKeyboardShortcuts({
		activateSearchMode: () => onSearchClick(),
		toggleSidebar: () => toggleExpandedMode()
	});

	let isExpandedMode = $state(false);
	let hoveredTooltip = $state<string | null>(null);
	let logoHovered = $state(false);

	const isStripExpanded = $derived(isExpandedMode || hoveredTooltip !== null);
	const isOnMobile = $derived(isMobile.current);
	const alwaysShowOnDesktop = $derived(config().alwaysShowSidebarOnDesktop as boolean);

	$effect(() => {
		if (alwaysShowOnDesktop && !isOnMobile) {
			isExpandedMode = true;
		}
	});

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
			if (isSelectionMode) exitSelectionMode();
			cancelMobileCollapse();
		}
	});

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

	let isSelectionMode = $state(false);
	let selectedIds = new SvelteSet<string>();

	let renameDialogOpen = $state(false);
	let renameTargetConversationId = $state<string | null>(null);
	let renameDraft = $state('');
	let renameOriginalTitle = $state('');

	const renderedOrderIds = $derived(
		buildConversationTree(filteredConversations).map((t) => t.conversation.id)
	);

	const allSelectedArePinned = $derived.by(() => {
		if (selectedIds.size === 0) return false;
		const convs = conversations();
		for (const id of selectedIds) {
			const c = convs.find((conv) => conv.id === id);
			if (c && !c.pinned) return false;
		}
		return true;
	});

	const pinStateIsMixed = $derived.by(() => {
		if (selectedIds.size === 0) return false;
		const convs = conversations();
		let anyPinned = false;
		let anyUnpinned = false;
		for (const id of selectedIds) {
			const c = convs.find((conv) => conv.id === id);
			if (!c) continue;
			if (c.pinned) anyPinned = true;
			else anyUnpinned = true;
			if (anyPinned && anyUnpinned) return true;
		}
		return false;
	});

	const visibleSelectionStats = $derived.by(() => {
		const visibleIds = filteredConversations.map((c) => c.id);
		let selectedVisible = 0;
		for (const id of visibleIds) {
			if (selectedIds.has(id)) selectedVisible++;
		}
		return {
			visibleCount: visibleIds.length,
			selectedVisibleCount: selectedVisible
		};
	});

	function enterSelectionMode(id?: string) {
		isSelectionMode = true;
		if (id !== undefined) {
			selectedIds.add(id);
		}
	}

	function exitSelectionMode() {
		isSelectionMode = false;
		selectedIds.clear();
	}

	function toggleSelected(id: string) {
		if (selectedIds.has(id)) {
			selectedIds.delete(id);
		} else {
			selectedIds.add(id);
		}
	}

	function toggleSelectAllVisible() {
		const visibleIds = filteredConversations.map((c) => c.id);
		const allSelected = visibleIds.length > 0 && visibleIds.every((id) => selectedIds.has(id));

		if (allSelected) {
			for (const id of visibleIds) selectedIds.delete(id);
		} else {
			for (const id of visibleIds) selectedIds.add(id);
		}
	}

	async function handleBulkDelete() {
		const ids = Array.from(selectedIds);
		if (ids.length === 0) return;
		await conversationsStore.bulkDeleteConversations(ids);
		exitSelectionMode();
	}

	async function handleBulkPinToggle() {
		const ids = Array.from(selectedIds);
		if (ids.length === 0) return;
		await conversationsStore.bulkToggleConversationPin(ids);
	}

	async function handleBulkExport() {
		const ids = Array.from(selectedIds);
		if (ids.length === 0) return;
		await conversationsStore.bulkExportConversations(ids);
	}

	const marquee = useMarqueeSelection({
		selectedIds: () => selectedIds,
		orderedIds: () => renderedOrderIds,
		enabled: () => isSelectionMode
	});

	function handleRowMouseDown(id: string, event: MouseEvent) {
		if (!isSelectionMode) return;
		marquee.rowMouseDown(id, event);
	}

	function handleSelectionClick(id: string, options: { shiftKey: boolean }): void {
		if (!isSelectionMode) return;
		marquee.rowClick(id, options.shiftKey);
	}

	async function selectConversation(id: string) {
		if (isMobile.current) {
			scheduleMobileCollapse();
		}
		await goto(RouterService.chat(id));
	}

	async function handleEditConversation(id: string) {
		const conversation = conversations().find((conv) => conv.id === id);
		if (!conversation) return;

		renameTargetConversationId = id;
		renameOriginalTitle = conversation.name;
		renameDraft = conversation.name;
		renameDialogOpen = true;
	}

	async function handleRenameConfirm() {
		const id = renameTargetConversationId;
		if (!id) return;

		const nextName = renameDraft.trim();
		if (!nextName || nextName === renameOriginalTitle.trim()) return;

		await conversationsStore.updateConversationName(id, nextName);

		renameDialogOpen = false;
		renameTargetConversationId = null;
	}

	function handleRenameCancel() {
		renameDialogOpen = false;
		renameTargetConversationId = null;
		renameDraft = '';
		renameOriginalTitle = '';
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
	let pendingCollapse = $state<ReturnType<typeof setTimeout> | null>(null);

	function scheduleMobileCollapse() {
		if (pendingCollapse) {
			clearTimeout(pendingCollapse);
		}
		pendingCollapse = setTimeout(() => {
			isExpandedMode = false;
			pendingCollapse = null;
		}, 100);
	}

	function cancelMobileCollapse() {
		if (pendingCollapse) {
			clearTimeout(pendingCollapse);
			pendingCollapse = null;
		}
	}
</script>

<svelte:window onkeydown={handleKeydown} bind:innerWidth />

{#if innerWidth > 768 || (!page.url.hash.includes(ROUTES.SETTINGS) && !page.url.hash.includes(ROUTES.MCP_SERVERS) && !page.url.hash.includes(ROUTES.SEARCH))}
	<aside
		class={[
			'fixed md:sticky top-2 left-2 md:left-0 md:ml-2 md:mt-2 pt-2 z-10 w-[calc(100dvw-1rem)]',
			'md:h-[calc(100dvh-1.125rem)]',
			isExpandedMode &&
				(device.isStandalone
					? 'h-[calc(100dvh-2rem)]'
					: device.isIOSDevice
						? 'h-[calc(100dvh-0.5rem)]'
						: 'h-[calc(100dvh-1rem)]'),
			'rounded-3xl md:rounded-2xl',
			'flex flex-col justify-between',
			'md:transition-[width,padding] duration-200 ease-out',
			isStripExpanded && 'md:w-72 md:bg-muted/60 md:backdrop-blur-xl border-border shadow-md',
			!isStripExpanded && 'md:w-12',
			isExpandedMode && 'is-expanded'
		]}
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
						? 'bg-muted! md:bg-foreground/5!'
						: 'bg-transparent!'} md:h-9 md:w-9 h-10 w-10 rounded-full md:hover:bg-foreground/10! pointer-events-auto"
					href={isExpandedMode ? ROUTES.START : undefined}
					onclick={isExpandedMode ? undefined : toggleExpandedMode}
					tooltip={isExpandedMode ? undefined : 'Open Sidebar'}
					tooltipSide={TooltipSide.RIGHT}
					ariaLabel={isExpandedMode ? 'Go to start' : 'Expand navigation'}
				/>
			</div>

			{#if isOnMobile || (isExpandedMode && !alwaysShowOnDesktop)}
				<div
					class="flex items-center transition-all duration-150 ease-out {isMobile.current &&
					!isExpandedMode
						? 'opacity-0 h-0!'
						: ''}"
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

		<div
			class="mt-2 flex min-h-0 flex-1 flex-col gap-4 md:gap-1 {isMobile.current
				? 'transition-[opacity,height] duration-200 ease-out'
				: ''} {isMobile.current && !isExpandedMode ? 'opacity-0 !h-0' : ''}"
			in:fade={{ duration: 200 }}
			out:fade={{ duration: 200 }}
		>
			<SidebarNavigationActions
				isExpandedMode={innerWidth > 768 ? isExpandedMode : true}
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
						scheduleMobileCollapse();
					}
				}}
			/>

			{#if isExpandedMode || isOnMobile}
				<div class="flex min-h-0 flex-1 flex-col overflow-y-auto">
					<SidebarNavigationConversationList
						class="px-2"
						{filteredConversations}
						{currentChatId}
						{isSearchModeActive}
						{searchQuery}
						{isSelectionMode}
						{selectedIds}
						onSelect={selectConversation}
						onEdit={handleEditConversation}
						onDelete={handleDeleteConversation}
						onStop={handleStopGeneration}
						onToggleSelect={toggleSelected}
						onEnterSelectionMode={enterSelectionMode}
						onSelectionClick={handleSelectionClick}
						onRowMouseDown={handleRowMouseDown}
						visibleCount={visibleSelectionStats.visibleCount}
						allVisibleSelected={visibleSelectionStats.visibleCount > 0 &&
							visibleSelectionStats.selectedVisibleCount === visibleSelectionStats.visibleCount}
						someVisibleSelected={visibleSelectionStats.selectedVisibleCount > 0 &&
							visibleSelectionStats.selectedVisibleCount < visibleSelectionStats.visibleCount}
						{allSelectedArePinned}
						{pinStateIsMixed}
						onSelectAllToggle={toggleSelectAllVisible}
						onBulkPinToggle={handleBulkPinToggle}
						onBulkExport={handleBulkExport}
						onBulkDelete={handleBulkDelete}
						onCloseSelection={exitSelectionMode}
					/>
				</div>
			{/if}
		</div>
	</aside>
{/if}

<DialogConversationRename
	bind:open={renameDialogOpen}
	currentTitle={renameOriginalTitle}
	bind:value={renameDraft}
	onConfirm={handleRenameConfirm}
	onCancel={handleRenameCancel}
/>

<style>
	aside {
		@media (max-width: 768px) {
			--size: 1.125rem;
		}
	}

	@media (max-width: 768px) {
		aside {
			&:not(.is-expanded) {
				pointer-events: none;
			}
		}

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
