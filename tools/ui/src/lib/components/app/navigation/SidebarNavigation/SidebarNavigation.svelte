<script lang="ts">
	import { PanelLeftClose, PanelLeftOpen, X } from '@lucide/svelte';
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import {
		ActionIcon,
		DialogConversationRename,
		DialogDiscoverModels,
		DialogMcpServers,
		DialogModelInformation,
		DialogSettingsChat,
		Logo,
		SidebarNavigationActions,
		SidebarNavigationConversationList
	} from '$lib/components/app';
	import {
		ROUTES,
		SIDEBAR_COLLAPSE_WIDTH,
		SIDEBAR_COLLAPSED_WIDTH,
		SIDEBAR_MAX_WIDTH,
		SIDEBAR_MIN_WIDTH,
		SIDEBAR_OVERSTRETCH_PX,
		SIDEBAR_SETTLE_MS
	} from '$lib/constants';
	import { TooltipSide } from '$lib/enums';
	import { useKeyboardShortcuts } from '$lib/hooks/use-keyboard-shortcuts.svelte';
	import { useMarqueeSelection } from '$lib/hooks/use-marquee-selection.svelte';
	import { RouterService } from '$lib/services/router.service';
	import { chatStore, conversationsStore, deviceStore, settingsStore, uiStore } from '$lib/stores';
	import { buildConversationTree, remToPx } from '$lib/utils';
	import { circIn } from 'svelte/easing';
	import { SvelteSet } from 'svelte/reactivity';
	import { fade } from 'svelte/transition';

	interface Props {
		onSearchClick?: () => void;
	}

	let { onSearchClick = () => {} }: Props = $props();

	const { handleKeydown } = useKeyboardShortcuts({
		activateSearchMode: () => onSearchClick(),
		toggleSidebar: () => toggleExpandedMode()
	});

	let hoveredTooltip = $state<string | null>(null);
	let logoHovered = $state(false);

	const isStripExpanded = $derived(uiStore.isSidebarExpanded || hoveredTooltip !== null);
	const isOnMobile = $derived(deviceStore.isMobile);
	const alwaysShowOnDesktop = $derived(settingsStore.config.alwaysShowSidebarOnDesktop as boolean);

	$effect(() => {
		if (alwaysShowOnDesktop && !isOnMobile) {
			uiStore.isSidebarExpanded = true;
		}
	});

	function toggleExpandedMode() {
		uiStore.isSidebarExpanded = !uiStore.isSidebarExpanded;

		if (!uiStore.isSidebarExpanded) {
			hoveredTooltip = null;
		}
	}

	$effect(() => {
		if (!uiStore.isSidebarExpanded) {
			isSearchModeActive = false;
			searchQuery = '';

			if (isSelectionMode) exitSelectionMode();

			cancelMobileCollapse();
		}
	});

	$effect(() => {
		if (deviceStore.isMobile && page.url.hash.includes(ROUTES.SEARCH)) {
			uiStore.isSidebarExpanded = false;
		}
	});

	let currentChatId = $derived(page.params.id);
	let isSearchModeActive = $state(false);
	let searchQuery = $state('');

	let filteredConversations = $derived.by(() => {
		if (isSearchModeActive) {
			if (searchQuery.trim().length > 0) {
				return conversationsStore.conversations.filter((conversation: { name: string }) =>
					conversation.name.toLowerCase().includes(searchQuery.toLowerCase())
				);
			}

			return [];
		}

		return conversationsStore.conversations;
	});

	let isSelectionMode = $state(false);
	let selectedIds = new SvelteSet<string>();

	let renameDialogOpen = $state(false);
	let settingsDialogOpen = $state(false);
	let mcpDialogOpen = $state(false);
	let renameTargetConversationId = $state<string | null>(null);
	let renameDraft = $state('');
	let renameOriginalTitle = $state('');

	const renderedOrderIds = $derived(
		buildConversationTree(filteredConversations).map((t) => t.conversation.id)
	);

	const allSelectedArePinned = $derived.by(() => {
		if (selectedIds.size === 0) return false;

		const convs = conversationsStore.conversations;

		for (const id of selectedIds) {
			const c = convs.find((conv) => conv.id === id);

			if (c && !c.pinned) return false;
		}

		return true;
	});

	const pinStateIsMixed = $derived.by(() => {
		if (selectedIds.size === 0) return false;

		const convs = conversationsStore.conversations;

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
			selectedVisibleCount: selectedVisible,
			visibleCount: visibleIds.length
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
		enabled: () => isSelectionMode,
		orderedIds: () => renderedOrderIds,
		selectedIds: () => selectedIds
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
		if (deviceStore.isMobile) {
			scheduleMobileCollapse();
		}

		await goto(RouterService.chat(id));
	}

	async function handleEditConversation(id: string) {
		const conversation = conversationsStore.conversations.find((conv) => conv.id === id);

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
		const conversation = conversationsStore.conversations.find((conv) => conv.id === id);

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
			uiStore.isSidebarExpanded = false;
			pendingCollapse = null;
		}, 100);
	}

	function cancelMobileCollapse() {
		if (pendingCollapse) {
			clearTimeout(pendingCollapse);
			pendingCollapse = null;
		}
	}

	let asideEl = $state<HTMLElement | null>(null);
	/** Set briefly after a drag ends past a bound, while the width springs back to it. */
	let isSettling = $state(false);

	// the tab bar sits outside the sidebar, so the width lives on the root
	$effect(() => {
		const width = uiStore.sidebarWidth;

		if (width === null) {
			document.documentElement.style.removeProperty('--sidebar-width');
		} else {
			document.documentElement.style.setProperty('--sidebar-width', `${width}px`);
		}
	});

	// labels and the conversation list fade with the live width, so they follow
	// both the toggle's width transition and a drag below the minimum width
	$effect(() => {
		const el = asideEl;

		if (!el) return;

		const collapsedPx = remToPx(SIDEBAR_COLLAPSED_WIDTH);
		const minPx = remToPx(SIDEBAR_MIN_WIDTH);
		const observer = new ResizeObserver(([entry]) => {
			const fade = (entry.contentRect.width - collapsedPx) / (minPx - collapsedPx);

			el.style.setProperty('--sidebar-fade', String(Math.min(1, Math.max(0, fade))));
		});

		observer.observe(el);

		return () => observer.disconnect();
	});

	function handleResizePointerDown(event: PointerEvent) {
		if (!asideEl || isOnMobile) return;

		event.preventDefault();

		const handle = event.currentTarget as HTMLElement;
		const left = asideEl.getBoundingClientRect().left;
		const minPx = remToPx(SIDEBAR_MIN_WIDTH);
		const maxPx = remToPx(SIDEBAR_MAX_WIDTH);
		const collapsePx = remToPx(SIDEBAR_COLLAPSE_WIDTH);
		const startWidth = uiStore.sidebarWidth;

		let willCollapse = false;

		handle.setPointerCapture(event.pointerId);
		uiStore.isSidebarResizing = true;

		/** Rubber-band offset: grows with the distance, ever slower, never past the overstretch. */
		function stretch(distance: number): number {
			return SIDEBAR_OVERSTRETCH_PX * (1 - 1 / (distance / SIDEBAR_OVERSTRETCH_PX + 1));
		}

		function onPointerMove(moveEvent: PointerEvent) {
			const pointerWidth = moveEvent.clientX - left;

			// past either bound the edge resists harder the further it is pulled,
			// until a pull below the collapse width snaps it to the pointer
			willCollapse = pointerWidth < collapsePx;

			const width = willCollapse
				? Math.max(0, pointerWidth)
				: pointerWidth < minPx
					? minPx - stretch(minPx - pointerWidth)
					: pointerWidth > maxPx
						? maxPx + stretch(pointerWidth - maxPx)
						: pointerWidth;

			uiStore.setSidebarWidth(Math.round(width));
		}

		function onPointerUp() {
			handle.removeEventListener('pointermove', onPointerMove);
			handle.removeEventListener('pointerup', onPointerUp);
			handle.removeEventListener('pointercancel', onPointerCancel);

			uiStore.isSidebarResizing = false;

			// it reopens at its width from before the drag, without springing on the way out
			if (willCollapse) {
				uiStore.setSidebarWidth(startWidth);
				toggleExpandedMode();

				return;
			}

			const width = uiStore.sidebarWidth ?? minPx;
			const clamped = Math.min(maxPx, Math.max(minPx, width));

			if (clamped !== width) {
				isSettling = true;
				uiStore.setSidebarWidth(Math.round(clamped));
				setTimeout(() => (isSettling = false), SIDEBAR_SETTLE_MS);
			}
		}

		function onPointerCancel() {
			willCollapse = false;
			onPointerUp();
		}

		handle.addEventListener('pointermove', onPointerMove);
		handle.addEventListener('pointerup', onPointerUp);
		handle.addEventListener('pointercancel', onPointerCancel);
	}

	/** A dialog owns the screen on a phone, so the expanded list steps aside for it. */
	function openDialog(open: () => void): void {
		if (deviceStore.isMobile) {
			scheduleMobileCollapse();
		}

		open();
	}
</script>

<svelte:window bind:innerWidth onkeydown={handleKeydown} />

{#if innerWidth > 768 || !page.url.hash.includes(ROUTES.SEARCH)}
	<aside
		bind:this={asideEl}
		class={[
			'fixed md:sticky top-2 left-2 md:left-0 md:ml-2 md:mt-2 pt-2 z-10 w-[calc(100dvw-1rem)]',
			'md:h-[calc(100dvh-1.125rem)]',
			uiStore.isSidebarExpanded &&
				(deviceStore.isStandalone
					? 'h-[calc(100dvh-2rem)]'
					: deviceStore.isIOSDevice
						? 'h-[calc(100dvh-0.5rem)]'
						: 'h-[calc(100dvh-1rem)]'),
			'rounded-3xl md:rounded-2xl',
			'flex flex-col justify-between',
			'md:transition-[width,padding] duration-200 ease-out',
			uiStore.isSidebarResizing && 'md:duration-0',
			isSettling && 'md:duration-300 md:ease-[cubic-bezier(0.34,1.56,0.64,1)]',
			isStripExpanded && 'md:w-(--sidebar-width) md:bg-muted/60 md:backdrop-blur-xl shadow-md',
			!isStripExpanded && 'md:w-12',
			uiStore.isSidebarExpanded && 'is-expanded'
		]}
	>
		<div class="px-2 flex items-center justify-between">
			<div
				class="relative"
				onmouseenter={() => (logoHovered = true)}
				onmouseleave={() => (logoHovered = false)}
				role="button"
				tabindex="0"
			>
				<ActionIcon
					ariaLabel={uiStore.isSidebarExpanded ? 'Go to start' : 'Expand navigation'}
					class="{uiStore.isSidebarExpanded
						? 'bg-muted! md:bg-foreground/5!'
						: 'bg-transparent!'} md:h-9 md:w-9 h-10 w-10 rounded-full md:hover:bg-foreground/10! pointer-events-auto"
					href={uiStore.isSidebarExpanded ? ROUTES.START : undefined}
					icon={!uiStore.isSidebarExpanded && logoHovered && innerWidth > 768
						? PanelLeftOpen
						: Logo}
					iconSize="h-4.5 w-4.5 md:h-4 md:w-4"
					onclick={uiStore.isSidebarExpanded ? undefined : toggleExpandedMode}
					size="lg"
					tooltip={uiStore.isSidebarExpanded ? undefined : 'Open Sidebar'}
					tooltipSide={TooltipSide.RIGHT}
				/>
			</div>

			{#if isOnMobile || (uiStore.isSidebarExpanded && !alwaysShowOnDesktop)}
				<div
					in:fade={{ delay: 50, duration: 150, easing: circIn }}
					out:fade={{ duration: 100 }}
					class="flex items-center transition-all duration-150 ease-out {deviceStore.isMobile &&
					!uiStore.isSidebarExpanded
						? 'opacity-0 h-0!'
						: ''}"
				>
					<ActionIcon
						ariaLabel="Collapse navigation"
						class="backdrop-blur-none md:h-9 md:w-9 h-10 w-10 rounded-full mr-1 max-md:-mr-1 max-md:-mt-1 hover:bg-accent!"
						icon={deviceStore.isMobile ? X : PanelLeftClose}
						iconSize="h-4 w-4"
						onclick={toggleExpandedMode}
						size="lg"
						tooltip="Close Sidebar"
						tooltipSide={TooltipSide.LEFT}
					/>
				</div>
			{/if}
		</div>

		<div
			in:fade={{ duration: 200 }}
			out:fade={{ duration: 200 }}
			class="mt-2 flex min-h-0 flex-1 flex-col gap-4 md:gap-1 {deviceStore.isMobile
				? 'transition-[opacity,height] duration-200 ease-out'
				: ''} {deviceStore.isMobile && !uiStore.isSidebarExpanded ? 'opacity-0 !h-0' : ''}"
		>
			<SidebarNavigationActions
				bind:isSearchModeActive
				bind:searchQuery
				class="px-2"
				isExpandedMode={innerWidth > 768 ? uiStore.isSidebarExpanded : true}
				onManageModelsClick={() => openDialog(() => uiStore.openModelsManager())}
				onMcpClick={() => openDialog(() => (mcpDialogOpen = true))}
				onNewChat={() => {
					if (deviceStore.isMobile) {
						scheduleMobileCollapse();
					}
				}}
				onSearchClick={() => {
					uiStore.isSidebarExpanded = true;
					isSearchModeActive = true;
				}}
				onSearchDeactivated={() => {
					isSearchModeActive = false;
					searchQuery = '';
				}}
				onSettingsClick={() => openDialog(() => (settingsDialogOpen = true))}
			/>

			{#if uiStore.isSidebarExpanded || isOnMobile}
				<div class="flex min-h-0 flex-1 flex-col overflow-y-auto opacity-(--sidebar-fade)">
					<SidebarNavigationConversationList
						{allSelectedArePinned}
						allVisibleSelected={visibleSelectionStats.visibleCount > 0 &&
							visibleSelectionStats.selectedVisibleCount === visibleSelectionStats.visibleCount}
						class="px-2"
						{currentChatId}
						{filteredConversations}
						{isSearchModeActive}
						{isSelectionMode}
						onBulkDelete={handleBulkDelete}
						onBulkExport={handleBulkExport}
						onBulkPinToggle={handleBulkPinToggle}
						onCloseSelection={exitSelectionMode}
						onDelete={handleDeleteConversation}
						onEdit={handleEditConversation}
						onEnterSelectionMode={enterSelectionMode}
						onRowMouseDown={handleRowMouseDown}
						onSelect={selectConversation}
						onSelectAllToggle={toggleSelectAllVisible}
						onSelectionClick={handleSelectionClick}
						onStop={handleStopGeneration}
						onToggleSelect={toggleSelected}
						{pinStateIsMixed}
						{searchQuery}
						{selectedIds}
						someVisibleSelected={visibleSelectionStats.selectedVisibleCount > 0 &&
							visibleSelectionStats.selectedVisibleCount < visibleSelectionStats.visibleCount}
						visibleCount={visibleSelectionStats.visibleCount}
					/>
				</div>
			{/if}
		</div>

		{#if uiStore.isSidebarExpanded && !isOnMobile}
			<!-- svelte-ignore a11y_no_static_element_interactions -->
			<div
				class="peer absolute inset-y-3 -right-1 z-20 hidden w-2 cursor-ew-resize touch-none select-none md:block"
				data-slot="sidebar-resize-handle"
				ondblclick={() => uiStore.setSidebarWidth(null)}
				onpointerdown={handleResizePointerDown}
				title="Drag to resize or hide, double-click to reset"
			></div>

			<!-- the edge strip is clipped to the sidebar's rounded corners without
			     clipping the sidebar itself; faint on hover, solid while dragging -->
			<div
				class={[
					'pointer-events-none absolute inset-0 hidden overflow-hidden rounded-[inherit] transition-opacity duration-150 md:block',
					uiStore.isSidebarResizing ? 'opacity-100' : 'opacity-0 peer-hover:opacity-100'
				]}
			>
				<div
					class={[
						'absolute inset-y-0 right-0 w-1.5',
						uiStore.isSidebarResizing ? 'bg-foreground/15' : 'bg-foreground/6'
					]}
				></div>
			</div>
		{/if}
	</aside>
{/if}

<DialogConversationRename
	bind:open={renameDialogOpen}
	bind:value={renameDraft}
	currentTitle={renameOriginalTitle}
	onCancel={handleRenameCancel}
	onConfirm={handleRenameConfirm}
/>

<DialogSettingsChat bind:open={settingsDialogOpen} />

<DialogMcpServers bind:open={mcpDialogOpen} />

<DialogDiscoverModels bind:open={uiStore.discoverModelsOpen} />

<DialogModelInformation />

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
