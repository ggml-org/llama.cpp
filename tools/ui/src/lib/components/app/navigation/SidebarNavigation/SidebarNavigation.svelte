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
	import { SvelteSet } from 'svelte/reactivity';

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

	// Keep the sidebar expanded on desktop when the user pins it open
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

	let isSelectionMode = $state(false);
	let selectedIds = new SvelteSet<string>();

	// Visual order used by both shift+click range and marquee hit-testing.
	// buildConversationTree puts pinned items first then unpinned, with forks
	// interleaved under their parents - matching the rendered list exactly.
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

	// Mixed state happens when some selected items are pinned and others are not.
	// The Pin/Unpin bulk action toggles per-id, so applying it to a mixed selection
	// would invert pinned state inconsistently - we surface that as a disabled
	// state with an explanatory tooltip instead.
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
		const allSelected =
			visibleIds.length > 0 && visibleIds.every((id) => selectedIds.has(id));

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

	let dragAnchorId = $state<string | null>(null);
	let isMarqueeDragging = $state(false);
	let mouseDownActive = $state(false);
	let dragStartX = 0;
	let dragStartY = 0;
	let mousedownRowId: string | null = null;
	let dragMode: 'add' | 'remove' | null = null;
	let suppressNextClick = false;

	const DRAG_THRESHOLD_PX = 5;

	function rangeSelect(fromId: string, toId: string) {
		if (fromId === toId) {
			selectedIds.add(toId);
			return;
		}
		const order = renderedOrderIds;
		const fromIdx = order.indexOf(fromId);
		const toIdx = order.indexOf(toId);
		if (fromIdx === -1 || toIdx === -1) return;
		const [lo, hi] = fromIdx < toIdx ? [fromIdx, toIdx] : [toIdx, fromIdx];
		// A range from `hi` to `lo` would overwrite in-between items we don't want.
		// Range-select mode always ADDS within [lo, hi] - it should not deselect items
		// outside the range that the user previously picked via other means.
		for (let i = lo; i <= hi; i++) {
			selectedIds.add(order[i]);
		}
	}

	function findRowAtPoint(x: number, y: number): string | null {
		let bestMatch: HTMLElement | null = null;
		let bestCenterDistance = Infinity;

		for (const row of document.querySelectorAll<HTMLElement>('[data-conversation-row]')) {
			const rect = row.getBoundingClientRect();
			if (y >= rect.top && y <= rect.bottom && x >= rect.left && x <= rect.right) {
				return row.dataset.conversationRow ?? null;
			}
			if (x >= rect.left && x <= rect.right) {
				const centerDistance = Math.abs(y - (rect.top + rect.height / 2));
				if (centerDistance < bestCenterDistance) {
					bestCenterDistance = centerDistance;
					bestMatch = row;
				}
			}
		}
		return bestMatch ? (bestMatch.dataset.conversationRow ?? null) : null;
	}

	function handleRowMouseDown(id: string, event: MouseEvent) {
		if (!isSelectionMode) return;
		if (event.button !== 0) return;
		event.preventDefault();
		mouseDownActive = true;
		mousedownRowId = id;
		dragStartX = event.clientX;
		dragStartY = event.clientY;
		isMarqueeDragging = false;
		dragMode = null;

		if (event.shiftKey && dragAnchorId !== null && dragAnchorId !== id) {
			// Initial range guess on mousedown so the user sees an instant range
			// even if they release without crossing the drag threshold.
			rangeSelect(dragAnchorId, id);
		}
	}

	function updateMarqueeRect(currentX: number, currentY: number) {
		const left = Math.min(dragStartX, currentX);
		const top = Math.min(dragStartY, currentY);
		const right = Math.max(dragStartX, currentX);
		const bottom = Math.max(dragStartY, currentY);

		const visibleIds = new Set(renderedOrderIds);
		const rows = document.querySelectorAll<HTMLElement>('[data-conversation-row]');

		for (const row of rows) {
			const id = row.dataset.conversationRow;
			if (!id || !visibleIds.has(id)) continue;

			const rect = row.getBoundingClientRect();
			const intersects =
				!(rect.right < left || rect.left > right || rect.bottom < top || rect.top > bottom);

			if (dragMode === 'add') {
				// Additive marquee: rows the rect covers become selected; rows outside
				// the rect are untouched (prior selection is preserved). This is the
				// "I already have a group, drag from an unselected row to add more" case.
				if (intersects) selectedIds.add(id);
			} else if (dragMode === 'remove') {
				// Reverse marquee: only previously-selected rows inside the rect get
				// unselected. Rows that were never selected are not flipped to
				// 'selected then deselected' and prior selection outside the rect is
				// preserved. Idempotent per frame so cursor sweep speed can't flip-flop.
				if (intersects && selectedIds.has(id)) selectedIds.delete(id);
			}
		}
	}

	function handleDocumentMouseMove(event: MouseEvent) {
		if (!isSelectionMode || !mouseDownActive) return;

		if (event.shiftKey && dragAnchorId !== null) {
			// Shift+drag: live range selection, anchored on `dragAnchorId`.
			const target = findRowAtPoint(event.clientX, event.clientY);
			if (target && target !== mousedownRowId) {
				rangeSelect(dragAnchorId, target);
			}
			return;
		}

		if (!isMarqueeDragging) {
			const dx = event.clientX - dragStartX;
			const dy = event.clientY - dragStartY;
			if (Math.hypot(dx, dy) < DRAG_THRESHOLD_PX) return;
			isMarqueeDragging = true;
			// Mode is fixed for the entire drag based on the row the user started
			// on: dragging from a selected row enters "remove" mode (unselect);
			// dragging from an unselected row enters "add" mode (additive). The
			// user feels in control because the cursor position does not toggle the
			// mode mid-drag.
			dragMode = mousedownRowId !== null && selectedIds.has(mousedownRowId) ? 'remove' : 'add';
		}
		updateMarqueeRect(event.clientX, event.clientY);
	}

	function handleDocumentMouseUp(event: MouseEvent) {
		if (!isSelectionMode) return;
		if (isMarqueeDragging) {
			suppressNextClick = true;
			// Anchor for the next shift+click matches the drag-end row.
			const target = findRowAtPoint(event.clientX, event.clientY);
			if (target) dragAnchorId = target;
		}
		isMarqueeDragging = false;
		mouseDownActive = false;
		mousedownRowId = null;
		dragMode = null;
		dragStartX = 0;
		dragStartY = 0;
	}

	function handleClickCapture(event: MouseEvent) {
		if (suppressNextClick) {
			event.stopPropagation();
			event.preventDefault();
			suppressNextClick = false;
		}
	}

	$effect(() => {
		if (!isSelectionMode) {
			dragAnchorId = null;
			isMarqueeDragging = false;
			mouseDownActive = false;
			suppressNextClick = false;
			mousedownRowId = null;
			dragMode = null;
			return;
		}
		document.addEventListener('mousemove', handleDocumentMouseMove);
		document.addEventListener('mouseup', handleDocumentMouseUp);
		document.addEventListener('click', handleClickCapture, { capture: true });
		return () => {
			document.removeEventListener('mousemove', handleDocumentMouseMove);
			document.removeEventListener('mouseup', handleDocumentMouseUp);
			document.removeEventListener('click', handleClickCapture, { capture: true });
		};
	});

	function handleSelectionClick(id: string, options: { shiftKey: boolean }): void {
		if (options.shiftKey) {
			const fromId = dragAnchorId;
			if (fromId !== null) {
				// Range from anchor to current target. Anchor moves to the *target* after
				// each shift+click, so a chained shift+click after a marquee drag keeps
				// extending from where the previous drag / shift+click ended.
				rangeSelect(fromId, id);
				dragAnchorId = id;
			} else {
				selectedIds.add(id);
				dragAnchorId = id;
			}
			return;
		}

		if (selectedIds.has(id)) {
			selectedIds.delete(id);
		} else {
			selectedIds.add(id);
		}
		dragAnchorId = id;
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
			// Layout & positioning
			'fixed md:sticky top-2 left-2 md:left-0 md:ml-2 md:mt-2 pt-2 z-10 w-[calc(100dvw-1rem)]',
			// Dimensions & overflow
			'md:h-[calc(100dvh-1.125rem)]',
			isExpandedMode &&
				(device.isStandalone
					? 'h-[calc(100dvh-2rem)]'
					: device.isIOSDevice
						? 'h-[calc(100dvh-0.5rem)]'
						: 'h-[calc(100dvh-1rem)]'),
			// Shape & depth
			'rounded-3xl md:rounded-2xl',
			// Flex layout
			'flex flex-col justify-between',
			// Transition
			'md:transition-[width,padding] duration-200 ease-out',
			// Expanded state: width, surface, depth
			isStripExpanded && 'md:w-72 md:bg-muted/60 md:backdrop-blur-xl border-border shadow-md',
			// Collapsed state
			!isStripExpanded && 'md:w-12',
			// Expanded mode flag (for mobile ::before overlay)
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
							visibleSelectionStats.selectedVisibleCount ===
								visibleSelectionStats.visibleCount}
						someVisibleSelected={visibleSelectionStats.selectedVisibleCount > 0 &&
							visibleSelectionStats.selectedVisibleCount <
								visibleSelectionStats.visibleCount}
						allSelectedArePinned={allSelectedArePinned}
						pinStateIsMixed={pinStateIsMixed}
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
