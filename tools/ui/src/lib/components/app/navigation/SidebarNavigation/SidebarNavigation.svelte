<script lang="ts">
	import { PanelLeftClose, PanelLeftOpen, Search } from '@lucide/svelte';
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import {
		ActionIcon,
		DialogConversationDelete,
		DialogConversationRename,
		DialogDiscoverModels,
		DialogGroup,
		DialogGroupPicker,
		DialogMcpServers,
		DialogSearch,
		DialogSettingsChat,
		Logo,
		ModelsManagerModelDrawer,
		SidebarNavigationActions,
		SidebarNavigationConversationList,
		SidebarNavigationFooter
	} from '$lib/components/app';
	import {
		ROUTES,
		SIDEBAR_ACTIONS_ITEMS,
		SIDEBAR_COLLAPSE_WIDTH,
		SIDEBAR_COLLAPSED_WIDTH,
		SIDEBAR_CONTENT_FADE_MS,
		SIDEBAR_FOOTER_ITEMS,
		SIDEBAR_MAX_WIDTH,
		SIDEBAR_MIN_WIDTH,
		SIDEBAR_OVERSTRETCH_PX,
		SIDEBAR_REVEAL_ACTION_DELAY_MS,
		SIDEBAR_SETTLE_MS,
		SIDEBAR_TOGGLE_MS
	} from '$lib/constants';
	import { SidebarAction, TooltipSide } from '$lib/enums';
	import { useKeyboardShortcuts } from '$lib/hooks/use-keyboard-shortcuts.svelte';
	import { useMarqueeSelection } from '$lib/hooks/use-marquee-selection.svelte';
	import { RouterService } from '$lib/services/router.service';
	import { chatStore, conversationsStore, deviceStore, settingsStore, uiStore } from '$lib/stores';
	import { buildConversationTree, remToPx } from '$lib/utils';
	import { tick } from 'svelte';
	import { SvelteSet } from 'svelte/reactivity';
	import { fade } from 'svelte/transition';

	const { handleKeydown } = useKeyboardShortcuts({
		openModelsManager: () => handleSidebarAction(SidebarAction.MANAGE_MODELS),
		openSearch: () => handleSidebarAction(SidebarAction.SEARCH),
		openSettings: () => handleSidebarAction(SidebarAction.SETTINGS),
		toggleSidebar: () => toggleExpandedMode()
	});

	let innerWidth = $state(0);
	let logoHovered = $state(false);
	/** Set while the collapsed sidebar is revealed by hovering its logo, until the pointer leaves it. */
	let isPeeking = $state(false);
	/** Set after the toggle closes the sidebar under the pointer, until the pointer leaves the logo. */
	let isPeekSuppressed = $state(false);
	/** Keeps the collapsed sidebar in its strip-width slot while it slides closed or before it slides open. */
	let isDocked = $state(false);
	/** Drops the transitions for the frame the sidebar swaps between docked and floating, out of sight. */
	let isSnapping = $state(false);
	let dockTimeout: ReturnType<typeof setTimeout> | undefined;

	const isOnMobile = $derived(deviceStore.isMobile);
	const isHiddenByChatForm = $derived(
		isOnMobile && uiStore.isComposerExpanded && !uiStore.isSidebarExpanded
	);
	const alwaysShowOnDesktop = $derived(settingsStore.config.alwaysShowSidebarOnDesktop as boolean);
	// the collapsed sidebar shows only its logo until it is peeked at, which fades the rest in
	const isStripHidden = $derived(!uiStore.isSidebarExpanded && !isOnMobile && !isPeeking);
	// once settled, the collapsed sidebar keeps its full width, floating over the page,
	// so a peek only fades it in instead of resizing it
	const isFloating = $derived(!uiStore.isSidebarExpanded && !isOnMobile && !isDocked);
	// on desktop the logo doubles as the sidebar toggle, unless the sidebar is pinned open
	const isLogoToggle = $derived(
		innerWidth > 768 && (!uiStore.isSidebarExpanded || !alwaysShowOnDesktop)
	);

	$effect(() => {
		if (alwaysShowOnDesktop && !isOnMobile) {
			uiStore.isSidebarExpanded = true;
		}
	});

	/** Swaps between docked and floating without a transition, on the next frames. */
	function snapDocked(docked: boolean, then?: () => void) {
		isSnapping = true;
		isDocked = docked;

		requestAnimationFrame(() =>
			requestAnimationFrame(() => {
				isSnapping = false;
				then?.();
			})
		);
	}

	function toggleExpandedMode() {
		clearTimeout(dockTimeout);

		if (isOnMobile) {
			uiStore.isSidebarExpanded = !uiStore.isSidebarExpanded;

			return;
		}

		if (uiStore.isSidebarExpanded) {
			// slide closed at strip width, kept hidden under the pointer, then float once out of sight
			isPeeking = false;
			isPeekSuppressed = true;
			isDocked = true;
			uiStore.isSidebarExpanded = false;
			dockTimeout = setTimeout(() => snapDocked(false), SIDEBAR_TOGGLE_MS);

			return;
		}

		if (isPeeking) {
			// already shown at full width, so only the page slides aside
			isDocked = false;
			uiStore.isSidebarExpanded = true;

			return;
		}

		// dock to strip width out of sight, then slide open from there
		snapDocked(true, () => {
			isDocked = false;
			uiStore.isSidebarExpanded = true;
		});
	}

	function peek() {
		if (!isPeekSuppressed) {
			isPeeking = true;
		}
	}

	function handleAsideFocusOut(event: FocusEvent) {
		if (!asideEl?.contains(event.relatedTarget as Node | null)) {
			isPeeking = false;
		}
	}

	$effect(() => {
		if (!uiStore.isSidebarExpanded) {
			if (isSelectionMode) exitSelectionMode();
		}
	});

	$effect(() => {
		if (deviceStore.isMobile && page.url.hash.includes(ROUTES.SEARCH)) {
			uiStore.isSidebarExpanded = false;
		}
	});

	let currentChatId = $derived(page.params.id);
	let searchOpen = $state(false);

	// a phone shows the search as a full-height sheet, once the sidebar has stepped aside
	function openSearch() {
		afterMobileClose(() => (searchOpen = true));
	}

	let isSelectionMode = $state(false);
	let selectedIds = new SvelteSet<string>();

	let renameDialogOpen = $state(false);
	let settingsDialogOpen = $state(false);
	let mcpDialogOpen = $state(false);
	let renameTargetConversationId = $state<string | null>(null);
	let renameDraft = $state('');
	let renameOriginalTitle = $state('');

	const renderedOrderIds = $derived(
		buildConversationTree(conversationsStore.conversations).map((t) => t.conversation.id)
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
		const visibleIds = conversationsStore.conversations.map((c) => c.id);

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
		const visibleIds = conversationsStore.conversations.map((c) => c.id);
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

	function selectConversation(id: string) {
		navigateBehind(() => goto(RouterService.chat(id)));
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

	let deleteDialogOpen = $state(false);
	let deleteTargetId = $state<string | null>(null);

	function handleDeleteConversation(id: string) {
		const conversation = conversationsStore.conversations.find((conv) => conv.id === id);

		if (!conversation) return;

		deleteTargetId = conversation.id;
		deleteDialogOpen = true;
	}

	async function handleDeleteConfirm() {
		const id = deleteTargetId;

		deleteDialogOpen = false;

		if (!id) return;

		await conversationsStore.deleteConversation(id, { deleteWithForks: false });
	}

	function handleDeleteCancel() {
		deleteDialogOpen = false;
	}

	function handleStopGeneration(id: string) {
		chatStore.stopGenerationForChat(id);
	}

	/**
	 * A phone opens the page a tap in its sidebar goes to while the current one is still aside,
	 * and slides the sidebar shut once that page has rendered, so it is ready, settled, as it
	 * slides back. Anything else navigates at once.
	 */
	function navigateBehind(go: () => Promise<unknown>): void {
		if (!deviceStore.isMobile || !uiStore.isSidebarExpanded) {
			void go();

			return;
		}

		void go()
			.then(tick)
			.finally(() => requestAnimationFrame(() => (uiStore.isSidebarExpanded = false)));
	}

	/**
	 * A phone slides its open sidebar shut first, and only then opens what a tap in it asks for,
	 * so a dialog or the search screen appears once the sidebar is out of the way. Anything else
	 * acts at once.
	 */
	function afterMobileClose(run: () => void): void {
		if (!deviceStore.isMobile || !uiStore.isSidebarExpanded) {
			run();

			return;
		}

		uiStore.isSidebarExpanded = false;
		setTimeout(run, SIDEBAR_REVEAL_ACTION_DELAY_MS);
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

			// persisted once on release, not on every move
			uiStore.sidebarWidth = Math.round(width);
		}

		function onPointerUp() {
			handle.removeEventListener('pointermove', onPointerMove);
			handle.removeEventListener('pointerup', onPointerUp);
			handle.removeEventListener('pointercancel', onPointerCancel);

			uiStore.isSidebarResizing = false;

			// it reopens at its width from before the drag, without springing on the way out
			if (willCollapse) {
				uiStore.sidebarWidth = startWidth;
				toggleExpandedMode();

				return;
			}

			if (uiStore.sidebarWidth === startWidth) return;

			const width = uiStore.sidebarWidth ?? minPx;
			const clamped = Math.min(maxPx, Math.max(minPx, width));

			if (clamped !== width) {
				isSettling = true;
				setTimeout(() => (isSettling = false), SIDEBAR_SETTLE_MS);
			}

			uiStore.setSidebarWidth(Math.round(clamped));
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
		afterMobileClose(open);
	}

	/** Runs a sidebar action, whether picked from the icon strip, the search dialog or a shortcut. */
	function handleSidebarAction(action: SidebarAction): void {
		switch (action) {
			case SidebarAction.NEW_CHAT:
				navigateBehind(() => conversationsStore.openNewChat({ inNewTab: true }));

				break;
			case SidebarAction.SEARCH:
				openSearch();

				break;
			case SidebarAction.MANAGE_MODELS:
				openDialog(() => uiStore.openModelsManager());

				break;
			// dialogs on a desktop or tablet; on a phone, pages of their own
			case SidebarAction.MCP:
				if (isOnMobile) navigateBehind(() => goto(ROUTES.MCP_SERVERS));
				else openDialog(() => (mcpDialogOpen = true));

				break;
			case SidebarAction.SETTINGS:
				if (isOnMobile) navigateBehind(() => goto(ROUTES.SETTINGS));
				else openDialog(() => (settingsDialogOpen = true));

				break;
		}
	}
</script>

<svelte:window bind:innerWidth onkeydown={handleKeydown} />

{#if innerWidth > 768 || !page.url.hash.includes(ROUTES.SEARCH)}
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<aside
		bind:this={asideEl}
		class={[
			'fixed md:sticky top-0 left-0 md:top-2 md:left-0 md:ml-2 md:mt-2 pt-2 z-0 md:z-20 w-(--sidebar-reveal)',
			'h-dvh md:h-[calc(100dvh-1.125rem)]',
			// phone: it stays put under the page, which slides aside to uncover it
			isOnMobile && 'bg-(--sidebar-reveal-bg)',
			'md:rounded-2xl',
			'flex flex-col justify-between',
			// desktop only: on a phone the background follows the reveal frame by frame
			'md:transition-[width,padding,margin,background-color,box-shadow,backdrop-filter] md:duration-200 md:ease-out',
			(uiStore.isSidebarResizing || isSnapping) && 'md:duration-0',
			isSettling && 'md:duration-300 md:ease-[cubic-bezier(0.34,1.56,0.64,1)]',
			isDocked && !uiStore.isSidebarExpanded ? 'md:w-12' : 'md:w-(--sidebar-width)',
			(uiStore.isSidebarExpanded || (isFloating && isPeeking)) &&
				'md:bg-muted/60 md:backdrop-blur-xl max-md:shadow-md',
			isFloating && 'md:-mr-[calc(var(--sidebar-width)-3rem)]',
			isStripHidden && 'md:pointer-events-none',
			// on a phone the expanded chat form fills the screen, so the logo button over
			// its corner fades away (and goes inert) until the form collapses
			'max-md:transition-opacity',
			isHiddenByChatForm && 'opacity-0',
			uiStore.isSidebarExpanded && 'is-expanded'
		]}
		inert={isHiddenByChatForm || (isOnMobile && !uiStore.isSidebarExpanded)}
		onfocusin={peek}
		onfocusout={handleAsideFocusOut}
		onmouseleave={() => (isPeeking = false)}
	>
		<!-- phone: inset so the logo lands where the page's menu toggle sits -->
		<div
			class={[
				'flex min-h-0 flex-1 flex-col',
				isOnMobile && 'sidebar-reveal-content px-2 pt-[calc(var(--corner-control-top)-0.5rem)]'
			]}
		>
			<div class="px-2 flex items-center justify-between">
				<div
					class="relative"
					onmouseenter={() => {
						logoHovered = true;
						peek();
					}}
					onmouseleave={() => {
						logoHovered = false;
						isPeekSuppressed = false;
					}}
					role="button"
					tabindex="0"
				>
					<!-- the chat form's add-button key whenever the sidebar shows, open or peeking;
					     no fill while it is hidden; hover and focus never change the fill -->
					<ActionIcon
						ariaLabel={!uiStore.isSidebarExpanded
							? 'Expand navigation'
							: isLogoToggle
								? 'Collapse navigation'
								: 'Go to start'}
						class="pointer-events-auto size-10 rounded-full! md:size-9 {isStripHidden
							? 'bg-transparent! hover:bg-transparent!'
							: 'bg-muted/60 border border-border/20 backdrop-blur-sm bg-background! hover:bg-background! dark:bg-muted-foreground/15! dark:hover:bg-muted-foreground/15!'}"
						href={uiStore.isSidebarExpanded && !isLogoToggle ? ROUTES.START : undefined}
						icon={isLogoToggle && logoHovered
							? uiStore.isSidebarExpanded
								? PanelLeftClose
								: PanelLeftOpen
							: Logo}
						iconSize="size-4.5 md:size-4"
						onclick={!uiStore.isSidebarExpanded || isLogoToggle ? toggleExpandedMode : undefined}
						size="lg"
						tooltip={!uiStore.isSidebarExpanded
							? 'Open Sidebar'
							: isLogoToggle
								? 'Close Sidebar'
								: undefined}
						tooltipSide={TooltipSide.RIGHT}
					/>
				</div>

				<!-- a phone reaches search from a round key across from the logo, in place of the
				     Search row among the sidebar's actions -->
				{#if isOnMobile}
					<ActionIcon
						ariaLabel="Search"
						class="bg-muted/60 border border-border/20 backdrop-blur-sm size-10 rounded-full! bg-background! hover:bg-background! dark:bg-muted-foreground/15! dark:hover:bg-muted-foreground/15!"
						icon={Search}
						iconSize="size-4.5"
						onclick={() => handleSidebarAction(SidebarAction.SEARCH)}
						size="lg"
					/>
				{/if}
			</div>

			<!-- the hidden strip's content goes inert, not only invisible: descendants can opt
			     back into visibility (an open CollapsibleRegion) and pointer events (the inline
			     style of a context-menu trigger), and would catch clicks meant for the page -->
			<div
				in:fade={{ duration: SIDEBAR_CONTENT_FADE_MS }}
				out:fade={{ duration: SIDEBAR_CONTENT_FADE_MS }}
				class="mt-2 flex min-h-0 flex-1 flex-col gap-7 max-md:mt-6 md:gap-1 md:transition-opacity md:duration-150 md:ease-out {isStripHidden
					? 'md:opacity-0 md:invisible'
					: ''}"
				inert={isStripHidden}
			>
				<!-- a phone has no footer: its items join these -->
				<SidebarNavigationActions
					class="px-2 md:my-3"
					items={isOnMobile ? [...SIDEBAR_ACTIONS_ITEMS, ...SIDEBAR_FOOTER_ITEMS] : undefined}
					onAction={handleSidebarAction}
					onNavigate={navigateBehind}
				/>

				<div
					class="flex min-h-0 flex-1 flex-col overflow-y-auto opacity-(--sidebar-fade) max-md:pb-safe-2"
				>
					<SidebarNavigationConversationList
						{allSelectedArePinned}
						allVisibleSelected={visibleSelectionStats.visibleCount > 0 &&
							visibleSelectionStats.selectedVisibleCount === visibleSelectionStats.visibleCount}
						class="px-2"
						conversations={conversationsStore.conversations}
						{currentChatId}
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
						{selectedIds}
						someVisibleSelected={visibleSelectionStats.selectedVisibleCount > 0 &&
							visibleSelectionStats.selectedVisibleCount < visibleSelectionStats.visibleCount}
						visibleCount={visibleSelectionStats.visibleCount}
					/>
				</div>

				<!-- the footer, at the bottom of the sidebar and of its collapsed strip alike (a
				     phone lists its items with the actions above) -->
				{#if !isOnMobile}
					<SidebarNavigationFooter onAction={handleSidebarAction} />
				{/if}
			</div>
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

<DialogConversationDelete
	bind:open={deleteDialogOpen}
	onCancel={handleDeleteCancel}
	onConfirm={handleDeleteConfirm}
/>

<DialogSettingsChat bind:open={settingsDialogOpen} />

<DialogSearch bind:open={searchOpen} onAction={handleSidebarAction} />

<DialogMcpServers bind:open={mcpDialogOpen} />

<DialogDiscoverModels bind:open={uiStore.discoverModelsOpen} />

<DialogGroup />

<DialogGroupPicker />

<ModelsManagerModelDrawer />

<style>
	aside {
		@media (max-width: 768px) {
			--size: 1.125rem;
		}
	}
</style>
