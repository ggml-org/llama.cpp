<script lang="ts">
	import SidebarNavigationConversationItem from './SidebarNavigationConversationItem.svelte';
	import SidebarNavigationDropZone from './SidebarNavigationDropZone.svelte';
	import SidebarNavigationSectionHeading from './SidebarNavigationSectionHeading.svelte';
	import SidebarNavigationSelectionBar from './SidebarNavigationSelectionBar.svelte';
	import SidebarNavigationShowMore from './SidebarNavigationShowMore.svelte';
	import { Pin } from '@lucide/svelte';
	import { CollapsibleRegion } from '$lib/components/app';
	import {
		SIDEBAR_ROW_SLIDE_MS,
		SIDEBAR_SECTION_IDS,
		SIDEBAR_SECTION_PREVIEW_COUNT,
		SIDEBAR_SHOW_MORE_ROW_CLASS
	} from '$lib/constants';
	import { SidebarSectionKind } from '$lib/enums';
	import { conversationsStore, sidebarDragStore, uiStore } from '$lib/stores';
	import { buildConversationTree } from '$lib/utils';
	import { flip } from 'svelte/animate';
	import { slide } from 'svelte/transition';

	interface Props {
		class: string;
		conversations: DatabaseConversation[];
		currentChatId: string | undefined;
		isSelectionMode?: boolean;
		selectedIds?: Set<string>;
		onSelect: (id: string) => void;
		onEdit: (id: string) => void;
		onDelete: (id: string) => void;
		onStop: (id: string) => void;
		onToggleSelect?: (id: string) => void;
		onEnterSelectionMode?: (id: string) => void;
		onSelectionClick?: (id: string, options: { shiftKey: boolean }) => void;
		onRowMouseDown?: (id: string, event: MouseEvent) => void;
		visibleCount: number;
		allVisibleSelected: boolean;
		someVisibleSelected: boolean;
		allSelectedArePinned: boolean;
		pinStateIsMixed: boolean;
		onSelectAllToggle: () => void;
		onBulkPinToggle: () => void;
		onBulkExport: () => void;
		onBulkDelete: () => void;
		onCloseSelection: () => void;
	}

	let {
		allSelectedArePinned,
		allVisibleSelected,
		class: className,
		conversations,
		currentChatId,
		isSelectionMode = false,
		onBulkDelete,
		onBulkExport,
		onBulkPinToggle,
		onCloseSelection,
		onDelete,
		onEdit,
		onEnterSelectionMode,
		onRowMouseDown,
		onSelect,
		onSelectAllToggle,
		onSelectionClick,
		onStop,
		onToggleSelect,
		pinStateIsMixed,
		selectedIds = new Set<string>(),
		someVisibleSelected,
		visibleCount
	}: Props = $props();

	let conversationTree = $derived(buildConversationTree(conversations));

	let pinnedConversations = $derived(
		conversationTree.filter(({ conversation }) => conversation.pinned)
	);

	let unpinnedConversations = $derived(
		conversationTree.filter(({ conversation }) => !conversation.pinned)
	);

	/** How long the other sections take to slide aside for a moved one. */
	const SECTION_MOVE_MS = 200;

	// Pinned, then Recent by default; dragging a heading reorders them
	let savedSections = $derived(
		uiStore.orderSidebarSections([
			...(pinnedConversations.length > 0
				? [{ id: SIDEBAR_SECTION_IDS.PINNED as string, kind: SidebarSectionKind.PINNED as const }]
				: []),
			{ id: SIDEBAR_SECTION_IDS.RECENT as string, kind: SidebarSectionKind.RECENT as const }
		])
	);

	// while a section is dragged the others make room for it as it passes them
	let sections = $derived.by(() => {
		const order = sidebarDragStore.sectionOrder;

		if (!order) return savedSections;

		return [...savedSections].sort((a, b) => order.indexOf(a.id) - order.indexOf(b.id));
	});

	// a section lists its first chats until its "Show more" row opens all of it
	function previewOf<T extends { conversation: DatabaseConversation }>(
		sectionId: string,
		items: T[]
	): T[] {
		// a dragged chat leaves its section until it is dropped somewhere
		const shown = items.filter(({ conversation }) => conversation.id !== sidebarDragStore.convId);

		return uiStore.isSidebarSectionExpanded(sectionId)
			? shown
			: shown.slice(0, SIDEBAR_SECTION_PREVIEW_COUNT);
	}

	function findConversation(convId: string) {
		return conversationsStore.conversations.find((item) => item.id === convId);
	}

	function isPinned(convId: string): boolean {
		return Boolean(findConversation(convId)?.pinned);
	}

	function isUnpinned(convId: string): boolean {
		const conversation = findConversation(convId);

		return Boolean(conversation && !conversation.pinned);
	}

	/** Dropping on Pinned pins the chat, wherever it came from. */
	async function handleDropPinned(convId: string) {
		const conversation = conversationsStore.conversations.find((item) => item.id === convId);

		if (conversation && !conversation.pinned)
			await conversationsStore.toggleConversationPin(convId);
	}

	/** Dropping on Recent unpins the chat. */
	async function handleDropRecent(convId: string) {
		const conversation = conversationsStore.conversations.find((item) => item.id === convId);

		if (conversation?.pinned) await conversationsStore.toggleConversationPin(convId);
	}
</script>

<div class="flex flex-col gap-2">
	{#each sections as section (section.id)}
		{@const isDragged = section.id === sidebarDragStore.sectionId}

		<!-- the dragged section follows the pointer and, let go, glides into its slot;
		     the others slide aside, the dragged one is already where the pointer is -->
		<div
			animate:flip={{ duration: isDragged ? 0 : SECTION_MOVE_MS }}
			style:transform={isDragged ? `translateY(${sidebarDragStore.sectionOffset}px)` : undefined}
			class={[
				// opaque under its tint, so the sections it passes do not show through
				isDragged && 'sidebar-drag-surface relative z-20 rounded-lg',
				isDragged &&
					sidebarDragStore.settling &&
					'transition-transform duration-200 ease-[cubic-bezier(0.23,1,0.32,1)]'
			]}
			data-sidebar-slot={section.id}
		>
			{#if section.kind === SidebarSectionKind.PINNED}
				{@render pinnedSection()}
			{:else}
				{@render recentSection()}
			{/if}
		</div>
	{/each}

	{#if isSelectionMode}
		<SidebarNavigationSelectionBar
			{allVisibleSelected}
			class="sticky top-0 z-10 m-2 mt-0"
			{onBulkDelete}
			{onBulkExport}
			{onBulkPinToggle}
			onClose={onCloseSelection}
			{onSelectAllToggle}
			{pinStateIsMixed}
			selectedCount={selectedIds.size}
			someSelectedPinned={allSelectedArePinned}
			{someVisibleSelected}
			{visibleCount}
		/>
	{/if}
</div>

{#snippet showMore(sectionId: string, total: number)}
	{#if total > SIDEBAR_SECTION_PREVIEW_COUNT}
		<!-- only an opened section is long enough for its row to need to stick -->
		<li
			class={uiStore.isSidebarSectionExpanded(sectionId) ? SIDEBAR_SHOW_MORE_ROW_CLASS : 'mb-1 p-0'}
		>
			<SidebarNavigationShowMore
				expanded={uiStore.isSidebarSectionExpanded(sectionId)}
				onToggle={() => uiStore.toggleSidebarSectionExpanded(sectionId)}
			/>
		</li>
	{/if}
{/snippet}

{#snippet conversationItem(conversation: DatabaseConversation, depth = 0)}
	<SidebarNavigationConversationItem
		conversation={{
			currNode: conversation.currNode,
			forkedFromConversationId: conversation.forkedFromConversationId,
			id: conversation.id,
			lastModified: conversation.lastModified,
			name: conversation.name,
			pinned: conversation.pinned
		}}
		{depth}
		isActive={currentChatId === conversation.id}
		isSelected={selectedIds.has(conversation.id)}
		{isSelectionMode}
		{onDelete}
		{onEdit}
		{onEnterSelectionMode}
		{onRowMouseDown}
		{onSelect}
		{onSelectionClick}
		{onStop}
		{onToggleSelect}
	/>
{/snippet}

{#snippet pinnedSection()}
	<SidebarNavigationDropZone
		class="flex flex-col"
		holds={(convId) => isPinned(convId)}
		onDrop={handleDropPinned}
		sectionId={SIDEBAR_SECTION_IDS.PINNED}
	>
		{#snippet heading()}
			<div class="whitespace-nowrap {className}">
				<SidebarNavigationSectionHeading
					dragId={SIDEBAR_SECTION_IDS.PINNED}
					icon={Pin}
					label="Pinned"
					onToggle={() => uiStore.toggleSidebarSection(SIDEBAR_SECTION_IDS.PINNED)}
					open={uiStore.isSidebarSectionOpen(SIDEBAR_SECTION_IDS.PINNED)}
				/>
			</div>
		{/snippet}

		<CollapsibleRegion open={uiStore.isSidebarSectionOpen(SIDEBAR_SECTION_IDS.PINNED)}>
			<ul class="flex w-full min-w-0 flex-col gap-4 md:gap-1 {className}">
				{#each previewOf(SIDEBAR_SECTION_IDS.PINNED, pinnedConversations) as { conversation, depth } (conversation.id)}
					<li
						transition:slide={{ duration: SIDEBAR_ROW_SLIDE_MS }}
						class="group/item relative mb-1 p-0"
					>
						{@render conversationItem(conversation, depth)}
					</li>
				{/each}

				{@render showMore(SIDEBAR_SECTION_IDS.PINNED, pinnedConversations.length)}
			</ul>
		</CollapsibleRegion>
	</SidebarNavigationDropZone>
{/snippet}

{#snippet recentSection()}
	<SidebarNavigationDropZone
		class="flex flex-col gap-4 md:gap-0 whitespace-nowrap {className}"
		holds={(convId) => isUnpinned(convId)}
		onDrop={handleDropRecent}
		sectionId={SIDEBAR_SECTION_IDS.RECENT}
	>
		{#snippet heading()}
			{#if conversations.length > 0}
				<SidebarNavigationSectionHeading
					dragId={SIDEBAR_SECTION_IDS.RECENT}
					label="Recent conversations"
					onToggle={() => uiStore.toggleSidebarSection(SIDEBAR_SECTION_IDS.RECENT)}
					open={uiStore.isSidebarSectionOpen(SIDEBAR_SECTION_IDS.RECENT)}
				/>
			{/if}
		{/snippet}

		<CollapsibleRegion open={uiStore.isSidebarSectionOpen(SIDEBAR_SECTION_IDS.RECENT)}>
			<ul class="flex w-full min-w-0 flex-col gap-4 md:gap-0">
				{#each previewOf(SIDEBAR_SECTION_IDS.RECENT, unpinnedConversations) as { conversation, depth } (conversation.id)}
					<li
						transition:slide={{ duration: SIDEBAR_ROW_SLIDE_MS }}
						class="group/item relative mb-1 p-0"
					>
						{@render conversationItem(conversation, depth)}
					</li>
				{/each}

				{#if conversations.length === 0}
					<li class="px-2 py-4 text-center">
						<p class="mb-4 p-4 text-sm text-muted-foreground">No conversations yet</p>
					</li>
				{/if}

				{@render showMore(SIDEBAR_SECTION_IDS.RECENT, unpinnedConversations.length)}
			</ul>
		</CollapsibleRegion>
	</SidebarNavigationDropZone>
{/snippet}
