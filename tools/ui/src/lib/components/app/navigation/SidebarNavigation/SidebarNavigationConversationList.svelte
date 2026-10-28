<script lang="ts">
	import SidebarNavigationConversationItem from './SidebarNavigationConversationItem.svelte';
	import SidebarNavigationSectionHeading from './SidebarNavigationSectionHeading.svelte';
	import SidebarNavigationSelectionBar from './SidebarNavigationSelectionBar.svelte';
	import { Pin } from '@lucide/svelte';
	import { CollapsibleRegion } from '$lib/components/app';
	import { SIDEBAR_SECTION_IDS } from '$lib/constants';
	import { uiStore } from '$lib/stores';
	import { buildConversationTree } from '$lib/utils';

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
</script>

<div class="flex min-h-0 flex-1 flex-col">
	{#if pinnedConversations.length > 0}
		<div class="py-2 whitespace-nowrap {className}">
			<SidebarNavigationSectionHeading
				icon={Pin}
				label="Pinned"
				onToggle={() => uiStore.toggleSidebarSection(SIDEBAR_SECTION_IDS.PINNED)}
				open={uiStore.isSidebarSectionOpen(SIDEBAR_SECTION_IDS.PINNED)}
			/>
		</div>

		<CollapsibleRegion open={uiStore.isSidebarSectionOpen(SIDEBAR_SECTION_IDS.PINNED)}>
			<ul class="flex w-full min-w-0 flex-col gap-4 md:gap-1 {className}">
				{#each pinnedConversations as { conversation, depth } (conversation.id)}
					<li class="group/item relative mb-1 p-0">
						{@render conversationItem(conversation, depth)}
					</li>
				{/each}
			</ul>
		</CollapsibleRegion>
	{/if}

	<div class="mt-2 flex min-h-0 flex-1 flex-col gap-4 md:gap-0 whitespace-nowrap {className}">
		{#if conversations.length > 0}
			<SidebarNavigationSectionHeading
				label="Recent conversations"
				onToggle={() => uiStore.toggleSidebarSection(SIDEBAR_SECTION_IDS.RECENT)}
				open={uiStore.isSidebarSectionOpen(SIDEBAR_SECTION_IDS.RECENT)}
			/>
		{/if}

		<div class="min-h-0 flex-1 md:overflow-y-auto">
			<CollapsibleRegion open={uiStore.isSidebarSectionOpen(SIDEBAR_SECTION_IDS.RECENT)}>
				<ul class="flex w-full min-w-0 flex-col gap-4 md:gap-0">
					{#each unpinnedConversations as { conversation, depth } (conversation.id)}
						<li class="group/item relative mb-1 p-0">
							{@render conversationItem(conversation, depth)}
						</li>
					{/each}

					{#if unpinnedConversations.length === 0}
						<li class="px-2 py-4 text-center">
							<p class="mb-4 p-4 text-sm text-muted-foreground">No conversations yet</p>
						</li>
					{/if}
				</ul>
			</CollapsibleRegion>
		</div>
	</div>

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
