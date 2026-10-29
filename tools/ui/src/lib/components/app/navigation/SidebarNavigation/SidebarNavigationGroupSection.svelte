<script lang="ts">
	import SidebarNavigationDropZone from './SidebarNavigationDropZone.svelte';
	import SidebarNavigationSectionHeading from './SidebarNavigationSectionHeading.svelte';
	import SidebarNavigationShowMore from './SidebarNavigationShowMore.svelte';
	import { MoreHorizontal, Pencil, Plus, SquarePen, Trash2 } from '@lucide/svelte';
	import {
		ActionIcon,
		CollapsibleRegion,
		ContextMenuActions,
		DialogConfirmation,
		DropdownMenuActions
	} from '$lib/components/app';
	import {
		SIDEBAR_ROW_SLIDE_MS,
		SIDEBAR_SECTION_IDS,
		SIDEBAR_SECTION_PREVIEW_COUNT,
		SIDEBAR_SHOW_MORE_ROW_CLASS
	} from '$lib/constants';
	import { conversationsStore, groupsStore, sidebarDragStore, uiStore } from '$lib/stores';
	import type { DatabaseConversation, Group, MenuAction } from '$lib/types';
	import type { Snippet } from 'svelte';
	import { slide } from 'svelte/transition';

	interface Props {
		class?: string;
		conversations: DatabaseConversation[];
		group: Group;
		/** Row renderer from the list, which owns the per-chat actions. */
		conversationRow: Snippet<[DatabaseConversation]>;
	}

	let { class: className = '', conversationRow, conversations, group }: Props = $props();

	let contextMenuOpen = $state(false);
	let dropdownOpen = $state(false);
	let confirmDeleteOpen = $state(false);

	let sectionId = $derived(`${SIDEBAR_SECTION_IDS.GROUP_PREFIX}${group.id}`);
	let open = $derived(uiStore.isSidebarSectionOpen(sectionId));
	let expanded = $derived(uiStore.isSidebarSectionExpanded(sectionId));
	let shownConversations = $derived.by(() => {
		// a dragged chat leaves its section until it is dropped somewhere
		const shown = conversations.filter(
			(conversation) => conversation.id !== sidebarDragStore.convId
		);

		return expanded ? shown : shown.slice(0, SIDEBAR_SECTION_PREVIEW_COUNT);
	});

	async function startChat() {
		await conversationsStore.openNewChat();
		await conversationsStore.assignActiveConversationGroup(group.id);
	}

	async function handleDelete() {
		confirmDeleteOpen = false;
		await groupsStore.remove(group.id);
	}

	/** A chat dropped on the section joins the group, and leaves Pinned. */
	async function handleDrop(convId: string) {
		const conversation = conversationsStore.conversations.find((item) => item.id === convId);

		if (conversation?.pinned) await conversationsStore.toggleConversationPin(convId);

		await conversationsStore.setConversationGroup(convId, group.id);
	}

	// the heading's "more actions" dropdown and its right-click menu offer the same actions
	const actions: MenuAction[] = [
		{ icon: SquarePen, label: 'New chat', onclick: () => void startChat() },
		{
			icon: Pencil,
			label: 'Rename',
			onclick: () => uiStore.openGroupDialog({ groupId: group.id }),
			separator: true
		},
		{
			icon: Trash2,
			label: 'Delete',
			onclick: () => (confirmDeleteOpen = true),
			separator: true,
			variant: 'destructive'
		}
	];
</script>

<!-- a group is a section like Pinned and Recent: its chats show here only,
     and a chat dropped anywhere on it joins the group -->
<SidebarNavigationDropZone
	class="flex min-w-0 flex-col"
	holds={(convId) => conversations.some((conversation) => conversation.id === convId)}
	onDrop={handleDrop}
	{sectionId}
>
	{#snippet heading()}
		<ContextMenuActions bind:open={contextMenuOpen} {actions}>
			<SidebarNavigationSectionHeading
				class={className}
				dragId={sectionId}
				highlighted={contextMenuOpen || dropdownOpen}
				label={group.name}
				onToggle={() => uiStore.toggleSidebarSection(sectionId)}
				{open}
			>
				{#snippet trailing()}
					<div
						class="flex items-center gap-0.5 opacity-0 transition-opacity group-focus-within/heading:opacity-100 group-hover/heading:opacity-100 {dropdownOpen
							? 'opacity-100'
							: ''}"
					>
						<ActionIcon
							ariaLabel="New chat in {group.name}"
							class="h-6 w-6"
							icon={Plus}
							iconSize="h-3 w-3"
							onclick={() => void startChat()}
							tooltip="New chat in group"
						/>

						<DropdownMenuActions
							bind:open={dropdownOpen}
							{actions}
							triggerIcon={MoreHorizontal}
							triggerTooltip="Group actions"
						/>
					</div>
				{/snippet}
			</SidebarNavigationSectionHeading>
		</ContextMenuActions>
	{/snippet}

	<CollapsibleRegion {open}>
		<ul class="flex w-full min-w-0 flex-col gap-4 md:gap-0.5 {className}">
			{#each shownConversations as conversation (conversation.id)}
				<li transition:slide={{ duration: SIDEBAR_ROW_SLIDE_MS }} class="group/item relative p-0">
					{@render conversationRow(conversation)}
				</li>
			{:else}
				<li>
					<button
						class="w-full rounded-lg px-2 py-1.5 text-left text-xs text-muted-foreground transition-colors hover:bg-foreground/10 hover:text-foreground"
						onclick={() => void startChat()}
						type="button"
					>
						Start a chat in this group
					</button>
				</li>
			{/each}

			{#if conversations.length > SIDEBAR_SECTION_PREVIEW_COUNT}
				<!-- only an opened section is long enough for its row to need to stick -->
				<li class={expanded ? SIDEBAR_SHOW_MORE_ROW_CLASS : 'p-0'}>
					<SidebarNavigationShowMore
						{expanded}
						onToggle={() => uiStore.toggleSidebarSectionExpanded(sectionId)}
					/>
				</li>
			{/if}
		</ul>
	</CollapsibleRegion>
</SidebarNavigationDropZone>

<DialogConfirmation
	confirmText="Delete"
	description="Its chats stay in your history, outside any group."
	onCancel={() => (confirmDeleteOpen = false)}
	onConfirm={handleDelete}
	open={confirmDeleteOpen}
	title="Delete {group.name}?"
	variant="destructive"
/>
