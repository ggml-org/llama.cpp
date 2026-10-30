<script lang="ts">
	import SidebarNavigationDropZone from './SidebarNavigationDropZone.svelte';
	import SidebarNavigationGroupSubmenu from './SidebarNavigationGroupSubmenu.svelte';
	import SidebarNavigationShowMore from './SidebarNavigationShowMore.svelte';
	import {
		Archive,
		ChevronRight,
		Download,
		FolderInput,
		FolderOpen,
		MoreHorizontal,
		Pencil,
		Pin,
		PinOff,
		Plus,
		SquarePen,
		Trash2,
		Unlink
	} from '@lucide/svelte';
	import { goto } from '$app/navigation';
	import {
		ActionIcon,
		CollapsibleRegion,
		ContextMenuActions,
		DialogConfirmation,
		DropdownMenuActions,
		TruncatedText
	} from '$lib/components/app';
	import {
		ICON_CLASS_DEFAULT,
		PROJECT_ICON,
		SIDEBAR_ROW_SLIDE_MS,
		SIDEBAR_SECTION_IDS,
		SIDEBAR_SECTION_PREVIEW_COUNT,
		SIDEBAR_SHOW_MORE_ROW_CLASS,
		UI_DATA_ATTRS
	} from '$lib/constants';
	import type { MenuKind } from '$lib/enums';
	import { RouterService } from '$lib/services';
	import {
		conversationsStore,
		groupsStore,
		projectsStore,
		sidebarDragStore,
		uiStore
	} from '$lib/stores';
	import type { DatabaseConversation, MenuAction, Project } from '$lib/types';
	import type { Snippet } from 'svelte';
	import { slide } from 'svelte/transition';

	interface Props {
		/** The project's chats, newest first. */
		conversations: DatabaseConversation[];
		/** Chat open right now, to mark a folded project that holds it. */
		currentChatId?: string;
		/** Whether the list is picking chats; the row then only folds. */
		isSelectionMode?: boolean;
		project: Project;
		/** Row renderer from the list, which owns the per-chat actions. */
		conversationRow: Snippet<[DatabaseConversation]>;
	}

	let {
		conversationRow,
		conversations,
		currentChatId,
		isSelectionMode = false,
		project
	}: Props = $props();

	let contextMenuOpen = $state(false);
	let dropdownOpen = $state(false);
	let confirmDeleteOpen = $state(false);

	// folding and "Show more" are remembered per project, like a section's
	let foldId = $derived(`${SIDEBAR_SECTION_IDS.PROJECT_PREFIX}${project.id}`);
	let dragId = $derived(sidebarDragStore.projectDragId(project.id));
	let open = $derived(uiStore.isSidebarSectionOpen(foldId));
	let expanded = $derived(uiStore.isSidebarSectionExpanded(foldId));
	let isLanding = $derived(sidebarDragStore.landingConvId === dragId);
	let holdsActive = $derived(
		conversations.some((conversation) => conversation.id === currentChatId)
	);

	let shownConversations = $derived.by(() => {
		// a dragged chat leaves its project until it is dropped somewhere
		const shown = conversations.filter(
			(conversation) => conversation.id !== sidebarDragStore.convId
		);

		return expanded ? shown : shown.slice(0, SIDEBAR_SECTION_PREVIEW_COUNT);
	});

	// a group the project points at may be gone; then it is in none
	let groupId = $derived(
		project.groupId && groupsStore.get(project.groupId) ? project.groupId : null
	);

	// the project's page is where its new chats start
	function startChat() {
		void goto(RouterService.project(project.id));
	}

	async function handleDelete() {
		confirmDeleteOpen = false;
		await projectsStore.remove(project.id);
	}

	// the name drags the row; its own controls (new chat, its menu) keep their press
	function handlePointerDown(event: PointerEvent) {
		if (!isSelectionMode) sidebarDragStore.pressConversation(event, dragId);
	}

	function setGroup(id: string | null) {
		void projectsStore.setGroup(project.id, id);
	}

	/** A chat dropped on the project joins it, and leaves Pinned. */
	async function handleDrop(dropped: string) {
		const conversation = conversationsStore.conversations.find((item) => item.id === dropped);

		if (!conversation) return;

		if (conversation.pinned) await conversationsStore.toggleConversationPin(dropped);

		await conversationsStore.setConversationProject(dropped, project.id);
	}

	// another project, or a chat in this one already, has nowhere to go here
	function holds(dropped: string): boolean {
		return (
			sidebarDragStore.projectIdOf(dropped) !== null ||
			conversations.some((conversation) => conversation.id === dropped)
		);
	}

	// the "more actions" dropdown and the right-click menu offer the same actions
	let actions = $derived<MenuAction[]>([
		{ icon: FolderOpen, label: 'Open project', onclick: startChat },
		{ icon: SquarePen, label: 'New chat', onclick: startChat },
		{
			icon: project.pinned ? PinOff : Pin,
			label: project.pinned ? 'Unpin' : 'Pin',
			onclick: () => void projectsStore.setPinned(project.id, !project.pinned),
			separator: true
		},
		{
			icon: Pencil,
			label: 'Edit project',
			onclick: () => uiStore.openProjectDialog({ projectId: project.id })
		},
		{
			icon: FolderInput,
			label: 'Move to group',
			// a phone has no flyouts: the same entry opens the group picker there
			onclick: () => uiStore.openGroupPicker({ currentGroupId: groupId, onSelect: setGroup }),
			submenu: groupSubmenu
		},
		...(groupId
			? [{ icon: Unlink, label: 'Remove from group', onclick: () => setGroup(null) }]
			: []),
		{
			icon: Download,
			label: 'Export',
			onclick: () => void projectsStore.exportArchive(project.id)
		},
		{
			icon: Archive,
			label: 'Archive',
			onclick: () => void projectsStore.setArchived(project.id, true)
		},
		{
			icon: Trash2,
			label: 'Delete',
			onclick: () => (confirmDeleteOpen = true),
			separator: true,
			variant: 'destructive'
		}
	]);
</script>

<!-- a project is a row with its chats under it, and sits in Pinned, a group or
     Ungrouped like a chat; a chat dropped anywhere on it joins the project -->
<SidebarNavigationDropZone
	class="flex min-w-0 flex-col"
	{holds}
	onDrop={handleDrop}
	sectionId={foldId}
>
	{#snippet heading()}
		<ContextMenuActions bind:open={contextMenuOpen} {actions} disabled={isSelectionMode}>
			<div
				class="group/project flex min-h-9 w-full items-center gap-1 rounded-lg pr-1 transition-colors hover:bg-foreground/10 hover:text-foreground md:min-h-8 {isLanding
					? 'invisible'
					: ''} {holdsActive && !open
					? 'bg-foreground/5 text-accent-foreground'
					: 'text-foreground/75'} {contextMenuOpen || dropdownOpen ? 'bg-foreground/10' : ''}"
				{...{ [UI_DATA_ATTRS.CONVERSATION_ROW]: dragId }}
			>
				<button
					aria-expanded={open}
					class="flex min-w-0 flex-1 cursor-pointer items-center gap-2 self-stretch rounded-lg pl-2 text-left"
					onclick={() => uiStore.toggleSidebarSection(foldId)}
					ondragstart={(event) => event.preventDefault()}
					onpointerdown={handlePointerDown}
					type="button"
				>
					<!-- the icon turns into the fold chevron while the row is hovered -->
					<span class="relative flex {ICON_CLASS_DEFAULT} shrink-0 items-center justify-center">
						<PROJECT_ICON
							class="{ICON_CLASS_DEFAULT} transition-opacity group-hover/project:opacity-0"
						/>

						<ChevronRight
							aria-hidden="true"
							class="absolute {ICON_CLASS_DEFAULT} opacity-0 transition-[opacity,rotate] duration-200 ease-out group-hover/project:opacity-100 {open
								? 'rotate-90'
								: ''}"
						/>
					</span>

					<TruncatedText class="text-sm" showTooltip={false} text={project.name} />
				</button>

				{#if !isSelectionMode}
					<div
						class="flex shrink-0 items-center gap-0.5 opacity-0 transition-opacity group-focus-within/project:opacity-100 group-hover/project:opacity-100 max-md:opacity-100 {dropdownOpen
							? 'opacity-100'
							: ''}"
					>
						<ActionIcon
							ariaLabel="New chat in {project.name}"
							class="h-6 w-6"
							icon={Plus}
							iconSize="h-3 w-3"
							onclick={startChat}
							tooltip="New chat in project"
						/>

						<DropdownMenuActions
							bind:open={dropdownOpen}
							{actions}
							triggerIcon={MoreHorizontal}
							triggerTooltip="Project actions"
						/>
					</div>
				{/if}
			</div>
		</ContextMenuActions>
	{/snippet}

	<CollapsibleRegion {open}>
		<!-- the chats hang off the project's icon on a thin guide line -->
		<ul
			class="ml-[15px] flex min-w-0 flex-col gap-4 border-l border-foreground/10 pl-1.5 md:gap-0.5"
		>
			{#each shownConversations as conversation (conversation.id)}
				<li transition:slide={{ duration: SIDEBAR_ROW_SLIDE_MS }} class="group/item relative p-0">
					{@render conversationRow(conversation)}
				</li>
			{:else}
				<li>
					<button
						class="w-full rounded-lg px-2 py-1.5 text-left text-xs text-muted-foreground transition-colors hover:bg-foreground/10 hover:text-foreground"
						onclick={startChat}
						type="button"
					>
						Start a chat in this project
					</button>
				</li>
			{/each}

			{#if conversations.length > SIDEBAR_SECTION_PREVIEW_COUNT}
				<li class={expanded ? SIDEBAR_SHOW_MORE_ROW_CLASS : 'p-0'}>
					<SidebarNavigationShowMore
						{expanded}
						onToggle={() => uiStore.toggleSidebarSectionExpanded(foldId)}
					/>
				</li>
			{/if}
		</ul>
	</CollapsibleRegion>
</SidebarNavigationDropZone>

{#snippet groupSubmenu({ close, kind }: { close: () => void; kind: MenuKind })}
	<SidebarNavigationGroupSubmenu
		currentGroupId={groupId}
		{kind}
		label="Move to group"
		onClose={close}
		onSelect={setGroup}
	/>
{/snippet}

<DialogConfirmation
	confirmText="Delete"
	description="Its chats stay in your history, outside any project."
	onCancel={() => (confirmDeleteOpen = false)}
	onConfirm={handleDelete}
	open={confirmDeleteOpen}
	title="Delete {project.name}?"
	variant="destructive"
/>
