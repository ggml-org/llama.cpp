<script lang="ts">
	import SidebarNavigationGroupSubmenu from './SidebarNavigation/SidebarNavigationGroupSubmenu.svelte';
	import {
		Download,
		FolderInput,
		ListChecks,
		PanelTopOpen,
		Pencil,
		Pin,
		PinOff,
		Trash2,
		Unlink
	} from '@lucide/svelte';
	import {
		HUGGING_FACE_ENABLED,
		SHARE_ACTION,
		SHORTCUT_DELETE_CONVERSATION,
		SHORTCUT_EDIT_CONVERSATION,
		SHORTCUT_EXPORT_CONVERSATION
	} from '#lib/constants/index.ts';
	import type { MenuKind } from '#lib/enums/index.ts';
	import {
		conversationsStore,
		groupsStore,
		settingsStore,
		tabsStore,
		uiStore
	} from '#lib/stores/index.ts';
	import type { MenuAction } from '#lib/types/index.ts';
	import type { Snippet } from 'svelte';

	const MOVE_TO_GROUP_LABEL = 'Move to group';

	interface Props {
		conversation: DatabaseConversation;
		/** The active conversation's menu shows the shortcuts, which act on it. */
		isActive?: boolean;
		/** Offer "Open in new tab"; a tab's own menu leaves it out. */
		showOpenInNewTab?: boolean;
		/** Actions of the caller's own, listed before the conversation's, with a separator between. */
		leadingActions?: MenuAction[];
		/** Enter the sidebar's selection mode; the action shows only when given. */
		onEnterSelectionMode?: () => void;
		/** Renders the menu that shows the actions. */
		children: Snippet<[MenuAction[]]>;
	}

	let {
		children,
		conversation,
		isActive = false,
		leadingActions = [],
		onEnterSelectionMode,
		showOpenInNewTab = true
	}: Props = $props();

	let canOpenInNewTab = $derived(
		showOpenInNewTab && Boolean(settingsStore.config.conversationTabs)
	);

	// a group the chat points at may be gone; then it is in none
	let groupId = $derived(groupsStore.groupIdOf(conversation));

	// menus sit on clickable rows and tabs, so an action must not also click them
	function run(action: () => void) {
		return (event: Event) => {
			event.stopPropagation();
			action();
		};
	}

	function setGroup(groupId: string | null) {
		void conversationsStore.setConversationGroup(conversation.id, groupId);
	}

	let conversationActions = $derived<MenuAction[]>([
		...(canOpenInNewTab
			? [
					{
						icon: PanelTopOpen,
						label: 'Open in new tab',
						onclick: run(() => void tabsStore.openInNewTab(conversation.id))
					}
				]
			: []),
		{
			icon: conversation.pinned ? PinOff : Pin,
			label: conversation.pinned ? 'Unpin' : 'Pin',
			onclick: run(() => void conversationsStore.toggleConversationPin(conversation.id)),
			separator: canOpenInNewTab
		},
		{
			icon: Pencil,
			label: 'Edit',
			onclick: run(() => uiStore.openConversationRename(conversation.id)),
			shortcut: isActive ? SHORTCUT_EDIT_CONVERSATION : undefined
		},
		// the picker stands in for the submenu where a phone's menu shows no flyouts
		{
			icon: FolderInput,
			label: MOVE_TO_GROUP_LABEL,
			onclick: run(() => uiStore.openGroupPicker({ currentGroupId: groupId, onSelect: setGroup })),
			submenu: groupSubmenu
		},
		...(groupId
			? [{ icon: Unlink, label: 'Remove from group', onclick: run(() => setGroup(null)) }]
			: []),
		{
			icon: Download,
			label: 'Export',
			onclick: run(() => void conversationsStore.downloadConversation(conversation.id)),
			shortcut: isActive ? SHORTCUT_EXPORT_CONVERSATION : undefined
		},
		...(HUGGING_FACE_ENABLED
			? [
					{
						...SHARE_ACTION,
						onclick: run(() => uiStore.openConversationShare({ conversationId: conversation.id }))
					}
				]
			: []),
		...(onEnterSelectionMode
			? [{ icon: ListChecks, label: 'Select', onclick: run(onEnterSelectionMode) }]
			: []),
		{
			icon: Trash2,
			label: 'Delete',
			onclick: run(() => uiStore.openConversationDelete(conversation.id)),
			separator: true,
			shortcut: isActive ? SHORTCUT_DELETE_CONVERSATION : undefined,
			variant: 'destructive'
		}
	]);

	// the caller's own actions come first, and the conversation's start under a separator
	let actions = $derived<MenuAction[]>(
		leadingActions.length
			? [
					...leadingActions,
					{ ...conversationActions[0], separator: true },
					...conversationActions.slice(1)
				]
			: conversationActions
	);
</script>

{@render children(actions)}

{#snippet groupSubmenu({ close, kind }: { close: () => void; kind: MenuKind })}
	<SidebarNavigationGroupSubmenu
		currentGroupId={groupId}
		{kind}
		label={MOVE_TO_GROUP_LABEL}
		onClose={close}
		onSelect={setGroup}
	/>
{/snippet}
