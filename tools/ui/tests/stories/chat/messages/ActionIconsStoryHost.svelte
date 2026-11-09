<script lang="ts">
	// Story-only host: sets the per-message contexts that ChatMessage.svelte
	// normally provides, so ChatMessageActionIcons can render alone.
	import ChatMessageActionIcons from '$lib/components/app/chat/ChatMessages/ChatMessageActions/ChatMessageActionIcons/ChatMessageActionIcons.svelte';
	import { setChatMessageActionsContext } from '$lib/contexts';
	import { useChatMessageEditContext } from '$lib/hooks/use-chat-message-edit-context.svelte';
	import type { ChatMessageDeletionInfo, ChatMessageSiblingInfo } from '$lib/types';

	interface Props {
		justify: 'start' | 'end';
		actionsPosition: 'left' | 'right';
		siblingInfo?: ChatMessageSiblingInfo | null;
		deletionInfo?: ChatMessageDeletionInfo | null;
		showDeleteDialog?: boolean;
		forkable?: boolean;
		onRegenerate?: () => void;
		onContinue?: () => void;
		showRawOutputSwitch?: boolean;
	}

	let {
		actionsPosition,
		deletionInfo = null,
		forkable = true,
		justify,
		onContinue,
		onRegenerate,
		showDeleteDialog = false,
		showRawOutputSwitch = false,
		siblingInfo = null
	}: Props = $props();

	let currentSiblingInfo = $derived(siblingInfo);
	let deleteDialogOpen = $derived(showDeleteDialog);
	let forkDialogOpen = $state(false);
	let rawOutputEnabled = $state(false);

	useChatMessageEditContext({
		getContent: () => '',
		getExtras: () => [],
		onSave: () => {}
	});

	setChatMessageActionsContext({
		confirmDelete: () => {},
		copy: () => {},
		get deletionInfo() {
			return deletionInfo;
		},
		get forkConversation() {
			return forkable ? () => {} : undefined;
		},
		navigateToSibling: (siblingId: string) => {
			if (!currentSiblingInfo) return;

			currentSiblingInfo = {
				...currentSiblingInfo,
				currentIndex: currentSiblingInfo.siblingIds.indexOf(siblingId)
			};
		},
		requestDelete: () => {
			deleteDialogOpen = true;
		},
		setShowDeleteDialog: (show: boolean) => {
			deleteDialogOpen = show;
		},
		setShowForkDialog: (show: boolean) => {
			forkDialogOpen = show;
		},
		get showDeleteDialog() {
			return deleteDialogOpen;
		},
		get showForkDialog() {
			return forkDialogOpen;
		},
		get siblingInfo() {
			return currentSiblingInfo;
		}
	});
</script>

<ChatMessageActionIcons
	{actionsPosition}
	{justify}
	{onContinue}
	onRawOutputToggle={(enabled) => (rawOutputEnabled = enabled)}
	{onRegenerate}
	{rawOutputEnabled}
	{showRawOutputSwitch}
/>
