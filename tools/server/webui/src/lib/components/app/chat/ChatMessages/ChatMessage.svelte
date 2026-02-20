<script lang="ts">
	import { goto } from '$app/navigation';
	import { base } from '$app/paths';
	import { getChatActionsContext, setMessageEditContext } from '$lib/contexts';
	import { chatStore, pendingEditMessageId } from '$lib/stores/chat.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { DatabaseService } from '$lib/services';
	import { SYSTEM_MESSAGE_PLACEHOLDER } from '$lib/constants/ui';
	import { MessageRole } from '$lib/enums';
	import {
		ChatMessageAssistant,
		ChatMessageUser,
		ChatMessageSystem
	} from '$lib/components/app/chat';
	import { parseFilesToMessageExtras } from '$lib/utils/browser-only';

	interface Props {
		class?: string;
		message: DatabaseMessage;
		isLastAssistantMessage?: boolean;
		siblingInfo?: ChatMessageSiblingInfo | null;
	}

	let {
		class: className = '',
		message,
		isLastAssistantMessage = false,
		siblingInfo = null
	}: Props = $props();

	const chatActions = getChatActionsContext();

	let deletionInfo = $state<{
		totalCount: number;
		userMessages: number;
		assistantMessages: number;
		messageTypes: string[];
	} | null>(null);
	let editedContent = $derived(message.content);
	let editedExtras = $derived<DatabaseMessageExtra[]>(message.extra ? [...message.extra] : []);
	let editedUploadedFiles = $state<ChatUploadedFile[]>([]);
	let isEditing = $state(false);
	let showDeleteDialog = $state(false);
	let shouldBranchAfterEdit = $state(false);
	let textareaElement: HTMLTextAreaElement | undefined = $state();

	let showSaveOnlyOption = $derived(message.role === MessageRole.USER);

	setMessageEditContext({
		get isEditing() {
			return isEditing;
		},
		get editedContent() {
			return editedContent;
		},
		get editedExtras() {
			return editedExtras;
		},
		get editedUploadedFiles() {
			return editedUploadedFiles;
		},
		get originalContent() {
			return message.content;
		},
		get originalExtras() {
			return message.extra || [];
		},
		get showSaveOnlyOption() {
			return showSaveOnlyOption;
		},
		setContent: (content: string) => {
			editedContent = content;
		},
		setExtras: (extras: DatabaseMessageExtra[]) => {
			editedExtras = extras;
		},
		setUploadedFiles: (files: ChatUploadedFile[]) => {
			editedUploadedFiles = files;
		},
		save: handleSaveEdit,
		saveOnly: handleSaveEditOnly,
		cancel: handleCancelEdit,
		startEdit: handleEdit
	});

	$effect(() => {
		const pendingId = pendingEditMessageId();

		if (pendingId && pendingId === message.id && !isEditing) {
			handleEdit();
			chatStore.clearPendingEditMessageId();
		}
	});

	async function handleCancelEdit() {
		isEditing = false;

		// If canceling a new system message with placeholder content, remove it without deleting children
		if (message.role === MessageRole.SYSTEM) {
			const conversationDeleted = await chatStore.removeSystemPromptPlaceholder(message.id);

			if (conversationDeleted) {
				goto(`${base}/`);
			}

			return;
		}

		editedContent = message.content;
		editedExtras = message.extra ? [...message.extra] : [];
		editedUploadedFiles = [];
	}

	function handleCopy() {
		chatActions.copy(message);
	}

	async function handleConfirmDelete() {
		if (message.role === MessageRole.SYSTEM) {
			const conversationDeleted = await chatStore.removeSystemPromptPlaceholder(message.id);

			if (conversationDeleted) {
				goto(`${base}/`);
			}
		} else {
			chatActions.delete(message);
		}

		showDeleteDialog = false;
	}

	async function handleDelete() {
		deletionInfo = await chatStore.getDeletionInfo(message.id);
		showDeleteDialog = true;
	}

	function handleEdit() {
		isEditing = true;
		// Clear temporary placeholder content for system messages
		editedContent =
			message.role === MessageRole.SYSTEM && message.content === SYSTEM_MESSAGE_PLACEHOLDER
				? ''
				: message.content;
		textareaElement?.focus();
		editedExtras = message.extra ? [...message.extra] : [];
		editedUploadedFiles = [];

		setTimeout(() => {
			if (textareaElement) {
				textareaElement.focus();
				textareaElement.setSelectionRange(
					textareaElement.value.length,
					textareaElement.value.length
				);
			}
		}, 0);
	}

	function handleRegenerate(modelOverride?: string) {
		chatActions.regenerateWithBranching(message, modelOverride);
	}

	function handleContinue() {
		chatActions.continueAssistantMessage(message);
	}

	function handleNavigateToSibling(siblingId: string) {
		chatActions.navigateToSibling(siblingId);
	}

	async function handleSaveEdit() {
		if (message.role === MessageRole.SYSTEM) {
			// System messages: update in place without branching
			const newContent = editedContent.trim();

			// If content is empty, remove without deleting children
			if (!newContent) {
				const conversationDeleted = await chatStore.removeSystemPromptPlaceholder(message.id);
				isEditing = false;
				if (conversationDeleted) {
					goto(`${base}/`);
				}
				return;
			}

			await DatabaseService.updateMessage(message.id, { content: newContent });
			const index = conversationsStore.findMessageIndex(message.id);
			if (index !== -1) {
				conversationsStore.updateMessageAtIndex(index, { content: newContent });
			}
		} else if (message.role === MessageRole.USER) {
			const finalExtras = await getMergedExtras();
			chatActions.editWithBranching(message, editedContent.trim(), finalExtras);
		} else {
			// For assistant messages, preserve exact content including trailing whitespace
			// This is important for the Continue feature to work properly
			chatActions.editWithReplacement(message, editedContent, shouldBranchAfterEdit);
		}

		isEditing = false;
		shouldBranchAfterEdit = false;
		editedUploadedFiles = [];
	}

	async function handleSaveEditOnly() {
		if (message.role === MessageRole.USER) {
			// For user messages, trim to avoid accidental whitespace
			const finalExtras = await getMergedExtras();
			chatActions.editUserMessagePreserveResponses(message, editedContent.trim(), finalExtras);
		}

		isEditing = false;
		editedUploadedFiles = [];
	}

	async function getMergedExtras(): Promise<DatabaseMessageExtra[]> {
		if (editedUploadedFiles.length === 0) {
			return editedExtras;
		}

		const plainFiles = $state.snapshot(editedUploadedFiles);
		const result = await parseFilesToMessageExtras(plainFiles);
		const newExtras = result?.extras || [];

		return [...editedExtras, ...newExtras];
	}

	function handleShowDeleteDialogChange(show: boolean) {
		showDeleteDialog = show;
	}
</script>

{#if message.role === MessageRole.SYSTEM}
	<ChatMessageSystem
		bind:textareaElement
		class={className}
		{deletionInfo}
		{message}
		onConfirmDelete={handleConfirmDelete}
		onCopy={handleCopy}
		onDelete={handleDelete}
		onEdit={handleEdit}
		onNavigateToSibling={handleNavigateToSibling}
		onShowDeleteDialogChange={handleShowDeleteDialogChange}
		{showDeleteDialog}
		{siblingInfo}
	/>
{:else if message.role === MessageRole.USER}
	<ChatMessageUser
		class={className}
		{deletionInfo}
		{message}
		onConfirmDelete={handleConfirmDelete}
		onCopy={handleCopy}
		onDelete={handleDelete}
		onEdit={handleEdit}
		onNavigateToSibling={handleNavigateToSibling}
		onShowDeleteDialogChange={handleShowDeleteDialogChange}
		{showDeleteDialog}
		{siblingInfo}
	/>
{:else}
	<ChatMessageAssistant
		bind:textareaElement
		class={className}
		{deletionInfo}
		{isLastAssistantMessage}
		{message}
		messageContent={message.content}
		onConfirmDelete={handleConfirmDelete}
		onContinue={handleContinue}
		onCopy={handleCopy}
		onDelete={handleDelete}
		onEdit={handleEdit}
		onNavigateToSibling={handleNavigateToSibling}
		onRegenerate={handleRegenerate}
		onShowDeleteDialogChange={handleShowDeleteDialogChange}
		{showDeleteDialog}
		{siblingInfo}
	/>
{/if}
