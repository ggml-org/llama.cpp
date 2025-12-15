<script lang="ts">
	import { goto } from '$app/navigation';
	import { base } from '$app/paths';
	import {
		chatStore,
		pendingEditMessageId,
		clearPendingEditMessageId,
		removeSystemPromptPlaceholder
	} from '$lib/stores/chat.svelte';
	import { getChatActionsContext, setMessageEditContext } from '$lib/contexts';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { DatabaseService } from '$lib/services';
	import { config } from '$lib/stores/settings.svelte';
	import { SYSTEM_MESSAGE_PLACEHOLDER } from '$lib/constants/ui';
	import { copyToClipboard, isIMEComposing, formatMessageForClipboard } from '$lib/utils';
	import ChatMessageAssistant from './ChatMessageAssistant.svelte';
	import ChatMessageUser from './ChatMessageUser.svelte';
	import ChatMessageSystem from './ChatMessageSystem.svelte';

	interface Props {
		class?: string;
		message: DatabaseMessage;
		siblingInfo?: ChatMessageSiblingInfo | null;
		toolParentIds?: string[];
	}

	let {
		class: className = '',
		message,
		siblingInfo = null,
		toolParentIds
	}: Props = $props();

	type MessageWithToolExtras = DatabaseMessage & {
		_actionTargetId?: string;
		_toolMessagesCollected?: { toolCallId?: string | null; parsed: unknown }[];
	};

	const actionTargetId = $derived((message as MessageWithToolExtras)._actionTargetId ?? message.id);

	function getActionTarget(): DatabaseMessage {
		return conversationsStore.activeMessages.find((m) => m.id === actionTargetId) ?? message;
	}

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
	let showSaveOnlyOption = $derived(message.role === 'user');

	let thinkingContent = $derived.by(() => {
		if (message.role === 'assistant') {
			const trimmedThinking = message.thinking?.trim();

			return trimmedThinking ? trimmedThinking : null;
		}
		return null;
	});

	let toolCallContent = $derived.by((): ApiChatCompletionToolCall[] | string | null => {
		if (message.role === 'assistant') {
			const trimmedToolCalls = message.toolCalls?.trim();

			if (!trimmedToolCalls) {
				return null;
			}

			try {
				const parsed = JSON.parse(trimmedToolCalls);

				if (Array.isArray(parsed)) {
					return parsed as ApiChatCompletionToolCall[];
				}
			} catch {
				// Harmony-only path: fall back to the raw string so issues surface visibly.
			}

			return trimmedToolCalls;
		}
		return null;
	});

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

	// Auto-start edit mode if this message is the pending edit target
	$effect(() => {
		const pendingId = pendingEditMessageId();

		if (pendingId && pendingId === message.id && !isEditing) {
			handleEdit();
			clearPendingEditMessageId();
		}
	});

	async function handleCancelEdit() {
		isEditing = false;

		// If canceling a new system message with placeholder content, remove it without deleting children
		if (message.role === 'system') {
			const conversationDeleted = await removeSystemPromptPlaceholder(message.id);

			if (conversationDeleted) {
				goto(`${base}/`);
			}

			return;
		}

		editedContent = message.content;
		editedExtras = message.extra ? [...message.extra] : [];
		editedUploadedFiles = [];
	}

	function handleEditedExtrasChange(extras: DatabaseMessageExtra[]) {
		editedExtras = extras;
	}

	function handleEditedUploadedFilesChange(files: ChatUploadedFile[]) {
		editedUploadedFiles = files;
	}

	async function handleCopy() {
		const asPlainText = Boolean(config().copyTextAttachmentsAsPlainText);
		const clipboardContent = formatMessageForClipboard(message.content, message.extra, asPlainText);
		await copyToClipboard(clipboardContent, 'Message copied to clipboard');
		chatActions.copy(message);
	}

	async function handleConfirmDelete() {
		const target = getActionTarget();
		if (target.role === 'system') {
			const conversationDeleted = await removeSystemPromptPlaceholder(target.id);

			if (conversationDeleted) {
				goto('/');
			}
		} else {
			chatActions.delete(target);
		}
		showDeleteDialog = false;
	}

	async function handleDelete() {
		const target = getActionTarget();
		deletionInfo = await chatStore.getDeletionInfo(target.id);
		showDeleteDialog = true;
	}

	function handleEdit() {
		isEditing = true;
		// Clear placeholder content for system messages
		editedContent =
			message.role === 'system' && message.content === SYSTEM_MESSAGE_PLACEHOLDER
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

	function handleEditedContentChange(content: string) {
		editedContent = content;
	}

	function handleEditKeydown(event: KeyboardEvent) {
		// Check for IME composition using isComposing property and keyCode 229 (specifically for IME composition on Safari)
		// This prevents saving edit when confirming IME word selection (e.g., Japanese/Chinese input)
		if (event.key === 'Enter' && !event.shiftKey && !isIMEComposing(event)) {
			event.preventDefault();
			handleSaveEdit();
		} else if (event.key === 'Escape') {
			event.preventDefault();
			handleCancelEdit();
		}
	}

	function handleRegenerate(modelOverride?: string) {
		const target = getActionTarget();
		chatActions.regenerateWithBranching(target, modelOverride);
	}

	function handleContinue() {
		const target = getActionTarget();
		chatActions.continueAssistantMessage(target);
	}

	function handleNavigateToSibling(siblingId: string) {
		chatActions.navigateToSibling(siblingId);
	}

	async function handleSaveEdit() {
		if (message.role === 'system') {
			// System messages: update in place without branching
			const newContent = editedContent.trim();

			// If content is empty or still the placeholder, remove without deleting children
			if (!newContent) {
				const conversationDeleted = await removeSystemPromptPlaceholder(message.id);
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
		} else if (message.role === 'user') {
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
		if (message.role === 'user') {
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

		const { parseFilesToMessageExtras } = await import('$lib/utils/browser-only');
		const result = await parseFilesToMessageExtras(editedUploadedFiles);
		const newExtras = result?.extras || [];

		return [...editedExtras, ...newExtras];
	}

	function handleShowDeleteDialogChange(show: boolean) {
		showDeleteDialog = show;
	}
</script>

{#if message.role === 'system'}
	<ChatMessageSystem
		bind:textareaElement
		class={className}
		{deletionInfo}
		{editedContent}
		{isEditing}
		{message}
		onCancelEdit={handleCancelEdit}
		onConfirmDelete={handleConfirmDelete}
		onCopy={handleCopy}
		onDelete={handleDelete}
		onEdit={handleEdit}
		onEditKeydown={handleEditKeydown}
		onEditedContentChange={handleEditedContentChange}
		onNavigateToSibling={handleNavigateToSibling}
		onSaveEdit={handleSaveEdit}
		onShowDeleteDialogChange={handleShowDeleteDialogChange}
		{showDeleteDialog}
		{siblingInfo}
	/>
{:else if message.role === 'user'}
	<ChatMessageUser
		bind:textareaElement
		class={className}
		{deletionInfo}
		{editedContent}
		{editedExtras}
		{editedUploadedFiles}
		{isEditing}
		{message}
		onCancelEdit={handleCancelEdit}
		onConfirmDelete={handleConfirmDelete}
		onCopy={handleCopy}
		onDelete={handleDelete}
		onEdit={handleEdit}
		onEditKeydown={handleEditKeydown}
		onEditedContentChange={handleEditedContentChange}
		onEditedExtrasChange={handleEditedExtrasChange}
		onEditedUploadedFilesChange={handleEditedUploadedFilesChange}
		onNavigateToSibling={handleNavigateToSibling}
		onSaveEdit={handleSaveEdit}
		onSaveEditOnly={handleSaveEditOnly}
		onShowDeleteDialogChange={handleShowDeleteDialogChange}
		{showDeleteDialog}
		{siblingInfo}
	/>
{:else if message.role === 'assistant'}
	<ChatMessageAssistant
		bind:textareaElement
		class={className}
		{deletionInfo}
		{editedContent}
		{isEditing}
		{message}
		messageContent={message.content}
		onCancelEdit={handleCancelEdit}
		onConfirmDelete={handleConfirmDelete}
		onContinue={handleContinue}
		onCopy={handleCopy}
		onDelete={handleDelete}
		onEdit={handleEdit}
		onEditKeydown={handleEditKeydown}
		onEditedContentChange={handleEditedContentChange}
		onNavigateToSibling={handleNavigateToSibling}
		onRegenerate={handleRegenerate}
		onSaveEdit={handleSaveEdit}
		onShowDeleteDialogChange={handleShowDeleteDialogChange}
		{shouldBranchAfterEdit}
		onShouldBranchAfterEditChange={(value) => (shouldBranchAfterEdit = value)}
		{showDeleteDialog}
		{siblingInfo}
		{thinkingContent}
		{toolCallContent}
		toolParentIds={toolParentIds ?? [message.id]}
		toolMessagesCollected={(message as MessageWithToolExtras)._toolMessagesCollected}
	/>
{:else if message.role === 'tool'}
	<!-- Tool messages are rendered inline inside their parent assistant's reasoning block.
	     Skip standalone rendering to avoid duplicate bubbles. -->
	<!-- Intentionally left blank -->
{/if}
