<script lang="ts">
	import { getDeletionInfo } from '$lib/stores/chat.svelte';
	import { copyToClipboard } from '$lib/utils/copy';
	import { parseThinkingContent } from '$lib/utils/thinking';
	import ChatMessageAssistant from './ChatMessageAssistant.svelte';
	import ChatMessageUser from './ChatMessageUser.svelte';

	interface Props {
		class?: string;
		message: DatabaseMessage;
		onCopy?: (message: DatabaseMessage) => void;
		onDelete?: (message: DatabaseMessage) => void;
		onEditWithBranching?: (message: DatabaseMessage, newContent: string) => void;
		onEditWithReplacement?: (
			message: DatabaseMessage,
			newContent: string,
			shouldBranch: boolean
		) => void;
		onNavigateToSibling?: (siblingId: string) => void;
		onRegenerateWithBranching?: (message: DatabaseMessage) => void;
		siblingInfo?: ChatMessageSiblingInfo | null;
	}

	let {
		class: className = '',
		message,
		onCopy,
		onDelete,
		onEditWithBranching,
		onEditWithReplacement,
		onNavigateToSibling,
		onRegenerateWithBranching,
		siblingInfo = null
	}: Props = $props();

	let deletionInfo = $state<{
		totalCount: number;
		userMessages: number;
		assistantMessages: number;
		messageTypes: string[];
	} | null>(null);
	let editedContent = $state(message.content);
	let isEditing = $state(false);
	let showDeleteDialog = $state(false);
	let shouldBranchAfterEdit = $state(false);
	let textareaElement: HTMLTextAreaElement | undefined = $state();

	let thinkingContent = $derived.by(() => {
		if (message.role === 'assistant') {
			if (message.thinking) {
				return message.thinking;
			}

			const parsed = parseThinkingContent(message.content);

			return parsed.thinking;
		}
		return null;
	});

	let messageContent = $derived.by(() => {
		if (message.role === 'assistant') {
			const parsed = parseThinkingContent(message.content);
			return parsed.cleanContent?.replace('<|channel|>analysis', '');
		}

		return message.content?.replace('<|channel|>analysis', '');
	});

	function handleCancelEdit() {
		isEditing = false;
		editedContent = message.content;
	}

	async function handleCopy() {
		await copyToClipboard(message.content, 'Message copied to clipboard');
		onCopy?.(message);
	}

	function handleConfirmDelete() {
		onDelete?.(message);
		showDeleteDialog = false;
	}

	async function handleDelete() {
		deletionInfo = await getDeletionInfo(message.id);
		showDeleteDialog = true;
	}

	function handleEdit() {
		isEditing = true;
		editedContent = message.content;

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
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			handleSaveEdit();
		} else if (event.key === 'Escape') {
			event.preventDefault();
			handleCancelEdit();
		}
	}

	function handleRegenerate() {
		onRegenerateWithBranching?.(message);
	}

	function handleSaveEdit() {
		if (message.role === 'user') {
			onEditWithBranching?.(message, editedContent.trim());
		} else {
			onEditWithReplacement?.(message, editedContent.trim(), shouldBranchAfterEdit);
		}

		isEditing = false;
		shouldBranchAfterEdit = false;
	}

	function handleShowDeleteDialogChange(show: boolean) {
		showDeleteDialog = show;
	}
</script>

{#if message.role === 'user'}
	<ChatMessageUser
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
		{onNavigateToSibling}
		onSaveEdit={handleSaveEdit}
		onShowDeleteDialogChange={handleShowDeleteDialogChange}
		{showDeleteDialog}
		{siblingInfo}
	/>
{:else}
	<ChatMessageAssistant
		bind:textareaElement
		class={className}
		{deletionInfo}
		{editedContent}
		{isEditing}
		{message}
		{messageContent}
		onCancelEdit={handleCancelEdit}
		onConfirmDelete={handleConfirmDelete}
		onCopy={handleCopy}
		onDelete={handleDelete}
		onEdit={handleEdit}
		onEditKeydown={handleEditKeydown}
		onEditedContentChange={handleEditedContentChange}
		{onNavigateToSibling}
		onRegenerate={handleRegenerate}
		onSaveEdit={handleSaveEdit}
		onShowDeleteDialogChange={handleShowDeleteDialogChange}
		{shouldBranchAfterEdit}
		onShouldBranchAfterEditChange={(value) => (shouldBranchAfterEdit = value)}
		{showDeleteDialog}
		{siblingInfo}
		{thinkingContent}
	/>
{/if}
