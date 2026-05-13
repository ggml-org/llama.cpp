<script lang="ts">
	import { goto } from '$app/navigation';
	import { getChatActionsContext, setMessageEditContext } from '$lib/contexts';
	import { chatStore, pendingEditMessageId } from '$lib/stores/chat.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { DatabaseService } from '$lib/services/database.service';
	import { SYSTEM_MESSAGE_PLACEHOLDER } from '$lib/constants';
	import { REASONING_TAGS } from '$lib/constants/agentic';
	import { MessageRole, AttachmentType, AgenticSectionType } from '$lib/enums';
	import { fadeInView } from '$lib/actions/fade-in-view.svelte';
	import {
		ChatMessageAssistant,
		ChatMessageUser,
		ChatMessageSystem,
		ChatMessageMcpPrompt
	} from '$lib/components/app/chat';
	import { parseFilesToMessageExtras } from '$lib/utils/browser-only';
	import { deriveAgenticSections } from '$lib/utils';
	import type { DatabaseMessageExtraMcpPrompt } from '$lib/types';
	import { ROUTES } from '$lib/constants/routes';

	interface Props {
		class?: string;
		message: DatabaseMessage;
		toolMessages?: DatabaseMessage[];
		isLastAssistantMessage?: boolean;
		siblingInfo?: ChatMessageSiblingInfo | null;
	}

	let {
		class: className = '',
		message,
		toolMessages = [],
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

	let rawEditContent = $derived.by(() => {
		if (message.role !== MessageRole.ASSISTANT) return undefined;

		const sections = deriveAgenticSections(message, toolMessages, [], false);
		const parts: string[] = [];

		for (const section of sections) {
			switch (section.type) {
				case AgenticSectionType.REASONING:
				case AgenticSectionType.REASONING_PENDING:
					parts.push(`${REASONING_TAGS.START}\n${section.content}\n${REASONING_TAGS.END}`);
					break;

				case AgenticSectionType.TEXT:
					parts.push(section.content);
					break;

				case AgenticSectionType.TOOL_CALL:
				case AgenticSectionType.TOOL_CALL_PENDING:
				case AgenticSectionType.TOOL_CALL_STREAMING: {
					const callObj: Record<string, unknown> = { name: section.toolName };

					if (section.toolArgs) {
						try {
							callObj.arguments = JSON.parse(section.toolArgs);
						} catch {
							callObj.arguments = section.toolArgs;
						}
					}

					parts.push(JSON.stringify(callObj, null, 2));

					if (section.toolResult) {
						parts.push(`[Tool Result]\n${section.toolResult}`);
					}

					break;
				}
			}
		}

		return parts.join('\n\n\n');
	});
	let editedExtras = $derived<DatabaseMessageExtra[]>(message.extra ? [...message.extra] : []);
	let editedUploadedFiles = $state<ChatUploadedFile[]>([]);
	let isEditing = $state(false);
	let showDeleteDialog = $state(false);
	let shouldBranchAfterEdit = $state(false);
	let textareaElement: HTMLTextAreaElement | undefined = $state();

	let showSaveOnlyOption = $derived(message.role === MessageRole.USER);
	let showBranchAfterEditOption = $derived(message.role === MessageRole.ASSISTANT);

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
			return message.role === MessageRole.ASSISTANT
				? (rawEditContent ?? message.content)
				: message.content;
		},
		get originalExtras() {
			return message.extra || [];
		},
		get showSaveOnlyOption() {
			return showSaveOnlyOption;
		},
		get showBranchAfterEditOption() {
			return showBranchAfterEditOption;
		},
		get shouldBranchAfterEdit() {
			return shouldBranchAfterEdit;
		},
		get messageRole() {
			return message.role;
		},
		get rawEditContent() {
			return rawEditContent;
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
		setShouldBranchAfterEdit: (value: boolean) => {
			shouldBranchAfterEdit = value;
		},
		save: handleSaveEdit,
		saveOnly: handleSaveEditOnly,
		cancel: handleCancelEdit,
		startEdit: handleEdit
	});

	let mcpPromptExtra = $derived.by(() => {
		if (message.role !== MessageRole.USER) return null;
		if (message.content.trim()) return null;
		if (!message.extra || message.extra.length !== 1) return null;

		const extra = message.extra[0];

		if (extra.type === AttachmentType.MCP_PROMPT) {
			return extra as DatabaseMessageExtraMcpPrompt;
		}

		return null;
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
		if (message.role === MessageRole.SYSTEM && message.content === SYSTEM_MESSAGE_PLACEHOLDER) {
			const conversationDeleted = await chatStore.removeSystemPromptPlaceholder(message.id);

			if (conversationDeleted) {
				goto(ROUTES.START);
			}

			return;
		}

		editedContent =
			message.role === MessageRole.ASSISTANT
				? rawEditContent || message.content || ''
				: message.content;
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
				goto(ROUTES.START);
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
		if (message.role === MessageRole.SYSTEM && message.content === SYSTEM_MESSAGE_PLACEHOLDER) {
			editedContent = '';
		} else if (message.role === MessageRole.ASSISTANT) {
			editedContent = rawEditContent || message.content || '';
		} else {
			editedContent = message.content;
		}

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

	function handleForkConversation(options: { name: string; includeAttachments: boolean }) {
		chatActions.forkConversation(message, options);
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
					goto(ROUTES.START);
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

<div use:fadeInView>
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
	{:else if mcpPromptExtra}
		<ChatMessageMcpPrompt
			class={className}
			{deletionInfo}
			{message}
			mcpPrompt={mcpPromptExtra}
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
			onForkConversation={handleForkConversation}
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
			{toolMessages}
			messageContent={message.content}
			onConfirmDelete={handleConfirmDelete}
			onContinue={handleContinue}
			onCopy={handleCopy}
			onDelete={handleDelete}
			onEdit={handleEdit}
			onForkConversation={handleForkConversation}
			onNavigateToSibling={handleNavigateToSibling}
			onRegenerate={handleRegenerate}
			onShowDeleteDialogChange={handleShowDeleteDialogChange}
			{showDeleteDialog}
			{siblingInfo}
		/>
	{/if}
</div>
