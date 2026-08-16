<script lang="ts">
	import { goto } from '$app/navigation';
	import {
		ChatMessageAssistant,
		ChatMessageMcpPrompt,
		ChatMessageSynthetic,
		ChatMessageSystem,
		ChatMessageUser
	} from '$lib/components/app/chat';
	import { REASONING_TAGS, ROUTES, SYSTEM_MESSAGE_PLACEHOLDER } from '$lib/constants';
	import { setChatMessageActionsContext, setChatMessageEditContext } from '$lib/contexts';
	import { AgenticSectionType, AttachmentType, MessageRole } from '$lib/enums';
	import { DatabaseService } from '$lib/services/database.service';
	import { chatStore, conversationsStore, isMobile } from '$lib/stores';
	import type {
		ChatMessageActions,
		ChatMessageDeletionInfo,
		DatabaseMessageExtraMcpPrompt
	} from '$lib/types';
	import { deriveAgenticSections } from '$lib/utils';
	import { parseFilesToMessageExtras } from '$lib/utils/browser-only';

	interface Props {
		class?: string;
		chatActions: ChatMessageActions;
		message: DatabaseMessage;
		toolMessages?: DatabaseMessage[];
		isLastAssistantMessage?: boolean;
		isLastUserMessage?: boolean;
		nextAssistantMessage?: DatabaseMessage | null;
		siblingInfo?: ChatMessageSiblingInfo | null;
	}

	let {
		chatActions,
		class: className = '',
		isLastAssistantMessage = false,
		isLastUserMessage = false,
		message,
		nextAssistantMessage = null,
		siblingInfo = null,
		toolMessages = []
	}: Props = $props();

	let deletionInfo = $state<ChatMessageDeletionInfo | null>(null);
	// The system message placeholder must never surface as editable content; keeping
	// it in the derived (not just in handleEdit) guards against prop invalidation
	// reverting the override while editing
	let editedContent = $derived(
		message.role === MessageRole.SYSTEM && message.content === SYSTEM_MESSAGE_PLACEHOLDER
			? ''
			: message.content
	);

	// Synthetic cwd-change messages render with the folder-row UI instead
	// of a user bubble. The persisted flag is the single source of truth.
	let isSynthetic = $derived(Boolean(message.isSynthetic));

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

	setChatMessageEditContext({
		cancel: handleCancelEdit,
		get editedContent() {
			return editedContent;
		},
		get editedExtras() {
			return editedExtras;
		},
		get editedUploadedFiles() {
			return editedUploadedFiles;
		},
		get isEditing() {
			return isEditing;
		},
		get messageRole() {
			return message.role;
		},
		get originalContent() {
			return message.role === MessageRole.ASSISTANT
				? (rawEditContent ?? message.content)
				: message.content;
		},
		get originalExtras() {
			return message.extra || [];
		},
		get rawEditContent() {
			return rawEditContent;
		},
		save: handleSaveEdit,
		saveOnly: handleSaveEditOnly,
		setContent: (content: string) => {
			editedContent = content;
		},
		setExtras: (extras: DatabaseMessageExtra[]) => {
			editedExtras = extras;
		},
		setShouldBranchAfterEdit: (value: boolean) => {
			shouldBranchAfterEdit = value;
		},
		setUploadedFiles: (files: ChatUploadedFile[]) => {
			editedUploadedFiles = files;
		},
		get shouldBranchAfterEdit() {
			return shouldBranchAfterEdit;
		},
		get showBranchAfterEditOption() {
			return showBranchAfterEditOption;
		},
		get showSaveOnlyOption() {
			return showSaveOnlyOption;
		},
		startEdit: handleEdit
	});

	setChatMessageActionsContext({
		confirmDelete: handleConfirmDelete,
		copy: handleCopy,
		get deletionInfo() {
			return deletionInfo;
		},
		get forkConversation() {
			const isForkableUser = message.role === MessageRole.USER && !mcpPromptExtra;

			return isForkableUser || message.role === MessageRole.ASSISTANT
				? handleForkConversation
				: undefined;
		},
		navigateToSibling: handleNavigateToSibling,
		requestDelete: handleDelete,
		setShowDeleteDialog: handleShowDeleteDialogChange,
		get showDeleteDialog() {
			return showDeleteDialog;
		},
		get siblingInfo() {
			return siblingInfo;
		}
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
		const pendingId = chatStore.pendingEditMessageId;

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

		textareaElement?.focus({ preventScroll: true });
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

	// After the system message flow ends, hand focus to the main chat form
	function focusMainChatForm() {
		if (isMobile.current) return;

		document.querySelector<HTMLTextAreaElement>('.chat-screen-form-wrapper textarea')?.focus();
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
				} else {
					focusMainChatForm();
				}

				return;
			}

			await DatabaseService.updateMessage(message.id, { content: newContent });
			const index = conversationsStore.findMessageIndex(message.id);

			if (index !== -1) {
				conversationsStore.updateMessageAtIndex(index, { content: newContent });
			}

			focusMainChatForm();
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

<div class="chat-message" class:chat-message--synthetic={isSynthetic}>
	{#if message.role === MessageRole.SYSTEM}
		<ChatMessageSystem bind:textareaElement class={className} {message} />
	{:else if mcpPromptExtra}
		<ChatMessageMcpPrompt class={className} {message} mcpPrompt={mcpPromptExtra} />
	{:else if isSynthetic}
		<ChatMessageSynthetic {message} class={className} />
	{:else if message.role === MessageRole.USER}
		<ChatMessageUser class={className} {isLastUserMessage} {message} {nextAssistantMessage} />
	{:else}
		<ChatMessageAssistant
			bind:textareaElement
			class={className}
			{isLastAssistantMessage}
			{message}
			{toolMessages}
			onContinue={handleContinue}
			onRegenerate={handleRegenerate}
		/>
	{/if}
</div>

<style>
	/*
	 * The browser skips layout and paint for messages outside the
	 * viewport. contain-intrinsic-size reuses the last rendered size
	 * once known; 500px sizes messages that have never been rendered.
	 */
	.chat-message {
		--chat-message-intrinsic-size: 500px;
		content-visibility: auto;
		contain-intrinsic-size: auto var(--chat-message-intrinsic-size);
	}

	/*
	 * Synthetic rows (e.g. the working-directory change) are small, so an
	 * accurate placeholder keeps the injected row from inflating the
	 * auto-scroll offset; the 500px default is for ordinary bubbles.
	 */
	.chat-message--synthetic {
		--chat-message-intrinsic-size: 40px;
	}
</style>
