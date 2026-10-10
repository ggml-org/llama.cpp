<script lang="ts">
	import { goto } from '$app/navigation';
	import {
		ChatMessageAssistant,
		ChatMessageMcpPrompt,
		ChatMessageSynthetic,
		ChatMessageSystem,
		ChatMessageUser
	} from '$lib/components/app/chat';
	import { AGENTIC_TEXT_COPY_SEPARATOR, ROUTES, SYSTEM_MESSAGE_PLACEHOLDER } from '$lib/constants';
	import { setChatMessageActionsContext, setChatMessageEditContext } from '$lib/contexts';
	import { AgenticSectionType, AttachmentType, MessageRole } from '$lib/enums';
	import { DatabaseService } from '$lib/services/database.service';
	import { chatStore, conversationsStore, deviceStore } from '$lib/stores';
	import type {
		ChatMessageActions,
		ChatMessageDeletionInfo,
		DatabaseMessageExtraMcpPrompt
	} from '$lib/types';
	import { deriveAgenticSections, hasAgenticContent } from '$lib/utils';
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
	// The edit buffer is plain state seeded by handleEdit. Deriving it from the message
	// would tie it to a value the store rewrites at will, discarding what the user types.
	let editedContent = $state('');
	let editedExtras = $state<DatabaseMessageExtra[]>([]);
	let editedReasoning = $state('');
	let editedUploadedFiles = $state<ChatUploadedFile[]>([]);
	let isEditing = $state(false);
	let showDeleteDialog = $state(false);
	let shouldBranchAfterEdit = $state(false);
	let textareaElement: HTMLTextAreaElement | undefined = $state();

	// Synthetic cwd-change messages render with the folder-row UI instead
	// of a user bubble. The persisted flag is the single source of truth.
	let isSynthetic = $derived(Boolean(message.isSynthetic));

	let canEdit = $derived(!hasAgenticContent(message, toolMessages));

	let showSaveOnlyOption = $derived(message.role === MessageRole.USER);
	let showBranchAfterEditOption = $derived(message.role === MessageRole.ASSISTANT);
	// Tool calls and tool results live in their own fields and rows, so the edit form
	// exposes exactly the two free text fields of an assistant turn: content and reasoning
	let showReasoningField = $derived(
		message.role === MessageRole.ASSISTANT && Boolean(message.reasoningContent)
	);

	setChatMessageEditContext({
		cancel: handleCancelEdit,
		get canEdit() {
			return canEdit;
		},
		get editedContent() {
			return editedContent;
		},
		get editedExtras() {
			return editedExtras;
		},
		get editedReasoning() {
			return editedReasoning;
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
			return message.content;
		},
		get originalExtras() {
			return message.extra || [];
		},
		get originalReasoning() {
			return message.reasoningContent ?? '';
		},
		save: handleSaveEdit,
		saveOnly: handleSaveEditOnly,
		setContent: (content: string) => {
			editedContent = content;
		},
		setExtras: (extras: DatabaseMessageExtra[]) => {
			editedExtras = extras;
		},
		setReasoning: (reasoning: string) => {
			editedReasoning = reasoning;
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
		get showReasoningField() {
			return showReasoningField;
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

		if (pendingId && pendingId === message.id && !isEditing && canEdit) {
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
		}
	}

	function handleCopy() {
		// Agentic sessions render as a single entry anchored on the first assistant
		// turn, whose own content is typically just the first tool call. Copy the
		// text sections of the whole session so the clipboard matches the visible
		// response instead of the anchor turn.
		if (message.role === MessageRole.ASSISTANT) {
			const sections = deriveAgenticSections(message, toolMessages, [], false);
			const text = sections
				.filter((section) => section.type === AgenticSectionType.TEXT)
				.map((section) => section.content)
				.join(AGENTIC_TEXT_COPY_SEPARATOR);

			if (text) {
				chatActions.copy(message, text);

				return;
			}
		}

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

		// The system placeholder is a marker, never content the user should see
		const isSystemPlaceholder =
			message.role === MessageRole.SYSTEM && message.content === SYSTEM_MESSAGE_PLACEHOLDER;

		editedContent = isSystemPlaceholder ? '' : message.content;
		editedReasoning = message.reasoningContent ?? '';

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
		// A grouped agentic turn keeps its continuations in toolMessages, and the group
		// renders under the anchor. Continue resumes from the last assistant of the turn,
		// otherwise it resumes from the anchor and drops every turn generated after it.
		const lastAssistant = toolMessages.findLast((m) => m.role === MessageRole.ASSISTANT);

		chatActions.continueAssistantMessage(lastAssistant ?? message);
	}

	function handleForkConversation(options: { name: string; includeAttachments: boolean }) {
		chatActions.forkConversation(message, options);
	}

	function handleNavigateToSibling(siblingId: string) {
		chatActions.navigateToSibling(siblingId);
	}

	// After the system message flow ends, hand focus to the main chat form
	function focusMainChatForm() {
		if (deviceStore.isMobile) return;

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
			// Assistant content and reasoning go back untrimmed, trailing whitespace included,
			// so Continue resumes on the exact byte the model stopped at
			chatActions.editWithReplacement(
				message,
				editedContent,
				editedReasoning,
				shouldBranchAfterEdit
			);
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

<div>
	{#if message.role === MessageRole.SYSTEM}
		<ChatMessageSystem bind:textareaElement class={className} {message} />
	{:else if mcpPromptExtra}
		<ChatMessageMcpPrompt class={className} mcpPrompt={mcpPromptExtra} {message} />
	{:else if isSynthetic}
		<ChatMessageSynthetic class={className} {message} />
	{:else if message.role === MessageRole.USER}
		<ChatMessageUser class={className} {isLastUserMessage} {message} {nextAssistantMessage} />
	{:else}
		<ChatMessageAssistant
			class={className}
			{isLastAssistantMessage}
			{message}
			onContinue={handleContinue}
			onRegenerate={handleRegenerate}
			{toolMessages}
		/>
	{/if}
</div>
