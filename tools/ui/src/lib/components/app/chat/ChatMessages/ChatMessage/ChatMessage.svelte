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
	import { DialogPromptAddNew, DialogPromptSync } from '$lib/components/app';
	import { parseFilesToMessageExtras } from '$lib/utils/browser-only';
	import { deriveAgenticSections, hasContentDiff } from '$lib/utils';
	import { promptsStore } from '$lib/stores/prompts.svelte';
	import type {
		DatabaseMessageExtraMcpPrompt,
		DatabaseMessageExtraCustomInstruction
	} from '$lib/types';
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
	let editedContent = $state(message.content);

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
	let customInstructionExtra = $derived.by(() => {
		if (message.role !== MessageRole.SYSTEM) return null;
		if (!message.extra || message.extra.length === 0) return null;
		for (const extra of message.extra) {
			if (extra.type === AttachmentType.CUSTOM_INSTRUCTION) {
				return extra as DatabaseMessageExtraCustomInstruction;
			}
		}
		return null;
	});
	let isEditing = $state(false);
	let showDeleteDialog = $state(false);
	let shouldBranchAfterEdit = $state(false);
	let promptDialogOpen = $state(false);

	// Pull the referenced library prompt (if any). A null prompt means the
	// CUSTOM_INSTRUCTION points at a prompt the user has since deleted.
	let referencedPrompt = $derived(
		customInstructionExtra
			? promptsStore.getPrompt(customInstructionExtra.instructionId)
			: undefined
	);
	let promptIsStale = $derived.by(() => {
		if (!customInstructionExtra || !referencedPrompt) return false;
		return hasContentDiff(message.content, referencedPrompt.content);
	});
	let showPromptSyncDialog = $state(false);

	// System-message affordances surfaced to the edit form.
	let isSystemMessage = $derived(message.role === MessageRole.SYSTEM);
	let isSystemPlaceholder = $derived(
		isSystemMessage && message.content === SYSTEM_MESSAGE_PLACEHOLDER
	);
	let canAddToLibrary = $derived.by(() => {
		if (message.role !== MessageRole.SYSTEM) return false;
		const custom = customInstructionExtra;
		if (!custom) return true;
		if (custom.instructionId.startsWith('mcp:')) return false;
		return !referencedPrompt;
	});
	let canUpdateLibraryPrompt = $derived.by(() => {
		if (message.role !== MessageRole.SYSTEM) return false;
		const custom = customInstructionExtra;
		if (!custom) return false;
		if (custom.instructionId.startsWith('mcp:')) return false;
		return !!referencedPrompt;
	});

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
		get isSystemMessage() {
			return isSystemMessage;
		},
		get isSystemPlaceholder() {
			return isSystemPlaceholder;
		},
		get canAddToLibrary() {
			return canAddToLibrary;
		},
		get canUpdateLibraryPrompt() {
			return canUpdateLibraryPrompt;
		},
		get libraryPromptTitle() {
			return referencedPrompt?.title;
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
		saveAsLibrary: handleSaveAsLibrary,
		updateLibraryPrompt: handleUpdateLibraryPrompt,
		cancel: handleCancelEdit,
		startEdit: handleEdit
	});

	let mcpPromptExtra = $derived.by((): DatabaseMessageExtraMcpPrompt | null => {
		if (message.role === MessageRole.USER) {
			if (message.content.trim()) return null;
			if (!message.extra || message.extra.length !== 1) return null;

			const extra = message.extra[0];

			if (extra.type === AttachmentType.MCP_PROMPT) {
				return extra as DatabaseMessageExtraMcpPrompt;
			}

			return null;
		}

		if (message.role === MessageRole.SYSTEM) {
			// System messages created from an MCP prompt carry the server/prompt
			// identity through their synthetic `mcp:<server>:<prompt>` instructionId.
			// Treat them as MCP-prompt messages so they render with the same styled
			// surface instead of a plain system textarea.
			const custom = customInstructionExtra;

			if (!custom?.instructionId.startsWith('mcp:')) return null;

			const stripped = custom.instructionId.slice('mcp:'.length);
			const sepIndex = stripped.indexOf(':');

			if (sepIndex <= 0) return null;

			const serverName = stripped.slice(0, sepIndex);
			const promptName = stripped.slice(sepIndex + 1);

			return {
				type: AttachmentType.MCP_PROMPT,
				name: promptName,
				serverName,
				promptName,
				content: message.content
			};
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

		editedExtras = message.extra ? [...message.extra] : [];
		editedUploadedFiles = [];
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
			await persistSystemMessageEdit();
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

	async function handleSaveAsLibrary() {
		if (message.role !== MessageRole.SYSTEM) return;

		// Empty content falls back to the normal save flow (which removes the
		// placeholder system message). We can't create a library prompt without
		// content anyway.
		if (!editedContent.trim()) {
			await handleSaveEdit();
			return;
		}

		promptDialogOpen = true;
	}

	async function handleUpdateLibraryPrompt() {
		if (message.role !== MessageRole.SYSTEM) return;
		if (!referencedPrompt) return;

		const newContent = editedContent.trim();

		// Empty content falls back to the normal save flow (which removes the
		// placeholder system message). We can't push empty content to the
		// library prompt either.
		if (!newContent) {
			await handleSaveEdit();
			return;
		}

		await promptsStore.updatePrompt(referencedPrompt.id, { content: newContent });
		await handleSaveEdit();
	}

	async function persistSystemMessageEdit() {
		const newContent = editedContent.trim();

		// If content is empty, remove without deleting children
		if (!newContent) {
			const conversationDeleted = await chatStore.removeSystemPromptPlaceholder(message.id);
			if (conversationDeleted) {
				goto(ROUTES.START);
			}
			return;
		}

		// Preserve existing extras (drop stale CUSTOM_INSTRUCTION so we can re-add cleanly)
		const existingExtras = message.extra || [];
		const extrasToSave = existingExtras.filter(
			(e: DatabaseMessageExtra) => e.type !== AttachmentType.CUSTOM_INSTRUCTION
		);

		if (extrasToSave.length > 0) {
			await DatabaseService.updateMessage(message.id, {
				content: newContent,
				extra: extrasToSave
			});
			const index = conversationsStore.findMessageIndex(message.id);
			if (index !== -1) {
				conversationsStore.updateMessageAtIndex(index, {
					content: newContent,
					extra: extrasToSave
				});
			}
		} else {
			await DatabaseService.updateMessage(message.id, { content: newContent });
			const index = conversationsStore.findMessageIndex(message.id);
			if (index !== -1) {
				conversationsStore.updateMessageAtIndex(index, { content: newContent });
			}
		}
	}

	async function saveSystemPromptWithPrompt(content: string, instructionId: string, title: string) {
		// message.extra is a deep $state proxy; snapshot to plain values so
		// Dexie/IndexedDB can structured-clone it (see getMergedExtras precedent)
		const existingExtras = (
			message.extra ? $state.snapshot(message.extra) : []
		) as DatabaseMessageExtra[];
		const extras: DatabaseMessageExtra[] = [
			...existingExtras.filter(
				(e: DatabaseMessageExtra) => e.type !== AttachmentType.CUSTOM_INSTRUCTION
			),
			{
				type: AttachmentType.CUSTOM_INSTRUCTION,
				instructionId,
				title
			}
		];

		await DatabaseService.updateMessage(message.id, { content, extra: extras });
		const index = conversationsStore.findMessageIndex(message.id);
		if (index !== -1) {
			conversationsStore.updateMessageAtIndex(index, { content, extra: extras });
		}
	}

	// Apply the latest library prompt content to this system message, keeping
	// the CUSTOM_INSTRUCTION extra (title may also have changed in the library)
	async function syncPromptFromLibrary() {
		if (!customInstructionExtra || !referencedPrompt) return;
		const { instructionId } = customInstructionExtra;
		await saveSystemPromptWithPrompt(
			referencedPrompt.content,
			instructionId,
			referencedPrompt.title
		);
		showPromptSyncDialog = false;
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

<div use:fadeInView class="chat-message">
	{#if mcpPromptExtra}
		<ChatMessageMcpPrompt
			class={className}
			{deletionInfo}
			{message}
			role={message.role}
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
	{:else if message.role === MessageRole.SYSTEM}
		<ChatMessageSystem
			class={className}
			{deletionInfo}
			{message}
			instructionId={customInstructionExtra?.instructionId}
			title={customInstructionExtra?.title}
			{promptIsStale}
			onPromptUpdate={() => (showPromptSyncDialog = true)}
			onConfirmDelete={handleConfirmDelete}
			onCopy={handleCopy}
			onDelete={handleDelete}
			onEdit={handleEdit}
			onNavigateToSibling={handleNavigateToSibling}
			onShowDeleteDialogChange={handleShowDeleteDialogChange}
			{showDeleteDialog}
			{siblingInfo}
		/>

		{#if promptDialogOpen}
			<DialogPromptAddNew
				open={promptDialogOpen}
				initialContent={editedContent.trim()}
				onAddToLibraryComplete={(id: string, title: string) => {
					promptDialogOpen = false;
					isEditing = false;
					saveSystemPromptWithPrompt(editedContent.trim(), id, title);
				}}
			/>
		{/if}

		{#if showPromptSyncDialog && referencedPrompt}
			<DialogPromptSync
				bind:open={showPromptSyncDialog}
				promptTitle={referencedPrompt.title}
				currentContent={message.content}
				updatedContent={referencedPrompt.content}
				onUpdate={syncPromptFromLibrary}
			/>
		{/if}
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
