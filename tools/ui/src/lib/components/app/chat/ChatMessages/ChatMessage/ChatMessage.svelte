<script lang="ts">
	import { goto } from '$app/navigation';
	import { getChatActionsContext, setMessageEditContext } from '$lib/contexts';
	import { chatStore, pendingEditMessageId } from '$lib/stores/chat.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { DatabaseService } from '$lib/services/database.service';
	import { SYSTEM_MESSAGE_PLACEHOLDER } from '$lib/constants';
	import { REASONING_TAGS } from '$lib/constants/agentic';
	import { MessageRole, AttachmentType, AgenticSectionType, MCPPromptIdPrefix } from '$lib/enums';
	import { fadeInView } from '$lib/actions/fade-in-view.svelte';
	import {
		ChatMessageAssistant,
		ChatMessageUser,
		ChatMessageSystem,
		ChatMessageSkillAttachment
	} from '$lib/components/app/chat';
	import { DialogSkillAddNew, DialogSkillSync } from '$lib/components/app';
	import { parseFilesToMessageExtras } from '$lib/utils/browser-only';
	import { deriveAgenticSections, hasContentDiff } from '$lib/utils';
	import { skillsStore } from '$lib/stores/skills.svelte';
	import type { DatabaseMessageExtraSkill, Skill } from '$lib/types';
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
	let skillExtras = $derived.by(() => {
		if (message.role !== MessageRole.SYSTEM) return [];
		if (!message.extra || message.extra.length === 0) return [];
		const extras: DatabaseMessageExtraSkill[] = [];
		for (const extra of message.extra) {
			if (extra.type === AttachmentType.SKILL) extras.push(extra);
		}
		return extras;
	});
	let skillExtra = $derived(skillExtras[0] ?? null);
	let isEditing = $state(false);
	let showDeleteDialog = $state(false);
	let shouldBranchAfterEdit = $state(false);
	let skillDialogOpen = $state(false);

	// Pull the referenced library skill (if any). A null skill means the
	// SKILL extra points at a skill the user has since deleted.
	let referencedSkill = $derived(skillExtra ? skillsStore.getSkill(skillExtra.skillId) : undefined);
	let skillIsStale = $derived.by(() => {
		if (!skillExtra || !referencedSkill) return false;
		return hasContentDiff(message.content, referencedSkill.content);
	});
	// Which side has changed more recently?
	//   - `libraryEdited`: the library row's `lastModified` is newer than
	//     the message's `timestamp` (or the message has never been edited
	//     since creation) — user wants to pull the new library version
	//     into the conversation.
	//   - `messageEdited`: the message body differs from the library and
	//     the library hasn't been touched since the message was created —
	//     user wants to push the message's edits back to the library.
	//   - both true: the two diverged independently. The dialog offers
	//     both directions and lets the user pick.
	let libraryEdited = $derived(
		!!referencedSkill && referencedSkill.lastModified > message.timestamp
	);
	let messageEdited = $derived(
		!!referencedSkill && hasContentDiff(message.content, referencedSkill.content) && !libraryEdited
	);
	let showUpdateLibraryDialog = $state(false);
	// Diff sides for the unified dialog:
	//   - `messageEdited`: library on the left, conversation on the right
	//     (the user is pushing the conversation's edits back to the library).
	//   - `libraryEdited`:  conversation on the left, library on the right
	//     (the user is pulling the new library content into the conversation).
	let dialogCurrentContent = $derived(
		messageEdited ? (referencedSkill?.content ?? '') : message.content
	);
	let dialogUpdatedContent = $derived(
		messageEdited
			? isEditing
				? editedContent.trim()
				: message.content.trim()
			: (referencedSkill?.content ?? '')
	);
	let updatedTitle: string | undefined = $state(undefined);
	let updatedDescription: string | undefined = $state(undefined);

	// System-message affordances surfaced to the edit form.
	let isSystemMessage = $derived(message.role === MessageRole.SYSTEM);
	let isSystemPlaceholder = $derived(
		isSystemMessage && message.content === SYSTEM_MESSAGE_PLACEHOLDER
	);
	let canAddToLibrary = $derived.by(() => {
		if (message.role !== MessageRole.SYSTEM) return false;
		const skill = skillExtra;
		if (!skill) return true;
		if (skill.skillId.startsWith(MCPPromptIdPrefix.PROMPT)) return false;
		return !referencedSkill;
	});
	let canUpdateLibrarySkill = $derived.by(() => {
		if (message.role !== MessageRole.SYSTEM) return false;
		const skill = skillExtra;
		if (!skill) return false;
		if (skill.skillId.startsWith(MCPPromptIdPrefix.PROMPT)) return false;
		return !!referencedSkill;
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
		get canUpdateLibrarySkill() {
			return canUpdateLibrarySkill;
		},
		get librarySkillTitle() {
			return referencedSkill?.name;
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
		updateLibrarySkill: handleUpdateLibrarySkill,
		updateLibraryPrompt: handleUpdateLibrarySkill,

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
		// placeholder system message). We can't create a library skill without
		// content anyway.
		if (!editedContent.trim()) {
			await handleSaveEdit();
			return;
		}

		skillDialogOpen = true;
	}

	async function handleUpdateLibrarySkill() {
		if (message.role !== MessageRole.SYSTEM) return;
		if (!referencedSkill) return;

		const newContent = editedContent.trim();

		// Empty content falls back to the normal save flow. We can't push
		// empty content to the library skill either.
		if (!newContent) {
			await handleSaveEdit();
			return;
		}

		showUpdateLibraryDialog = true;
	}

	// Card-button entrypoint: open the dialog seeded with the *current*
	// message content (no edit form involvement). The confirm handler
	// applies the updates directly to the library row and refreshes the
	// message extras to reflect the new title/description.
	function openUpdateLibraryDialogFromCard() {
		if (message.role !== MessageRole.SYSTEM) return;
		if (!referencedSkill) return;
		updatedTitle = referencedSkill.name;
		updatedDescription = referencedSkill.description;
		showUpdateLibraryDialog = true;
	}

	async function handleUpdateLibraryConfirm(updatedName?: string, updatedDescription?: string) {
		if (!referencedSkill) return;

		const newContent = isEditing ? editedContent.trim() : message.content.trim();
		const newName = updatedName?.trim() ?? '';
		const newDescription = updatedDescription?.trim() ?? '';

		const updates: Partial<{ content: string; name: string; description: string }> = {
			content: newContent
		};
		if (newName && newName !== referencedSkill.name) {
			updates.name = newName;
		}
		if (newDescription && newDescription !== referencedSkill.description) {
			updates.description = newDescription;
		}

		await skillsStore.updateSkill(referencedSkill.id, updates);

		// Refresh the message extras so the card reflects the new
		// title/description, and persist the (possibly unchanged) body.
		const refreshed = skillsStore.getSkill(referencedSkill.id);
		if (refreshed) {
			await saveSystemPromptWithSkill(newContent, refreshed);
		}

		if (isEditing) await handleSaveEdit();
		showUpdateLibraryDialog = false;
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

		// Preserve existing extras (drop stale SKILL so we can re-add cleanly)
		const existingExtras = message.extra || [];
		const extrasToSave = existingExtras.filter(
			(e: DatabaseMessageExtra) => e.type !== AttachmentType.SKILL
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

	async function saveSystemPromptWithSkill(content: string, skill: Skill) {
		// message.extra is a deep $state proxy; snapshot to plain values so
		// Dexie/IndexedDB can structured-clone it (see getMergedExtras precedent)
		const existingExtras = (
			message.extra ? $state.snapshot(message.extra) : []
		) as DatabaseMessageExtra[];
		const extras: DatabaseMessageExtra[] = [
			...existingExtras.filter((e: DatabaseMessageExtra) => e.type !== AttachmentType.SKILL),
			{
				type: AttachmentType.SKILL,
				skillId: skill.id,
				title: skill.name,
				description: skill.description,
				origin: skill.origin,
				path: skill.path
			}
		];

		await DatabaseService.updateMessage(message.id, { content, extra: extras });
		const index = conversationsStore.findMessageIndex(message.id);
		if (index !== -1) {
			conversationsStore.updateMessageAtIndex(index, { content, extra: extras });
		}
	}

	// Overwrite the system message body with the library version (sync
	// from library). Used when the library has been updated and the user
	// wants the conversation to reflect the new library content.
	async function syncSkillFromLibrary() {
		if (!skillExtra || !referencedSkill) return;
		await saveSystemPromptWithSkill(referencedSkill.content, referencedSkill);
		showUpdateLibraryDialog = false;
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
	{#if message.role === MessageRole.SYSTEM}
		{#if skillExtras.length > 0}
			<div class="flex flex-col items-end gap-2 {className}">
				{#each skillExtras as extra: DatabaseMessageExtraSkill (extra.skillId)}
					<ChatMessageSkillAttachment
						{message}
						skillExtra={extra}
						messageRole={message.role}
						skillName={skillsStore.getSkill(extra.skillId)?.name ?? extra.title}
						{skillIsStale}
						{deletionInfo}
						onSkillUpdate={openUpdateLibraryDialogFromCard}
						onConfirmDelete={handleConfirmDelete}
						onCopy={handleCopy}
						onDelete={handleDelete}
						onEdit={handleEdit}
						onNavigateToSibling={handleNavigateToSibling}
						onShowDeleteDialogChange={handleShowDeleteDialogChange}
						{showDeleteDialog}
						{siblingInfo}
					/>
				{/each}
			</div>
		{:else}
			<ChatMessageSystem
				class={className}
				{deletionInfo}
				{message}
				skillId={undefined}
				title={undefined}
				{skillIsStale}
				onSkillUpdate={openUpdateLibraryDialogFromCard}
				onConfirmDelete={handleConfirmDelete}
				onCopy={handleCopy}
				onDelete={handleDelete}
				onEdit={handleEdit}
				onNavigateToSibling={handleNavigateToSibling}
				onShowDeleteDialogChange={handleShowDeleteDialogChange}
				{showDeleteDialog}
				{siblingInfo}
			/>

			{#if skillDialogOpen}
				<DialogSkillAddNew
					open={skillDialogOpen}
					initialContent={editedContent.trim()}
					onAddSkillComplete={(id: string) => {
						skillDialogOpen = false;
						isEditing = false;
						const skill = skillsStore.getSkill(id);
						if (skill) {
							saveSystemPromptWithSkill(editedContent.trim(), skill);
						}
					}}
				/>
			{/if}
		{/if}

		{#if showUpdateLibraryDialog && referencedSkill}
			<DialogSkillSync
				bind:open={showUpdateLibraryDialog}
				skillName={referencedSkill.name}
				currentTitle={skillExtra?.title}
				currentContent={dialogCurrentContent}
				updatedContent={dialogUpdatedContent}
				{messageEdited}
				{libraryEdited}
				bind:updatedTitle
				bind:updatedDescription
				editableTitle
				editableDescription
				onUpdate={messageEdited ? handleUpdateLibraryConfirm : syncSkillFromLibrary}
				onUseLibraryVersion={syncSkillFromLibrary}
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
