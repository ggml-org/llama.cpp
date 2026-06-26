<script lang="ts">
	import { X, AlertTriangle } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { Switch } from '$lib/components/ui/switch';
	import { ChatForm, DialogConfirmation } from '$lib/components/app';
	import { getMessageEditContext } from '$lib/contexts';
	import { ChatMessageEditFormVariant, KeyboardKey, MessageRole } from '$lib/enums';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { processFilesToChatUploaded } from '$lib/utils/browser-only';

	interface Props {
		variant?: ChatMessageEditFormVariant;
	}

	let { variant = ChatMessageEditFormVariant.DEFAULT }: Props = $props();

	const editCtx = getMessageEditContext();

	let saveWithoutRegenerate = $state(false);
	let showDiscardDialog = $state(false);
	let branchAfterEdit = $state(false);
	let addToLibrary = $state(false);
	let updateLibraryPrompt = $state(false);

	let isUserMessage = $derived(editCtx.messageRole === MessageRole.USER);
	let isAssistantMessage = $derived(editCtx.messageRole === MessageRole.ASSISTANT);
	let isSystemMessage = $derived(editCtx.messageRole === MessageRole.SYSTEM);

	let hasUnsavedChanges = $derived.by(() => {
		if (editCtx.editedContent !== editCtx.originalContent) return true;
		if (editCtx.editedUploadedFiles.length > 0) return true;

		const extrasChanged =
			editCtx.editedExtras.length !== editCtx.originalExtras.length ||
			editCtx.editedExtras.some((extra, i) => extra !== editCtx.originalExtras[i]);

		if (extrasChanged) return true;

		return false;
	});

	let hasAttachments = $derived(
		(editCtx.editedExtras && editCtx.editedExtras.length > 0) ||
			(editCtx.editedUploadedFiles && editCtx.editedUploadedFiles.length > 0)
	);

	let canSubmit = $derived(editCtx.editedContent.trim().length > 0 || hasAttachments);

	function handleGlobalKeydown(event: KeyboardEvent) {
		if (event.key === KeyboardKey.ESCAPE) {
			event.preventDefault();
			attemptCancel();
		}
	}

	function attemptCancel() {
		if (hasUnsavedChanges) {
			showDiscardDialog = true;
		} else {
			editCtx.cancel();
		}
	}

	function handleSubmit() {
		if (!canSubmit) return;

		if (isUserMessage && saveWithoutRegenerate && editCtx.showSaveOnlyOption) {
			editCtx.saveOnly();
		} else if (isSystemMessage && addToLibrary) {
			editCtx.saveAsLibrary();
		} else if (isSystemMessage && updateLibraryPrompt) {
			editCtx.updateLibraryPrompt();
		} else {
			if (isAssistantMessage && editCtx.setShouldBranchAfterEdit) {
				editCtx.setShouldBranchAfterEdit(branchAfterEdit);
			}

			editCtx.save();
		}

		saveWithoutRegenerate = false;
		branchAfterEdit = false;
		addToLibrary = false;
		updateLibraryPrompt = false;
	}

	function handleAttachmentRemove(index: number) {
		const newExtras = [...editCtx.editedExtras];
		newExtras.splice(index, 1);
		editCtx.setExtras(newExtras);
	}

	function handleUploadedFileRemove(fileId: string) {
		const newFiles = editCtx.editedUploadedFiles.filter((f) => f.id !== fileId);
		editCtx.setUploadedFiles(newFiles);
	}

	async function handleFilesAdd(files: File[]) {
		const processed = await processFilesToChatUploaded(files);
		editCtx.setUploadedFiles([...editCtx.editedUploadedFiles, ...processed]);
	}

	$effect(() => {
		chatStore.setEditModeActive(handleFilesAdd);

		return () => {
			chatStore.clearEditMode();
		};
	});
</script>

<svelte:window onkeydown={handleGlobalKeydown} />

{#if variant === ChatMessageEditFormVariant.SYSTEM}
	<ChatForm
		class="w-full **:data-[slot=input-area]:border-2! **:data-[slot=input-area]:border-dashed! **:data-[slot=input-area]:border-border/50! **:data-[slot=input-area]:bg-muted!"
		value={editCtx.editedContent}
		attachments={editCtx.editedExtras}
		bind:uploadedFiles={editCtx.editedUploadedFiles}
		placeholder="Edit system message..."
		showMcpPromptButton={false}
		showAddButton={false}
		showModelSelector={false}
		showReasoningToggle={false}
		onValueChange={editCtx.setContent}
		onAttachmentRemove={handleAttachmentRemove}
		onUploadedFileRemove={handleUploadedFileRemove}
		onFilesAdd={handleFilesAdd}
		onSubmit={handleSubmit}
	/>
{:else}
	<div class="relative w-full w-full">
		<ChatForm
			value={editCtx.editedContent}
			attachments={editCtx.editedExtras}
			bind:uploadedFiles={editCtx.editedUploadedFiles}
			placeholder="Edit your message..."
			showMcpPromptButton
			showAddButton={editCtx.messageRole === MessageRole.USER}
			showModelSelector={editCtx.messageRole === MessageRole.USER}
			onValueChange={editCtx.setContent}
			onAttachmentRemove={handleAttachmentRemove}
			onUploadedFileRemove={handleUploadedFileRemove}
			onFilesAdd={handleFilesAdd}
			onSubmit={handleSubmit}
		/>
	</div>
{/if}

<div class="mt-2 flex w-full w-full items-center justify-between gap-2">
	<div class="flex min-w-0 items-center gap-2">
		{#if isUserMessage && editCtx.showSaveOnlyOption}
			<Switch id="save-only-switch" bind:checked={saveWithoutRegenerate} class="scale-75" />

			<label for="save-only-switch" class="cursor-pointer text-xs text-muted-foreground">
				Update without re-sending
			</label>
		{:else if isAssistantMessage}
			<Switch id="branch-after-edit" bind:checked={branchAfterEdit} class="scale-75" />

			<label for="branch-after-edit" class="cursor-pointer text-xs text-muted-foreground">
				Branch conversation after edit
			</label>
		{:else if isSystemMessage && editCtx.canAddToLibrary}
			<Switch id="add-to-library" bind:checked={addToLibrary} class="scale-75" />

			<label for="add-to-library" class="cursor-pointer text-xs text-muted-foreground">
				Add Prompt to library
			</label>
		{:else if isSystemMessage && editCtx.canUpdateLibraryPrompt}
			<Switch id="update-library-prompt" bind:checked={updateLibraryPrompt} class="scale-75" />

			<label for="update-library-prompt" class="cursor-pointer text-xs text-muted-foreground">
				Update{editCtx.libraryPromptTitle ? ` "${editCtx.libraryPromptTitle}"` : ''} prompt in library
			</label>
		{:else}
			<div></div>
		{/if}
	</div>

	<Button class="h-7 px-3 text-xs" onclick={attemptCancel} size="sm" variant="ghost">
		<X class="mr-1 h-3 w-3" />

		Cancel
	</Button>
</div>

<DialogConfirmation
	bind:open={showDiscardDialog}
	title="Discard changes?"
	description="You have unsaved changes. Are you sure you want to discard them?"
	confirmText="Discard"
	cancelText="Keep editing"
	variant="destructive"
	icon={AlertTriangle}
	onConfirm={editCtx.cancel}
	onCancel={() => (showDiscardDialog = false)}
/>
