<script lang="ts">
	import { X, AlertTriangle } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { Switch } from '$lib/components/ui/switch';
	import { ChatFormInputArea, DialogConfirmation } from '$lib/components/app';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { processFilesToChatUploaded } from '$lib/utils/browser-only';

	interface Props {
		editedContent: string;
		editedExtras?: DatabaseMessageExtra[];
		editedUploadedFiles?: ChatUploadedFile[];
		originalContent: string;
		originalExtras?: DatabaseMessageExtra[];
		showSaveOnlyOption?: boolean;
		onCancelEdit: () => void;
		onSaveEdit: () => void;
		onSaveEditOnly?: () => void;
		onEditedContentChange: (content: string) => void;
		onEditedExtrasChange?: (extras: DatabaseMessageExtra[]) => void;
		onEditedUploadedFilesChange?: (files: ChatUploadedFile[]) => void;
	}

	let {
		editedContent,
		editedExtras = [],
		editedUploadedFiles = [],
		originalContent,
		originalExtras = [],
		showSaveOnlyOption = false,
		onCancelEdit,
		onSaveEdit,
		onSaveEditOnly,
		onEditedContentChange,
		onEditedExtrasChange,
		onEditedUploadedFilesChange
	}: Props = $props();

	let inputAreaRef: ChatFormInputArea | undefined = $state(undefined);
	let saveWithoutRegenerate = $state(false);
	let showDiscardDialog = $state(false);

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

	let canSubmit = $derived(editedContent.trim().length > 0 || hasAttachments);

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

		if (saveWithoutRegenerate && editCtx.showSaveOnlyOption) {
			editCtx.saveOnly();
		} else {
			editCtx.save();
		}

		saveWithoutRegenerate = false;
	}

	function handleAttachmentRemove(index: number) {
		if (!onEditedExtrasChange) return;

		const newExtras = [...editedExtras];
		newExtras.splice(index, 1);
		onEditedExtrasChange(newExtras);
	}

	function handleUploadedFileRemove(fileId: string) {
		if (!onEditedUploadedFilesChange) return;

		const newFiles = editedUploadedFiles.filter((f) => f.id !== fileId);
		onEditedUploadedFilesChange(newFiles);
	}

	async function handleFilesAdd(files: File[]) {
		if (!onEditedUploadedFilesChange) return;

		const processed = await processFilesToChatUploaded(files);

		onEditedUploadedFilesChange([...editedUploadedFiles, processed].flat());
	}

	function handleUploadedFilesChange(files: ChatUploadedFile[]) {
		onEditedUploadedFilesChange?.(files);
	}

	$effect(() => {
		chatStore.setEditModeActive(handleFilesAdd);

		return () => {
			chatStore.clearEditMode();
		};
	});
</script>

<svelte:window onkeydown={handleGlobalKeydown} />

<div class="relative w-full max-w-[80%]">
	<ChatFormInputArea
		bind:this={inputAreaRef}
		value={editedContent}
		attachments={editedExtras}
		uploadedFiles={editedUploadedFiles}
		placeholder="Edit your message..."
		onValueChange={onEditedContentChange}
		onAttachmentRemove={handleAttachmentRemove}
		onUploadedFileRemove={handleUploadedFileRemove}
		onUploadedFilesChange={handleUploadedFilesChange}
		onFilesAdd={handleFilesAdd}
		onSubmit={handleSubmit}
	/>
</div>

<div class="mt-2 flex w-full max-w-[80%] items-center justify-between">
	{#if editCtx.showSaveOnlyOption}
		<div class="flex items-center gap-2">
			<Switch id="save-only-switch" bind:checked={saveWithoutRegenerate} class="scale-75" />

			<label for="save-only-switch" class="cursor-pointer text-xs text-muted-foreground">
				Update without re-sending
			</label>
		</div>
	{:else}
		<div></div>
	{/if}

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
