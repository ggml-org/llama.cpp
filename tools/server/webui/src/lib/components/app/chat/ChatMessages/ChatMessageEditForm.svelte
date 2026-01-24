<script lang="ts">
	import { X, AlertTriangle } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { Switch } from '$lib/components/ui/switch';
	import { ChatFormInputArea, DialogConfirmation } from '$lib/components/app';
	import { chatStore } from '$lib/stores/chat.svelte';

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
		if (editedContent !== originalContent) return true;
		if (editedUploadedFiles.length > 0) return true;

		const extrasChanged =
			editedExtras.length !== originalExtras.length ||
			editedExtras.some((extra, i) => extra !== originalExtras[i]);

		if (extrasChanged) return true;

		return false;
	});

	let hasAttachments = $derived(
		(editedExtras && editedExtras.length > 0) ||
			(editedUploadedFiles && editedUploadedFiles.length > 0)
	);

	let canSubmit = $derived(editedContent.trim().length > 0 || hasAttachments);

	function handleGlobalKeydown(event: KeyboardEvent) {
		if (event.key === 'Escape') {
			event.preventDefault();
			attemptCancel();
		}
	}

	function attemptCancel() {
		if (hasUnsavedChanges) {
			showDiscardDialog = true;
		} else {
			onCancelEdit();
		}
	}

	function handleSubmit() {
		if (!canSubmit) return;

		if (saveWithoutRegenerate && onSaveEditOnly) {
			onSaveEditOnly();
		} else {
			onSaveEdit();
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

		const { processFilesToChatUploaded } = await import('$lib/utils/browser-only');
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
	{#if showSaveOnlyOption && onSaveEditOnly}
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
	onConfirm={onCancelEdit}
	onCancel={() => (showDiscardDialog = false)}
/>
