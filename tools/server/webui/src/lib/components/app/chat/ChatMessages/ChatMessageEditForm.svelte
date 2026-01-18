<script lang="ts">
	import { X, AlertTriangle } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { Switch } from '$lib/components/ui/switch';
	import { ChatAttachmentsList, DialogConfirmation, ModelsSelector } from '$lib/components/app';
	import { INPUT_CLASSES } from '$lib/constants/css-classes';
	import { SETTING_CONFIG_DEFAULT } from '$lib/constants/settings-config';
	import { MimeTypeText } from '$lib/enums';
	import { config } from '$lib/stores/settings.svelte';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { isRouterMode } from '$lib/stores/server.svelte';
	import { autoResizeTextarea, parseClipboardContent } from '$lib/utils';

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
		onEditKeydown: (event: KeyboardEvent) => void;
		onEditedContentChange: (content: string) => void;
		onEditedExtrasChange?: (extras: DatabaseMessageExtra[]) => void;
		onEditedUploadedFilesChange?: (files: ChatUploadedFile[]) => void;
		textareaElement?: HTMLTextAreaElement;
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
		onEditKeydown,
		onEditedContentChange,
		onEditedExtrasChange,
		onEditedUploadedFilesChange,
		textareaElement = $bindable()
	}: Props = $props();

	let fileInputElement: HTMLInputElement | undefined = $state();
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

	function handleFileInputChange(event: Event) {
		const input = event.target as HTMLInputElement;
		if (!input.files || input.files.length === 0) return;

		const files = Array.from(input.files);

		processNewFiles(files);
		input.value = '';
	}

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
		if (textareaElement) {
			autoResizeTextarea(textareaElement);
		}
	});

	$effect(() => {
		chatStore.setEditModeActive(processNewFiles);

		return () => {
			chatStore.clearEditMode();
		};
	});
</script>

<svelte:window onkeydown={handleGlobalKeydown} />

<div class="relative w-full max-w-[80%]">
	<ChatForm
		bind:this={inputAreaRef}
		value={editCtx.editedContent}
		attachments={editCtx.editedExtras}
		uploadedFiles={editCtx.editedUploadedFiles}
		placeholder="Edit your message..."
		onValueChange={editCtx.setContent}
		onAttachmentRemove={handleAttachmentRemove}
		onUploadedFileRemove={handleUploadedFileRemove}
		onFilesAdd={handleFilesAdd}
		onSubmit={handleSubmit}
	/>

	<div class="relative min-h-[48px] px-5 py-3">
		<textarea
			bind:this={textareaElement}
			bind:value={editedContent}
			class="field-sizing-content max-h-80 min-h-10 w-full resize-none bg-transparent text-sm outline-none"
			onkeydown={onEditKeydown}
			oninput={(e) => {
				autoResizeTextarea(e.currentTarget);
				onEditedContentChange(e.currentTarget.value);
			}}
			onpaste={handlePaste}
			placeholder="Edit your message..."
		></textarea>

		<div class="flex w-full items-center gap-3" style="container-type: inline-size">
			<Button
				class="h-8 w-8 shrink-0 rounded-full bg-transparent p-0 text-muted-foreground hover:bg-foreground/10 hover:text-foreground"
				onclick={() => fileInputElement?.click()}
				type="button"
				title="Add attachment"
			>
				<span class="sr-only">Attach files</span>

				<Paperclip class="h-4 w-4" />
			</Button>

			<div class="flex-1"></div>

			{#if isRouter}
				<ModelsSelector forceForegroundText={true} useGlobalSelection={true} />
			{/if}

			<Button
				class="h-8 w-8 shrink-0 rounded-full p-0"
				onclick={handleSubmit}
				disabled={!canSubmit}
				type="button"
				title={saveWithoutRegenerate ? 'Save changes' : 'Send and regenerate'}
			>
				<span class="sr-only">{saveWithoutRegenerate ? 'Save' : 'Send'}</span>

				<ArrowUp class="h-5 w-5" />
			</Button>
		</div>
	</div>
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
