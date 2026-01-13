<script lang="ts">
	import { X, ArrowUp, Paperclip, AlertTriangle } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { Switch } from '$lib/components/ui/switch';
	import { ChatAttachmentsList, DialogConfirmation, ModelsSelector } from '$lib/components/app';
	import { INPUT_CLASSES } from '$lib/constants/input-classes';
	import { SETTING_CONFIG_DEFAULT } from '$lib/constants/settings-config';
	import { AttachmentType, FileTypeCategory, MimeTypeText } from '$lib/enums';
	import { config } from '$lib/stores/settings.svelte';
	import { useModelChangeValidation } from '$lib/hooks/use-model-change-validation.svelte';
	import { setEditModeActive, clearEditMode } from '$lib/stores/chat.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { modelsStore } from '$lib/stores/models.svelte';
	import { isRouterMode } from '$lib/stores/server.svelte';
	import {
		autoResizeTextarea,
		getFileTypeCategory,
		getFileTypeCategoryByExtension,
		parseClipboardContent
	} from '$lib/utils';

	interface Props {
		messageId: string;
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
		messageId,
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
	let isRouter = $derived(isRouterMode());
	let currentConfig = $derived(config());

	let pasteLongTextToFileLength = $derived.by(() => {
		const n = Number(currentConfig.pasteLongTextToFileLen);

		return Number.isNaN(n) ? Number(SETTING_CONFIG_DEFAULT.pasteLongTextToFileLen) : n;
	});

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

	function getEditedAttachmentsModalities(): ModelModalities {
		const modalities: ModelModalities = { vision: false, audio: false };

		for (const extra of editedExtras) {
			if (extra.type === AttachmentType.IMAGE) {
				modalities.vision = true;
			}

			if (
				extra.type === AttachmentType.PDF &&
				'processedAsImages' in extra &&
				extra.processedAsImages
			) {
				modalities.vision = true;
			}

			if (extra.type === AttachmentType.AUDIO) {
				modalities.audio = true;
			}
		}

		for (const file of editedUploadedFiles) {
			const category = getFileTypeCategory(file.type) || getFileTypeCategoryByExtension(file.name);
			if (category === FileTypeCategory.IMAGE) {
				modalities.vision = true;
			}
			if (category === FileTypeCategory.AUDIO) {
				modalities.audio = true;
			}
		}

		return modalities;
	}

	function getRequiredModalities(): ModelModalities {
		const beforeModalities = conversationsStore.getModalitiesUpToMessage(messageId);
		const editedModalities = getEditedAttachmentsModalities();

		return {
			vision: beforeModalities.vision || editedModalities.vision,
			audio: beforeModalities.audio || editedModalities.audio
		};
	}

	const { handleModelChange } = useModelChangeValidation({
		getRequiredModalities,
		onValidationFailure: async (previousModelId) => {
			if (previousModelId) {
				await modelsStore.selectModelById(previousModelId);
			}
		}
	});

	function handleFileInputChange(event: Event) {
		const input = event.target as HTMLInputElement;
		if (!input.files || input.files.length === 0) return;

		const files = Array.from(input.files);

		processNewFiles(files);
		input.value = '';
	}

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

	function handleRemoveExistingAttachment(index: number) {
		if (!onEditedExtrasChange) return;

		const newExtras = [...editedExtras];

		newExtras.splice(index, 1);
		onEditedExtrasChange(newExtras);
	}

	function handleRemoveUploadedFile(fileId: string) {
		if (!onEditedUploadedFilesChange) return;

		const newFiles = editedUploadedFiles.filter((f) => f.id !== fileId);

		onEditedUploadedFilesChange(newFiles);
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

	async function processNewFiles(files: File[]) {
		if (!onEditedUploadedFilesChange) return;

		const { processFilesToChatUploaded } = await import('$lib/utils/browser-only');
		const processed = await processFilesToChatUploaded(files);

		onEditedUploadedFilesChange([...editedUploadedFiles, ...processed]);
	}

	function handlePaste(event: ClipboardEvent) {
		if (!event.clipboardData) return;

		const files = Array.from(event.clipboardData.items)
			.filter((item) => item.kind === 'file')
			.map((item) => item.getAsFile())
			.filter((file): file is File => file !== null);

		if (files.length > 0) {
			event.preventDefault();
			processNewFiles(files);

			return;
		}

		const text = event.clipboardData.getData(MimeTypeText.PLAIN);

		if (text.startsWith('"')) {
			const parsed = parseClipboardContent(text);

			if (parsed.textAttachments.length > 0) {
				event.preventDefault();
				onEditedContentChange(parsed.message);

				const attachmentFiles = parsed.textAttachments.map(
					(att) =>
						new File([att.content], att.name, {
							type: MimeTypeText.PLAIN
						})
				);

				processNewFiles(attachmentFiles);

				setTimeout(() => {
					textareaElement?.focus();
				}, 10);

				return;
			}
		}

		if (
			text.length > 0 &&
			pasteLongTextToFileLength > 0 &&
			text.length > pasteLongTextToFileLength
		) {
			event.preventDefault();

			const textFile = new File([text], 'Pasted', {
				type: MimeTypeText.PLAIN
			});

			processNewFiles([textFile]);
		}
	}

	$effect(() => {
		if (textareaElement) {
			autoResizeTextarea(textareaElement);
		}
	});

	$effect(() => {
		setEditModeActive(processNewFiles);

		return () => {
			clearEditMode();
		};
	});
</script>

<svelte:window onkeydown={handleGlobalKeydown} />

<input
	bind:this={fileInputElement}
	type="file"
	multiple
	class="hidden"
	onchange={handleFileInputChange}
/>

<div
	class="{INPUT_CLASSES} w-full max-w-[80%] overflow-hidden rounded-3xl backdrop-blur-md"
	data-slot="edit-form"
>
	<ChatAttachmentsList
		attachments={editedExtras}
		uploadedFiles={editedUploadedFiles}
		readonly={false}
		onFileRemove={(fileId) => {
			if (fileId.startsWith('attachment-')) {
				const index = parseInt(fileId.replace('attachment-', ''), 10);
				if (!isNaN(index) && index >= 0 && index < editedExtras.length) {
					handleRemoveExistingAttachment(index);
				}
			} else {
				handleRemoveUploadedFile(fileId);
			}
		}}
		limitToSingleRow
		class="py-5"
		style="scroll-padding: 1rem;"
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
				<ModelsSelector
					forceForegroundText={true}
					useGlobalSelection={true}
					onModelChange={handleModelChange}
				/>
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
