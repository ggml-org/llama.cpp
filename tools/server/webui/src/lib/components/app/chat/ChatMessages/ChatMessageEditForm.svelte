<script lang="ts">
	import { X, ArrowUp, Paperclip } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import { Label } from '$lib/components/ui/label';
	import { ChatAttachmentsList, ModelsSelector } from '$lib/components/app';
	import { INPUT_CLASSES } from '$lib/constants/input-classes';
	import { AttachmentType, FileTypeCategory } from '$lib/enums';
	import { useModelChangeValidation } from '$lib/hooks/use-model-change-validation.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { modelsStore } from '$lib/stores/models.svelte';
	import { isRouterMode } from '$lib/stores/server.svelte';
	import {
		autoResizeTextarea,
		getFileTypeCategory,
		getFileTypeCategoryByExtension
	} from '$lib/utils';

	interface Props {
		messageId: string;
		editedContent: string;
		editedExtras?: DatabaseMessageExtra[];
		editedUploadedFiles?: ChatUploadedFile[];
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
	let isRouter = $derived(isRouterMode());

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

	function handleSubmit() {
		if (!canSubmit) return;

		if (saveWithoutRegenerate && onSaveEditOnly) {
			onSaveEditOnly();
		} else {
			onSaveEdit();
		}
		saveWithoutRegenerate = false;
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

	function handleFileInputChange(event: Event) {
		const input = event.target as HTMLInputElement;
		if (!input.files || input.files.length === 0) return;

		const files = Array.from(input.files);
		processNewFiles(files);
		input.value = '';
	}

	async function processNewFiles(files: File[]) {
		if (!onEditedUploadedFilesChange) return;

		const { processFilesToChatUploaded } = await import('$lib/utils/browser-only');
		const processed = await processFilesToChatUploaded(files);
		onEditedUploadedFilesChange([...editedUploadedFiles, ...processed]);
	}

	$effect(() => {
		if (textareaElement) {
			autoResizeTextarea(textareaElement);
		}
	});
</script>

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
			<Checkbox id="save-without-regenerate" bind:checked={saveWithoutRegenerate} class="h-4 w-4" />
			1
			<Label for="save-without-regenerate" class="cursor-pointer text-xs text-muted-foreground">
				Save only
			</Label>
		</div>
	{:else}
		<div></div>
	{/if}

	<Button class="h-7 px-3 text-xs" onclick={onCancelEdit} size="sm" variant="ghost">
		<X class="mr-1 h-3 w-3" />

		Cancel
	</Button>
</div>
