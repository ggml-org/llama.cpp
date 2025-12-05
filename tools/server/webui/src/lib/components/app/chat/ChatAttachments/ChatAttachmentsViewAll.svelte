<script lang="ts">
	import {
		ChatAttachmentThumbnailImage,
		ChatAttachmentThumbnailFile,
		DialogChatAttachmentPreview
	} from '$lib/components/app';
	import { getAttachmentDisplayItems } from '$lib/utils';

	interface Props {
		uploadedFiles?: ChatUploadedFile[];
		attachments?: DatabaseMessageExtra[];
		readonly?: boolean;
		onFileRemove?: (fileId: string) => void;
		imageHeight?: string;
		imageWidth?: string;
		imageClass?: string;
		activeModelId?: string;
	}

	let {
		uploadedFiles = [],
		attachments = [],
		readonly = false,
		onFileRemove,
		imageHeight = 'h-24',
		imageWidth = 'w-auto',
		imageClass = '',
		activeModelId
	}: Props = $props();

	let previewDialogOpen = $state(false);
	let previewItem = $state<ChatAttachmentPreviewItem | null>(null);

	let displayItems = $derived(getAttachmentDisplayItems({ uploadedFiles, attachments }));
	let imageItems = $derived(displayItems.filter((item) => item.isImage));
	let fileItems = $derived(displayItems.filter((item) => !item.isImage));

	function openPreview(item: (typeof displayItems)[0], event?: Event) {
		if (event) {
			event.preventDefault();
			event.stopPropagation();
		}

		previewItem = {
			uploadedFile: item.uploadedFile,
			attachment: item.attachment,
			preview: item.preview,
			name: item.name,
			size: item.size,
			textContent: item.textContent
		};
		previewDialogOpen = true;
	}
</script>

<div class="space-y-4">
	<div class="min-h-0 flex-1 space-y-6 overflow-y-auto px-1">
		{#if fileItems.length > 0}
			<div>
				<h3 class="mb-3 text-sm font-medium text-foreground">Files ({fileItems.length})</h3>
				<div class="flex flex-wrap items-start gap-3">
					{#each fileItems as item (item.id)}
						<ChatAttachmentThumbnailFile
							class="cursor-pointer"
							id={item.id}
							name={item.name}
							size={item.size}
							{readonly}
							onRemove={onFileRemove}
							textContent={item.textContent}
							attachment={item.attachment}
							uploadedFile={item.uploadedFile}
							onClick={(event?: MouseEvent) => openPreview(item, event)}
						/>
					{/each}
				</div>
			</div>
		{/if}

		{#if imageItems.length > 0}
			<div>
				<h3 class="mb-3 text-sm font-medium text-foreground">Images ({imageItems.length})</h3>
				<div class="flex flex-wrap items-start gap-3">
					{#each imageItems as item (item.id)}
						{#if item.preview}
							<ChatAttachmentThumbnailImage
								class="cursor-pointer"
								id={item.id}
								name={item.name}
								preview={item.preview}
								{readonly}
								onRemove={onFileRemove}
								height={imageHeight}
								width={imageWidth}
								{imageClass}
								onClick={(event) => openPreview(item, event)}
							/>
						{/if}
					{/each}
				</div>
			</div>
		{/if}
	</div>
</div>

{#if previewItem}
	<DialogChatAttachmentPreview
		bind:open={previewDialogOpen}
		uploadedFile={previewItem.uploadedFile}
		attachment={previewItem.attachment}
		preview={previewItem.preview}
		name={previewItem.name}
		size={previewItem.size}
		textContent={previewItem.textContent}
		{activeModelId}
	/>
{/if}
