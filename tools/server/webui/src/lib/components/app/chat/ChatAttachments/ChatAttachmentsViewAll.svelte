<script lang="ts">
	import {
		ChatAttachmentThumbnailImage,
		ChatAttachmentThumbnailFile,
		DialogChatAttachmentPreview
	} from '$lib/components/app';
	import { FileTypeCategory } from '$lib/enums/files';
	import { getFileTypeCategory } from '$lib/utils/file-type';
	import type { ChatAttachmentDisplayItem, ChatAttachmentPreviewItem } from '$lib/types/chat';

	interface Props {
		uploadedFiles?: ChatUploadedFile[];
		attachments?: DatabaseMessageExtra[];
		readonly?: boolean;
		onFileRemove?: (fileId: string) => void;
		imageHeight?: string;
		imageWidth?: string;
		imageClass?: string;
	}

	let {
		uploadedFiles = [],
		attachments = [],
		readonly = false,
		onFileRemove,
		imageHeight = 'h-24',
		imageWidth = 'w-auto',
		imageClass = ''
	}: Props = $props();

	let previewDialogOpen = $state(false);
	let previewItem = $state<ChatAttachmentPreviewItem | null>(null);

	let displayItems = $derived(getDisplayItems());
	let imageItems = $derived(displayItems.filter((item) => item.isImage));
	let fileItems = $derived(displayItems.filter((item) => !item.isImage));

	function getDisplayItems(): ChatAttachmentDisplayItem[] {
		const items: ChatAttachmentDisplayItem[] = [];

		for (const file of uploadedFiles) {
			items.push({
				id: file.id,
				name: file.name,
				size: file.size,
				preview: file.preview,
				type: file.type,
				isImage: getFileTypeCategory(file.type) === FileTypeCategory.IMAGE,
				uploadedFile: file,
				textContent: file.textContent
			});
		}

		for (const [index, attachment] of attachments.entries()) {
			if (attachment.type === 'imageFile') {
				items.push({
					id: `attachment-${index}`,
					name: attachment.name,
					preview: attachment.base64Url,
					type: 'image',
					isImage: true,
					attachment,
					attachmentIndex: index
				});
			} else if (attachment.type === 'textFile') {
				items.push({
					id: `attachment-${index}`,
					name: attachment.name,
					type: 'text',
					isImage: false,
					attachment,
					attachmentIndex: index,
					textContent: attachment.content
				});
			} else if (attachment.type === 'context') {
				// Legacy format from old webui - treat as text file
				items.push({
					id: `attachment-${index}`,
					name: attachment.name,
					type: 'text',
					isImage: false,
					attachment,
					attachmentIndex: index,
					textContent: attachment.content
				});
			} else if (attachment.type === 'audioFile') {
				items.push({
					id: `attachment-${index}`,
					name: attachment.name,
					type: attachment.mimeType || 'audio',
					isImage: false,
					attachment,
					attachmentIndex: index
				});
			} else if (attachment.type === 'pdfFile') {
				items.push({
					id: `attachment-${index}`,
					name: attachment.name,
					type: 'application/pdf',
					isImage: false,
					attachment,
					attachmentIndex: index,
					textContent: attachment.content
				});
			}
		}

		return items.reverse();
	}

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
			type: item.type,
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
							type={item.type}
							size={item.size}
							{readonly}
							onRemove={onFileRemove}
							textContent={item.textContent}
							onClick={(event) => openPreview(item, event)}
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
		type={previewItem.type}
		size={previewItem.size}
		textContent={previewItem.textContent}
	/>
{/if}
