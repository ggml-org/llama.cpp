<script lang="ts">
	import { ChatAttachmentImagePreview, ChatAttachmentFilePreview } from '$lib/components';
	import ChatAttachmentPreviewDialog from './ChatAttachmentPreviewDialog.svelte';
	import type { ChatUploadedFile } from '$lib/types/chat.d.ts';
	import type { DatabaseMessageExtra } from '$lib/types/database.d.ts';

	interface Props {
		class?: string;
		// For ChatForm - pending uploads
		uploadedFiles?: ChatUploadedFile[];
		onFileRemove?: (fileId: string) => void;
		// For ChatMessage - stored attachments
		attachments?: DatabaseMessageExtra[];
		readonly?: boolean;
		// Image size customization
		imageHeight?: string;
		imageWidth?: string;
		imageClass?: string;
	}

	let {
		uploadedFiles = $bindable([]),
		onFileRemove,
		attachments = [],
		readonly = false,
		class: className = '',
		// Default to small size for form previews
		imageHeight = 'h-24',
		imageWidth = 'w-auto',
		imageClass = ''
	}: Props = $props();

	let displayItems = $derived(getDisplayItems());

	// Preview dialog state
	let previewDialogOpen = $state(false);
	let previewItem = $state<{
		uploadedFile?: ChatUploadedFile;
		attachment?: DatabaseMessageExtra;
		preview?: string;
		name?: string;
		type?: string;
		size?: number;
		textContent?: string;
	} | null>(null);

	function openPreview(item: (typeof displayItems)[0]) {
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

	function closePreview() {
		previewDialogOpen = false;
		previewItem = null;
	}

	function getDisplayItems() {
		const items: Array<{
			id: string;
			name: string;
			size?: number;
			preview?: string;
			type: string;
			isImage: boolean;
			uploadedFile?: ChatUploadedFile;
			attachment?: DatabaseMessageExtra;
			attachmentIndex?: number;
			textContent?: string;
		}> = [];

		// Add uploaded files (ChatForm)
		for (const file of uploadedFiles) {
			items.push({
				id: file.id,
				name: file.name,
				size: file.size,
				preview: file.preview,
				type: file.type,
				isImage: file.type.startsWith('image/'),
				uploadedFile: file,
				textContent: file.textContent
			});
		}

		// Add stored attachments (ChatMessage)
		for (const [index, attachment] of attachments.entries()) {
			if (attachment.type === 'imageFile') {
				items.push({
					id: `attachment-${index}`,
					name: attachment.name,
					preview: attachment.base64Url,
					type: 'image',
					isImage: true
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
			} else if (attachment.type === 'audioFile') {
				items.push({
					id: `attachment-${index}`,
					name: attachment.name,
					type: attachment.mimeType || 'audio',
					isImage: false
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

		return items;
	}
</script>

{#if displayItems.length > 0}
	<div class="flex flex-wrap items-start gap-3 {className}">
		{#each displayItems as item (item.id)}
			{#if item.isImage && item.preview}
				<ChatAttachmentImagePreview
					class="cursor-pointer"
					id={item.id}
					name={item.name}
					preview={item.preview}
					size={item.size}
					{readonly}
					onRemove={onFileRemove}
					height={imageHeight}
					width={imageWidth}
					{imageClass}
					onClick={() => openPreview(item)}
				/>
			{:else}
				<ChatAttachmentFilePreview
					class="cursor-pointer"
					id={item.id}
					name={item.name}
					type={item.type}
					size={item.size}
					{readonly}
					onRemove={onFileRemove}
					textContent={item.textContent}
					onClick={() => openPreview(item)}
				/>
			{/if}
		{/each}
	</div>
{/if}

{#if previewItem}
	<ChatAttachmentPreviewDialog
		bind:open={previewDialogOpen}
		onClose={closePreview}
		uploadedFile={previewItem.uploadedFile}
		attachment={previewItem.attachment}
		preview={previewItem.preview}
		name={previewItem.name}
		type={previewItem.type}
		size={previewItem.size}
		textContent={previewItem.textContent}
	/>
{/if}
