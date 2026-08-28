import { AttachmentType, FileTypeCategory, SpecialFileType } from '$lib/enums';
import type {
	AttachmentDisplayItemsOptions,
	ChatAttachmentDisplayItem,
	ChatUploadedFile
} from '$lib/types';
import { getFileTypeCategory, getFileTypeCategoryByExtension, isImageFile } from '$lib/utils';

/**
 * Check if a display item represents an MCP prompt
 * (either from attachment type or uploaded file with mcpPrompt metadata)
 */
export function isMcpPrompt(item: ChatAttachmentDisplayItem): boolean {
	if (item.attachment?.type === AttachmentType.MCP_PROMPT) {
		return true;
	}

	if (item.uploadedFile?.type === SpecialFileType.MCP_PROMPT && item.uploadedFile.mcpPrompt) {
		return true;
	}

	return false;
}

/**
 * Check if a display item represents an MCP resource
 */
export function isMcpResource(item: ChatAttachmentDisplayItem): boolean {
	return item.attachment?.type === AttachmentType.MCP_RESOURCE;
}

/**
 * Gets the file type category from an uploaded file, checking both MIME type and extension
 */
function getUploadedFileCategory(file: ChatUploadedFile): FileTypeCategory | null {
	const categoryByMime = getFileTypeCategory(file.type);

	if (categoryByMime) {
		return categoryByMime;
	}

	return getFileTypeCategoryByExtension(file.name);
}

/**
 * Creates a unified list of display items from uploaded files and stored attachments.
 * Items are returned in reverse order (newest first).
 */
export function getAttachmentDisplayItems(
	options: AttachmentDisplayItemsOptions
): ChatAttachmentDisplayItem[] {
	const { attachments = [], uploadedFiles = [] } = options;
	const items: ChatAttachmentDisplayItem[] = [];

	// Add uploaded files (ChatForm)
	for (const file of uploadedFiles) {
		items.push({
			id: file.id,
			isImage: getUploadedFileCategory(file) === FileTypeCategory.IMAGE,
			isLoading: file.isLoading,
			loadError: file.loadError,
			name: file.name,
			preview: file.preview,
			size: file.size,
			textContent: file.textContent,
			uploadedFile: file
		});
	}

	// Add stored attachments (ChatMessage)
	for (const [index, attachment] of attachments.entries()) {
		const isImage = isImageFile(attachment);

		items.push({
			attachment,
			attachmentIndex: index,
			id: `attachment-${index}`,
			isImage,
			name: attachment.name,
			preview: isImage && 'base64Url' in attachment ? attachment.base64Url : undefined,
			size: 'size' in attachment ? attachment.size : undefined,
			textContent: 'content' in attachment ? attachment.content : undefined
		});
	}

	return items.reverse();
}
