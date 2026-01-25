import { AttachmentType, FileTypeCategory, SpecialFileType } from '$lib/enums';
import { getFileTypeCategory, getFileTypeCategoryByExtension, isImageFile } from '$lib/utils';

/**
 * Formats attachment content for API requests with consistent header style.
 * Used when converting message attachments to text content parts.
 *
 * @param label - Type label (e.g., 'File', 'PDF File', 'MCP Prompt')
 * @param name - File or attachment name
 * @param content - The actual content to include
 * @param extra - Optional extra info to append to name (e.g., server name for MCP)
 * @returns Formatted string with header and content
 */
export function formatAttachmentText(
	label: string,
	name: string,
	content: string,
	extra?: string
): string {
	const header = extra ? `${name} (${extra})` : name;
	return `\n\n--- ${label}: ${header} ---\n${content}`;
}

export interface AttachmentDisplayItemsOptions {
	uploadedFiles?: ChatUploadedFile[];
	attachments?: DatabaseMessageExtra[];
}

/**
 * Check if an uploaded file is an MCP prompt
 */
function isMcpPromptUpload(file: ChatUploadedFile): boolean {
	return file.type === SpecialFileType.MCP_PROMPT && !!file.mcpPrompt;
}

/**
 * Check if an attachment is an MCP prompt
 */
function isMcpPromptAttachment(attachment: DatabaseMessageExtra): boolean {
	return attachment.type === AttachmentType.MCP_PROMPT;
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
	const { uploadedFiles = [], attachments = [] } = options;
	const items: ChatAttachmentDisplayItem[] = [];

	// Add uploaded files (ChatForm)
	for (const file of uploadedFiles) {
		items.push({
			id: file.id,
			name: file.name,
			size: file.size,
			preview: file.preview,
			isImage: getUploadedFileCategory(file) === FileTypeCategory.IMAGE,
			isMcpPrompt: isMcpPromptUpload(file),
			isLoading: file.isLoading,
			loadError: file.loadError,
			uploadedFile: file,
			textContent: file.textContent
		});
	}

	// Add stored attachments (ChatMessage)
	for (const [index, attachment] of attachments.entries()) {
		const isImage = isImageFile(attachment);
		const isMcpPrompt = isMcpPromptAttachment(attachment);

		items.push({
			id: `attachment-${index}`,
			name: attachment.name,
			preview: isImage && 'base64Url' in attachment ? attachment.base64Url : undefined,
			isImage,
			isMcpPrompt,
			attachment,
			attachmentIndex: index,
			textContent: 'content' in attachment ? attachment.content : undefined
		});
	}

	return items.reverse();
}
