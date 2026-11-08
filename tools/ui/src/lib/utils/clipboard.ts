import { AttachmentType } from '$lib/enums';
import type {
	ClipboardAttachment,
	ClipboardMcpPromptAttachment,
	ClipboardTextAttachment,
	DatabaseMessageExtra,
	DatabaseMessageExtraLegacyContext,
	DatabaseMessageExtraMcpPrompt,
	DatabaseMessageExtraMcpResource,
	DatabaseMessageExtraTextFile,
	ParsedClipboardContent
} from '$lib/types';
import { toast } from 'svelte-sonner';

/**
 * Copy text to clipboard with toast notification
 * Uses modern clipboard API when available, falls back to legacy method for non-secure contexts
 * @param text - Text to copy to clipboard
 * @param successMessage - Custom success message (optional)
 * @param errorMessage - Custom error message (optional)
 * @returns Promise<boolean> - True if successful, false otherwise
 */
export async function copyToClipboard(
	text: string,
	successMessage = 'Copied to clipboard',
	errorMessage = 'Failed to copy to clipboard'
): Promise<boolean> {
	try {
		// Try modern clipboard API first (secure contexts only)
		if (navigator.clipboard && navigator.clipboard.writeText) {
			await navigator.clipboard.writeText(text);
			toast.success(successMessage);

			return true;
		}

		// Fallback for non-secure contexts
		const textArea = document.createElement('textarea');

		textArea.value = text;
		textArea.style.position = 'fixed';
		textArea.style.left = '-999999px';
		textArea.style.top = '-999999px';
		document.body.appendChild(textArea);
		textArea.focus();
		textArea.select();

		const successful = document.execCommand('copy');

		document.body.removeChild(textArea);

		if (successful) {
			toast.success(successMessage);

			return true;
		} else {
			throw new Error('execCommand failed');
		}
	} catch (error) {
		console.error('Failed to copy to clipboard:', error);
		toast.error(errorMessage);

		return false;
	}
}

/**
 * Copy code with HTML entity decoding and toast notification
 * @param rawCode - Raw code string that may contain HTML entities
 * @param successMessage - Custom success message (optional)
 * @param errorMessage - Custom error message (optional)
 * @returns Promise<boolean> - True if successful, false otherwise
 */
export async function copyCodeToClipboard(
	rawCode: string,
	successMessage = 'Code copied to clipboard',
	errorMessage = 'Failed to copy code'
): Promise<boolean> {
	return copyToClipboard(rawCode, successMessage, errorMessage);
}

/**
 * Formats a message with text attachments for clipboard copying.
 *
 * Default format (asPlainText = false):
 * ```
 * "Text message content"
 * [
 *   {"type":"TEXT","name":"filename.txt","content":"..."},
 *   {"type":"TEXT","name":"another.txt","content":"..."}
 * ]
 * ```
 *
 * Plain text format (asPlainText = true):
 * ```
 * Text message content
 *
 * file content here
 *
 * another file content
 * ```
 *
 * @param content - The message text content
 * @param extras - Optional array of message attachments
 * @param asPlainText - If true, format as plain text without JSON structure
 * @returns Formatted string for clipboard
 */
export function formatMessageForClipboard(
	content: string,
	extras?: DatabaseMessageExtra[],
	asPlainText: boolean = false
): string {
	// Filter text-like attachments (TEXT, LEGACY_CONTEXT, MCP_PROMPT, and MCP_RESOURCE types)
	const textAttachments =
		extras?.filter(
			(
				extra
			): extra is
				| DatabaseMessageExtraTextFile
				| DatabaseMessageExtraLegacyContext
				| DatabaseMessageExtraMcpPrompt
				| DatabaseMessageExtraMcpResource =>
				extra.type === AttachmentType.TEXT ||
				extra.type === AttachmentType.LEGACY_CONTEXT ||
				extra.type === AttachmentType.MCP_PROMPT ||
				extra.type === AttachmentType.MCP_RESOURCE
		) ?? [];

	if (textAttachments.length === 0) {
		return content;
	}

	if (asPlainText) {
		const parts = [content];

		for (const att of textAttachments) {
			parts.push(att.content);
		}

		return parts.join('\n\n');
	}

	const clipboardAttachments: ClipboardAttachment[] = textAttachments.map((att) => {
		if (att.type === AttachmentType.MCP_PROMPT) {
			const mcpAtt = att as DatabaseMessageExtraMcpPrompt;

			return {
				arguments: mcpAtt.arguments,
				content: mcpAtt.content,
				name: mcpAtt.name,
				promptName: mcpAtt.promptName,
				serverName: mcpAtt.serverName,
				type: AttachmentType.MCP_PROMPT
			} as ClipboardMcpPromptAttachment;
		}

		return {
			content: att.content,
			name: att.name,
			type: AttachmentType.TEXT
		} as ClipboardTextAttachment;
	});

	return `${JSON.stringify(content)}\n${JSON.stringify(clipboardAttachments, null, 2)}`;
}

/**
 * Parses clipboard content to extract message and text attachments.
 * Supports both plain text and the special format with attachments.
 *
 * @param clipboardText - Raw text from clipboard
 * @returns Parsed content with message and attachments
 */
export function parseClipboardContent(clipboardText: string): ParsedClipboardContent {
	const defaultResult: ParsedClipboardContent = {
		mcpPromptAttachments: [],
		message: clipboardText,
		textAttachments: []
	};

	if (!clipboardText.startsWith('"')) {
		return defaultResult;
	}

	try {
		let stringEndIndex = -1;
		let escaped = false;

		for (let i = 1; i < clipboardText.length; i++) {
			const char = clipboardText[i];

			if (escaped) {
				escaped = false;

				continue;
			}

			if (char === '\\') {
				escaped = true;

				continue;
			}

			if (char === '"') {
				stringEndIndex = i;

				break;
			}
		}

		if (stringEndIndex === -1) {
			return defaultResult;
		}

		const jsonStringPart = clipboardText.substring(0, stringEndIndex + 1);
		const remainingPart = clipboardText.substring(stringEndIndex + 1).trim();
		const message = JSON.parse(jsonStringPart) as string;

		if (!remainingPart || !remainingPart.startsWith('[')) {
			return {
				mcpPromptAttachments: [],
				message,
				textAttachments: []
			};
		}

		const attachments = JSON.parse(remainingPart) as unknown[];
		const validTextAttachments: ClipboardTextAttachment[] = [];
		const validMcpPromptAttachments: ClipboardMcpPromptAttachment[] = [];

		for (const att of attachments) {
			if (isValidMcpPromptAttachment(att)) {
				validMcpPromptAttachments.push({
					arguments: att.arguments,
					content: att.content,
					name: att.name,
					promptName: att.promptName,
					serverName: att.serverName,
					type: AttachmentType.MCP_PROMPT
				});
			} else if (isValidTextAttachment(att)) {
				validTextAttachments.push({
					content: att.content,
					name: att.name,
					type: AttachmentType.TEXT
				});
			}
		}

		return {
			mcpPromptAttachments: validMcpPromptAttachments,
			message,
			textAttachments: validTextAttachments
		};
	} catch {
		return defaultResult;
	}
}

/**
 * Type guard to validate an MCP prompt attachment object
 * @param obj The object to validate
 * @returns true if the object is a valid MCP prompt attachment
 */
function isValidMcpPromptAttachment(obj: unknown): obj is {
	type: string;
	name: string;
	serverName: string;
	promptName: string;
	content: string;
	arguments?: Record<string, string>;
} {
	if (typeof obj !== 'object' || obj === null) {
		return false;
	}

	const record = obj as Record<string, unknown>;

	return (
		(record.type === AttachmentType.MCP_PROMPT || record.type === 'MCP_PROMPT') &&
		typeof record.name === 'string' &&
		typeof record.serverName === 'string' &&
		typeof record.promptName === 'string' &&
		typeof record.content === 'string'
	);
}

/**
 * Type guard to validate a text attachment object
 * @param obj The object to validate
 * @returns true if the object is a valid text attachment
 */
function isValidTextAttachment(
	obj: unknown
): obj is { type: string; name: string; content: string } {
	if (typeof obj !== 'object' || obj === null) {
		return false;
	}

	const record = obj as Record<string, unknown>;

	return (
		(record.type === AttachmentType.TEXT || record.type === 'TEXT') &&
		typeof record.name === 'string' &&
		typeof record.content === 'string'
	);
}

/**
 * Checks if clipboard content contains our special format with attachments
 * @param clipboardText - Raw text from clipboard
 * @returns true if the clipboard content contains our special format with attachments
 */
export function hasClipboardAttachments(clipboardText: string): boolean {
	if (!clipboardText.startsWith('"')) {
		return false;
	}

	const parsed = parseClipboardContent(clipboardText);

	return parsed.textAttachments.length > 0 || parsed.mcpPromptAttachments.length > 0;
}
