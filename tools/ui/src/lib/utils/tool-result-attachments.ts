import {
	DATA_URI_BASE64_REGEX,
	DEFAULT_IMAGE_EXTENSION,
	IMAGE_MIME_TO_EXTENSION,
	MCP_ATTACHMENT_NAME_PREFIX,
	NEWLINE_SEPARATOR
} from '$lib/constants';
import {
	AttachmentType,
	MimeTypeApplication,
	MimeTypeIncludes,
	MimeTypePrefix,
	MimeTypeText
} from '$lib/enums';
import type { DatabaseMessageExtra } from '$lib/types';

export interface ExtractedToolResultAttachments {
	cleanedResult: string;
	attachments: DatabaseMessageExtra[];
}

function base64Size(base64Data: string): number {
	const padding = base64Data.endsWith('==') ? 2 : base64Data.endsWith('=') ? 1 : 0;
	return Math.max(0, Math.floor((base64Data.length * 3) / 4) - padding);
}

function decodeBase64Text(base64Data: string): string {
	try {
		if (typeof Buffer !== 'undefined') {
			return Buffer.from(base64Data, 'base64').toString('utf-8');
		}

		const binary = atob(base64Data);
		const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
		return new TextDecoder().decode(bytes);
	} catch {
		return '';
	}
}

function getAttachmentExtension(mimeType: string): string {
	if (mimeType.startsWith(MimeTypePrefix.IMAGE)) {
		return IMAGE_MIME_TO_EXTENSION[mimeType] ?? DEFAULT_IMAGE_EXTENSION;
	}

	if (mimeType === MimeTypeApplication.PDF) return 'pdf';
	if (mimeType === MimeTypeText.JSON || mimeType.includes(MimeTypeIncludes.JSON)) return 'json';
	if (mimeType === MimeTypeText.MARKDOWN) return 'md';
	if (mimeType === MimeTypeText.HTML) return 'html';
	if (mimeType === MimeTypeText.CSS) return 'css';
	if (mimeType.startsWith(MimeTypePrefix.TEXT)) return 'txt';
	if (mimeType.startsWith('audio/mpeg') || mimeType.startsWith('audio/mp3')) return 'mp3';
	if (mimeType.startsWith('audio/wav')) return 'wav';
	if (mimeType.startsWith('audio/ogg')) return 'ogg';
	if (mimeType.startsWith('audio/')) return 'audio';
	if (mimeType.startsWith('video/mp4')) return 'mp4';
	if (mimeType.startsWith('video/webm')) return 'webm';
	if (mimeType.startsWith('video/ogg')) return 'ogg';
	if (mimeType.startsWith('video/')) return 'video';

	return 'bin';
}

function buildAttachmentName(mimeType: string, index: number, timestamp: number): string {
	const extension = getAttachmentExtension(mimeType);
	return `${MCP_ATTACHMENT_NAME_PREFIX}-${timestamp}-${index}.${extension}`;
}

export function extractToolResultAttachments(result: string): ExtractedToolResultAttachments {
	if (!result.trim()) {
		return { cleanedResult: result, attachments: [] };
	}

	const timestamp = Date.now();
	const lines = result.split(NEWLINE_SEPARATOR);
	const attachments: DatabaseMessageExtra[] = [];
	let attachmentIndex = 0;

	const cleanedLines = lines.map((line) => {
		const trimmedLine = line.trim();
		const match = trimmedLine.match(DATA_URI_BASE64_REGEX);

		if (!match) {
			return line;
		}

		const mimeType = match[1].toLowerCase();
		const base64Data = match[2];

		if (!base64Data) {
			return line;
		}

		attachmentIndex += 1;
		const name = buildAttachmentName(mimeType, attachmentIndex, timestamp);

		if (mimeType.startsWith(MimeTypePrefix.IMAGE)) {
			attachments.push({ type: AttachmentType.IMAGE, name, base64Url: trimmedLine });
			return `[Attachment saved: ${name}]`;
		}

		if (mimeType.startsWith(MimeTypePrefix.TEXT) || mimeType.includes(MimeTypeIncludes.JSON)) {
			attachments.push({
				type: AttachmentType.TEXT,
				name,
				content: decodeBase64Text(base64Data),
				size: base64Size(base64Data)
			});
			return `[Attachment saved: ${name}]`;
		}

		if (mimeType === MimeTypeApplication.PDF) {
			attachments.push({
				type: AttachmentType.PDF,
				name,
				base64Data,
				content: '',
				processedAsImages: false,
				size: base64Size(base64Data)
			});
			return `[Attachment saved: ${name}]`;
		}

		if (mimeType.startsWith('audio/')) {
			attachments.push({
				type: AttachmentType.AUDIO,
				name,
				mimeType,
				base64Data,
				size: base64Size(base64Data)
			});
			return `[Attachment saved: ${name}]`;
		}

		if (mimeType.startsWith('video/')) {
			attachments.push({
				type: AttachmentType.VIDEO,
				name,
				mimeType,
				base64Data,
				size: base64Size(base64Data)
			});
			return `[Attachment saved: ${name}]`;
		}

		return line;
	});

	return {
		cleanedResult: cleanedLines.join(NEWLINE_SEPARATOR),
		attachments
	};
}
