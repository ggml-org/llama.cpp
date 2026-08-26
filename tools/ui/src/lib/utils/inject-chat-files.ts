import { AttachmentType, MessageRole } from '$lib/enums';
import type { DatabaseMessage, DatabaseMessageExtra } from '$lib/types/database';

/**
 * Open WebUI-style file object injected into tool arguments so MCP tools
 * (e.g. thaiocr) can read chat attachments the model did not pass itself.
 */
export interface ChatFileInjection {
	id: string;
	type: 'image' | 'file';
	name: string;
	mimeType: string;
	url: string;
	data: string;
	base64: string;
}

function mimeFromDataUrl(url: string): string | undefined {
	const match = /^data:([^;,]+)/i.exec(url);

	return match?.[1];
}

function stripDataUrl(payload: string): string {
	const comma = payload.indexOf(',');

	return comma >= 0 ? payload.slice(comma + 1) : payload;
}

export function extrasToChatFiles(extras: DatabaseMessageExtra[]): ChatFileInjection[] {
	const files: ChatFileInjection[] = [];

	for (const extra of extras) {
		if (extra.type === AttachmentType.IMAGE && extra.base64Url) {
			const url = extra.base64Url;
			const data = stripDataUrl(url);

			files.push({
				base64: url,
				data,
				id: extra.name,
				mimeType: mimeFromDataUrl(url) || 'image/png',
				name: extra.name,
				type: 'image',
				url
			});
			continue;
		}

		if (extra.type === AttachmentType.PDF && extra.base64Data) {
			const data = extra.base64Data;
			const mime = 'application/pdf';
			const url = `data:${mime};base64,${data}`;

			files.push({
				base64: url,
				data,
				id: extra.name,
				mimeType: 'application/pdf',
				name: extra.name,
				type: 'file',
				url
			});
		}
	}

	return files;
}

/**
 * Last user message extras in a conversation (chat-level attachments).
 * API-normalized messages without `extra` are skipped.
 */
export function collectLastUserMessageExtras(messages: unknown[]): DatabaseMessageExtra[] {
	let last: DatabaseMessageExtra[] = [];

	for (const raw of messages) {
		if (!raw || typeof raw !== 'object') continue;

		const msg = raw as Partial<DatabaseMessage>;

		if (msg.role !== MessageRole.USER && msg.role !== 'user') continue;

		if (Array.isArray(msg.extra) && msg.extra.length > 0) {
			last = msg.extra;
		}
	}

	return last;
}

function isBlank(value: unknown): boolean {
	return value === undefined || value === null || value === '';
}

/**
 * Inject `__files__`, `__file__`, and `__image__` (Open WebUI extra params)
 * into tool-call arguments. Does not overwrite values the model already set.
 * Also fills empty `image` / `file_id` so file-input tools receive a data URI.
 */
export function injectChatFilesIntoToolArgs(
	args: Record<string, unknown>,
	extras: DatabaseMessageExtra[]
): Record<string, unknown> {
	const files = extrasToChatFiles(extras);

	if (files.length === 0) {
		return args;
	}

	const firstImage = files.find((file) => file.type === 'image');
	const firstFile = files.find((file) => file.type === 'file') ?? files[0];
	const out: Record<string, unknown> = { ...args };

	if (out.__files__ === undefined) {
		out.__files__ = files;
	}

	if (out.__file__ === undefined) {
		out.__file__ = firstFile;
	}

	if (out.__image__ === undefined && firstImage) {
		out.__image__ = firstImage.url;
	}

	if (isBlank(out.image)) {
		out.image = firstImage?.url ?? firstFile.url;
	}

	if (isBlank(out.file_id)) {
		out.file_id = firstFile.id;
	}

	return out;
}
