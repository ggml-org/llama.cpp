import { FILE_PATH_SEPARATOR_REGEX, NEWLINE } from '$lib/constants/code';
import {
	PREFIX_FILE,
	PREFIX_MIME,
	PREFIX_SIZE,
	READ_MEDIA_SIZE_REGEX
} from '$lib/constants/read-media';
import type { AgenticSection } from '$lib/utils';

export interface ReadMediaMeta {
	fileName: string;
	path: string;
	sizeBytes?: number;
	mimeType?: string;
}

/**
 * Parse read_media tool result to extract metadata.
 * Expected format (after extractBase64Attachments processing):
 *   File: /path/to/file.png
 *   Size: 12345 bytes
 *   MIME: image/png
 *   [Attachment saved: mcp-attachment-xxx.png]
 *
 * The data URI line is replaced by the attachment marker by
 * agenticStore.extractBase64Attachments before storage.
 */
export function parseReadMediaMeta(section: AgenticSection): ReadMediaMeta | null {
	if (!section.toolResult) return null;

	const lines = section.toolResult.split(NEWLINE);

	let fileName = '';
	let path = '';
	let sizeBytes: number | undefined;
	let mimeType: string | undefined;

	for (const line of lines) {
		const trimmed = line.trim();

		if (trimmed.startsWith(PREFIX_FILE)) {
			path = trimmed.slice(PREFIX_FILE.length).trim();
			fileName = path.split(FILE_PATH_SEPARATOR_REGEX).pop() ?? path;
		} else if (trimmed.startsWith(PREFIX_SIZE)) {
			const match = trimmed.match(READ_MEDIA_SIZE_REGEX);

			if (match) sizeBytes = Number(match[1]);
		} else if (trimmed.startsWith(PREFIX_MIME)) {
			mimeType = trimmed.slice(PREFIX_MIME.length).trim();
		}
	}

	if (!path) return null;

	return { fileName, mimeType, path, sizeBytes };
}
