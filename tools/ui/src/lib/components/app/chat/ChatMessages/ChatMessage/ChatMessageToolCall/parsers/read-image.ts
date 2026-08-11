import type { AgenticSection } from '$lib/utils';
import { NEWLINE } from '$lib/constants/code';
import { PREFIX_IMAGE, PREFIX_SIZE, PREFIX_MIME } from '$lib/constants/read-image';

export interface ReadImageMeta {
	fileName: string;
	path: string;
	sizeBytes?: number;
	mimeType?: string;
}

/**
 * Parse read_image tool result to extract metadata.
 * Expected format (after extractBase64Attachments processing):
 *   Image: /path/to/file.png
 *   Size: 12345 bytes
 *   MIME: image/png
 *   [Attachment saved: mcp-attachment-xxx.png]
 *
 * The data URI line is replaced by the attachment marker by
 * agenticStore.extractBase64Attachments before storage.
 */
export function parseReadImageMeta(section: AgenticSection): ReadImageMeta | null {
	if (!section.toolResult) return null;

	const lines = section.toolResult.split(NEWLINE);
	let fileName = '';
	let path = '';
	let sizeBytes: number | undefined;
	let mimeType: string | undefined;

	for (const line of lines) {
		const trimmed = line.trim();
		if (trimmed.startsWith(PREFIX_IMAGE)) {
			path = trimmed.slice(PREFIX_IMAGE.length).trim();
			fileName = path.split('/').pop() ?? path;
		} else if (trimmed.startsWith(PREFIX_SIZE)) {
			const match = trimmed.match(new RegExp(`${PREFIX_SIZE}\\s*(\\d+)\\s*bytes`));
			if (match) sizeBytes = parseInt(match[1], 10);
		} else if (trimmed.startsWith(PREFIX_MIME)) {
			mimeType = trimmed.slice(PREFIX_MIME.length).trim();
		}
	}

	if (!path) return null;

	return { fileName, path, sizeBytes, mimeType };
}
