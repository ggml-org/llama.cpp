import type { AgenticSection } from '$lib/utils';

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

	const lines = section.toolResult.split('\n');
	let fileName = '';
	let path = '';
	let sizeBytes: number | undefined;
	let mimeType: string | undefined;

	for (const line of lines) {
		const trimmed = line.trim();
		if (trimmed.startsWith('Image: ')) {
			path = trimmed.slice('Image: '.length).trim();
			fileName = path.split('/').pop() ?? path;
		} else if (trimmed.startsWith('Size: ')) {
			const match = trimmed.match(/Size:\s*(\d+)\s*bytes/);
			if (match) sizeBytes = parseInt(match[1], 10);
		} else if (trimmed.startsWith('MIME: ')) {
			mimeType = trimmed.slice('MIME: '.length).trim();
		}
	}

	if (!path) return null;

	return { fileName, path, sizeBytes, mimeType };
}
