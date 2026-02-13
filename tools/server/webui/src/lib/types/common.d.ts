import type { AttachmentType } from '$lib/enums';

/**
 * Represents a key-value pair.
 */
export interface KeyValuePair {
	key: string;
	value: string;
}

/**
 * Binary detection configuration options.
 */
export interface BinaryDetectionOptions {
	prefixLength: number;
	suspiciousCharThresholdRatio: number;
	maxAbsoluteNullBytes: number;
}

/**
 * Format for text attachments when copied to clipboard.
 */
export interface ClipboardTextAttachment {
	type: typeof AttachmentType.TEXT;
	name: string;
	content: string;
}

/**
 * Parsed result from clipboard content.
 */
export interface ParsedClipboardContent {
	message: string;
	textAttachments: ClipboardTextAttachment[];
}
