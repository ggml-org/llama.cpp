import type { AttachmentType } from '$lib/enums';

/**
 * Common utility types used across the application
 */

/**
 * Common utility types used across the application
 */

/**
 * Represents a key-value pair.
 * Used for headers, environment variables, query parameters, etc.
 */
export interface KeyValuePair {
	key: string;
	value: string;
}

/**
 * Binary detection configuration options
 */
export interface BinaryDetectionOptions {
	/** Number of characters to check from the beginning of the file */
	prefixLength: number;
	/** Maximum ratio of suspicious characters allowed (0.0 to 1.0) */
	suspiciousCharThresholdRatio: number;
	/** Maximum absolute number of null bytes allowed */
	maxAbsoluteNullBytes: number;
}

/**
 * Format for text attachments when copied to clipboard
 */
export interface ClipboardTextAttachment {
	type: typeof AttachmentType.TEXT;
	name: string;
	content: string;
}

/**
 * Format for MCP prompt attachments when copied to clipboard
 */
export interface ClipboardMcpPromptAttachment {
	type: typeof AttachmentType.MCP_PROMPT;
	name: string;
	serverName: string;
	promptName: string;
	content: string;
	arguments?: Record<string, string>;
}

/**
 * Union type for all clipboard attachment types
 */
export type ClipboardAttachment = ClipboardTextAttachment | ClipboardMcpPromptAttachment;

/**
 * Parsed result from clipboard content
 */
export interface ParsedClipboardContent {
	message: string;
	textAttachments: ClipboardTextAttachment[];
	mcpPromptAttachments: ClipboardMcpPromptAttachment[];
}
