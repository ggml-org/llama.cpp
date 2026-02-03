import type { ChatMessageTimings, ChatRole, ChatMessageType } from '$lib/types/chat';
import { AttachmentType } from '$lib/enums';

export interface DatabaseConversation {
	currNode: string | null;
	id: string;
	lastModified: number;
	name: string;
}

export interface DatabaseMessageExtraAudioFile {
	type: AttachmentType.AUDIO;
	name: string;
	base64Data: string;
	mimeType: string;
}

export interface DatabaseMessageExtraImageFile {
	type: AttachmentType.IMAGE;
	name: string;
	base64Url: string;
}

/**
 * Legacy format from old webui - pasted content was stored as "context" type
 * @deprecated Use DatabaseMessageExtraTextFile instead
 */
export interface DatabaseMessageExtraLegacyContext {
	type: AttachmentType.LEGACY_CONTEXT;
	name: string;
	content: string;
}

export interface DatabaseMessageExtraPdfFile {
	type: AttachmentType.PDF;
	base64Data: string;
	name: string;
	content: string; // Text content extracted from PDF
	images?: string[]; // Optional: PDF pages as base64 images
	processedAsImages: boolean; // Whether PDF was processed as images
}

export interface DatabaseMessageExtraTextFile {
	type: AttachmentType.TEXT;
	name: string;
	content: string;
}

export type DatabaseMessageExtra =
	| DatabaseMessageExtraImageFile
	| DatabaseMessageExtraTextFile
	| DatabaseMessageExtraAudioFile
	| DatabaseMessageExtraPdfFile
	| DatabaseMessageExtraLegacyContext;

export interface DatabaseMessage {
	id: string;
	convId: string;
	type: ChatMessageType;
	timestamp: number;
	role: ChatRole;
	content: string;
	parent: string;
	thinking: string;
	toolCalls?: string;
	children: string[];
	extra?: DatabaseMessageExtra[];
	timings?: ChatMessageTimings;
	model?: string;
}

/**
 * Represents a single conversation with its associated messages,
 * typically used for import/export operations.
 */
export type ExportedConversation = {
	conv: DatabaseConversation;
	messages: DatabaseMessage[];
};

/**
 * Type representing one or more exported conversations.
 * Can be a single conversation object or an array of them.
 */
export type ExportedConversations = ExportedConversation | ExportedConversation[];
