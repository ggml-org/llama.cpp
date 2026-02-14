import type { ErrorDialogType } from '$lib/enums';
import type { DatabaseMessage, DatabaseMessageExtra } from './database';

export type ChatMessageType = 'root' | 'text' | 'think' | 'system';
export type ChatRole = 'user' | 'assistant' | 'system';

export interface ChatUploadedFile {
	id: string;
	name: string;
	size: number;
	type: string;
	file: File;
	preview?: string;
	textContent?: string;
	isLoading?: boolean;
	loadError?: string;
}

export interface ChatAttachmentDisplayItem {
	id: string;
	name: string;
	size?: number;
	preview?: string;
	isImage: boolean;
	isLoading?: boolean;
	loadError?: string;
	uploadedFile?: ChatUploadedFile;
	attachment?: DatabaseMessageExtra;
	attachmentIndex?: number;
	textContent?: string;
}

export interface ChatAttachmentPreviewItem {
	uploadedFile?: ChatUploadedFile;
	attachment?: DatabaseMessageExtra;
	preview?: string;
	name?: string;
	size?: number;
	textContent?: string;
}

export interface ChatMessageSiblingInfo {
	message: DatabaseMessage;
	siblingIds: string[];
	currentIndex: number;
	totalSiblings: number;
}

export interface ChatMessagePromptProgress {
	cache: number;
	processed: number;
	time_ms: number;
	total: number;
}

export interface ChatMessageTimings {
	cache_n?: number;
	predicted_ms?: number;
	predicted_n?: number;
	prompt_ms?: number;
	prompt_n?: number;
}

export interface ChatStreamCallbacks {
	onChunk?: (chunk: string) => void;
	onReasoningChunk?: (chunk: string) => void;
	onToolCallChunk?: (chunk: string) => void;
	onAttachments?: (extras: DatabaseMessageExtra[]) => void;
	onModel?: (model: string) => void;
	onTimings?: (timings?: ChatMessageTimings, promptProgress?: ChatMessagePromptProgress) => void;
	onComplete?: (
		content?: string,
		reasoningContent?: string,
		timings?: ChatMessageTimings,
		toolCallContent?: string
	) => void;
	onError?: (error: Error) => void;
}

export interface ErrorDialogState {
	type: ErrorDialogType;
	message: string;
	contextInfo?: { n_prompt_tokens: number; n_ctx: number };
}

export interface LiveProcessingStats {
	tokensProcessed: number;
	totalTokens: number;
	timeMs: number;
	tokensPerSecond: number;
	etaSecs?: number;
}

export interface LiveGenerationStats {
	tokensGenerated: number;
	timeMs: number;
	tokensPerSecond: number;
}

export interface AttachmentDisplayItemsOptions {
	uploadedFiles?: ChatUploadedFile[];
	attachments?: DatabaseMessageExtra[];
}

export interface FileProcessingResult {
	extras: DatabaseMessageExtra[];
	emptyFiles: string[];
}
