import type { ApiChatCompletionToolCall } from './api';
import type { DatabaseMessage, DatabaseMessageExtra } from './database';
import type {
	AttachmentAction,
	AttachmentItemEnabledWhen,
	AttachmentItemVisibleWhen,
	AttachmentMenuItemId,
	ChatFormCommandAction,
	ErrorDialogType,
	FileMentionEntryType
} from '$lib/enums';
import type { Component } from 'svelte';

/**
 * A single item in the chat form attachment menu.
 */
export interface AttachmentMenuItem {
	/** Unique identifier for the item */
	id: AttachmentMenuItemId;
	/** Display label */
	label: string;
	/** Lucide icon component */
	icon: Component;
	/** Extra CSS class applied to the item (e.g. for test selectors) */
	class?: string;
	/** Whether the item requires a specific modality to be enabled */
	enabledWhen?: AttachmentItemEnabledWhen;
	/** Tooltip shown when the item is disabled */
	disabledTooltip?: string;
	/** Callback key on the Props interface to invoke when clicked */
	action: AttachmentAction;
	/** Whether the item is only shown when a specific capability is present */
	visibleWhen?: AttachmentItemVisibleWhen;
	/** Whether this item has a tooltip even when enabled (uses dynamic text) */
	hasEnabledTooltip?: boolean;
}

export interface ChatUploadedFile {
	id: string;
	name: string;
	size: number;
	type: string;
	file: File;
	preview?: string;
	textContent?: string;
	mcpPrompt?: {
		serverName: string;
		promptName: string;
		arguments?: Record<string, string>;
	};
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
	agentic?: ChatMessageAgenticTimings;
}

export interface ChatMessageAgenticTimings {
	turns: number;
	toolCallsCount: number;
	toolsMs: number;
	toolCalls?: ChatMessageToolCallTiming[];
	perTurn?: ChatMessageAgenticTurnStats[];
	llm: {
		predicted_n: number;
		predicted_ms: number;
		prompt_n: number;
		prompt_ms: number;
	};
}

export interface ChatMessageAgenticTurnStats {
	turn: number;
	llm: {
		predicted_n: number;
		predicted_ms: number;
		prompt_n: number;
		prompt_ms: number;
	};
	toolCalls: ChatMessageToolCallTiming[];
	toolsMs: number;
}

export interface ChatMessageToolCallTiming {
	name: string;
	duration_ms: number;
	success: boolean;
}

/**
 * Callbacks for streaming chat responses (used by both agentic and non-agentic paths)
 */
export interface ChatStreamCallbacks {
	onChunk?: (chunk: string) => void;
	onReasoningChunk?: (chunk: string) => void;
	onToolCallsStreaming?: (toolCalls: ApiChatCompletionToolCall[]) => void;
	onAttachments?: (messageId: string, extras: DatabaseMessageExtra[]) => void;
	onModel?: (model: string) => void;
	onCompletionId?: (id: string) => void;
	onTimings?: (timings?: ChatMessageTimings, promptProgress?: ChatMessagePromptProgress) => void;
	onAssistantTurnComplete?: (
		content: string,
		reasoningContent: string | undefined,
		timings: ChatMessageTimings | undefined,
		toolCalls: ApiChatCompletionToolCall[] | undefined
	) => Promise<void>;
	createToolResultMessage?: (
		toolCallId: string,
		content: string,
		extras?: DatabaseMessageExtra[],
		toolCwd?: string
	) => Promise<DatabaseMessage>;
	updateToolResultMessage?: (
		messageId: string,
		content: string,
		extras?: DatabaseMessageExtra[]
	) => Promise<void>;
	createAssistantMessage?: () => Promise<DatabaseMessage>;
	onFlowComplete?: (timings?: ChatMessageTimings) => void;
	onError?: (error: Error) => void;
	onTurnComplete?: (intermediateTimings: ChatMessageTimings) => void;
}

/**
 * Error dialog state for displaying server/timeout errors
 */
export interface ErrorDialogState {
	type: ErrorDialogType;
	message: string;
	contextInfo?: { n_prompt_tokens: number; n_ctx: number };
}

/**
 * Live processing stats during prompt evaluation
 */
export interface LiveProcessingStats {
	tokensProcessed: number;
	totalTokens: number;
	timeMs: number;
	tokensPerSecond: number;
	etaSecs?: number;
}

/**
 * Live generation stats during token generation
 */
export interface LiveGenerationStats {
	tokensGenerated: number;
	timeMs: number;
	tokensPerSecond: number;
}

/**
 * Options for getting attachment display items
 */
export interface AttachmentDisplayItemsOptions {
	uploadedFiles?: ChatUploadedFile[];
	attachments?: DatabaseMessageExtra[];
}

/**
 * Result of file processing operation
 */
export interface FileProcessingResult {
	extras: DatabaseMessageExtra[];
	emptyFiles: string[];
}

/**
 * A file or folder picked in the @-mention picker. `path` is the absolute
 * server-side path; `name` is the basename.
 */
export interface FileMentionEntry {
	path: string;
	name: string;
	type: FileMentionEntryType;
}

/**
 * A slash command surfaced by the `/` command picker. `disabled` marks a
 * command whose backing capability is unavailable (e.g. `/prompt` when no
 * MCP server exposes prompts): visible but greyed out and not selectable.
 */
export interface ChatCommandsOptions {
	/** Gates `/model`. */
	showModelSelector: boolean;
	/** Gates `/prompt`. */
	hasPrompts: () => boolean;
	/** Gates `/cwd`. */
	hasCwdTools: () => boolean;
}

/** Protocol-level verbs accepted by the realtime inference control endpoint. Mirrors `CONTROL_ACTION`. */
export type ControlAction = 'reasoning_end';

export interface ChatFormCommand {
	name: string;
	description: string;
	/** Extra search terms that should match this command in the picker. */
	keywords?: string[];
	action: ChatFormCommandAction;
	disabled: boolean;
}
