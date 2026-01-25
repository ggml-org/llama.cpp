import type { MessageRole } from '$lib/enums';
import type {
	ApiChatCompletionRequest,
	ApiChatMessageContentPart,
	ApiChatMessageData
} from './api';
import type { ChatMessageTimings, ChatMessagePromptProgress } from './chat';
import type { DatabaseMessage, DatabaseMessageExtra, McpServerOverride } from './database';

/**
 * Agentic orchestration configuration.
 */
export interface AgenticConfig {
	enabled: boolean;
	maxTurns: number;
	maxToolPreviewLines: number;
}

/**
 * Tool call payload for agentic messages.
 */
export type AgenticToolCallPayload = {
	id: string;
	type: 'function';
	function: {
		name: string;
		arguments: string;
	};
};

/**
 * Agentic message types for different roles.
 */
export type AgenticMessage =
	| {
			role: MessageRole.SYSTEM | MessageRole.USER;
			content: string | ApiChatMessageContentPart[];
	  }
	| {
			role: MessageRole.ASSISTANT;
			content?: string | ApiChatMessageContentPart[];
			tool_calls?: AgenticToolCallPayload[];
	  }
	| {
			role: MessageRole.TOOL;
			tool_call_id: string;
			content: string | ApiChatMessageContentPart[];
	  };

export type AgenticAssistantMessage = Extract<AgenticMessage, { role: MessageRole.ASSISTANT }>;
export type AgenticToolCallList = NonNullable<AgenticAssistantMessage['tool_calls']>;

export type AgenticChatCompletionRequest = Omit<ApiChatCompletionRequest, 'messages'> & {
	messages: AgenticMessage[];
	stream: true;
	tools?: ApiChatCompletionRequest['tools'];
};

/**
 * Per-conversation agentic session state.
 * Enables parallel agentic flows across multiple chats.
 */
export interface AgenticSession {
	isRunning: boolean;
	currentTurn: number;
	totalToolCalls: number;
	lastError: Error | null;
	streamingToolCall: { name: string; arguments: string } | null;
}

/**
 * Callbacks for agentic flow execution
 */
export interface AgenticFlowCallbacks {
	onChunk?: (chunk: string) => void;
	onReasoningChunk?: (chunk: string) => void;
	onToolCallChunk?: (serializedToolCalls: string) => void;
	onAttachments?: (extras: DatabaseMessageExtra[]) => void;
	onModel?: (model: string) => void;
	onComplete?: (
		content: string,
		reasoningContent?: string,
		timings?: ChatMessageTimings,
		toolCalls?: string
	) => void;
	onError?: (error: Error) => void;
	onTimings?: (timings?: ChatMessageTimings, promptProgress?: ChatMessagePromptProgress) => void;
}

/**
 * Options for agentic flow execution
 */
export interface AgenticFlowOptions {
	stream?: boolean;
	model?: string;
	temperature?: number;
	max_tokens?: number;
	[key: string]: unknown;
}

/**
 * Parameters for starting an agentic flow
 */
export interface AgenticFlowParams {
	conversationId: string;
	messages: (ApiChatMessageData | (DatabaseMessage & { extra?: DatabaseMessageExtra[] }))[];
	options?: AgenticFlowOptions;
	callbacks: AgenticFlowCallbacks;
	signal?: AbortSignal;
	perChatOverrides?: McpServerOverride[];
}

/**
 * Result of an agentic flow execution
 */
export interface AgenticFlowResult {
	handled: boolean;
	error?: Error;
}
