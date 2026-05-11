import type { MessageRole } from '$lib/enums';
import { ToolCallType } from '$lib/enums';
import type {
	ApiChatCompletionRequest,
	ApiChatCompletionToolCall,
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
	type: ToolCallType.FUNCTION;
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
			reasoning_content?: string;
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
	pendingPermissionRequest: { toolName: string; serverLabel: string } | null;
}

/**
 * Callbacks for agentic flow execution.
 *
 * The agentic loop creates separate DB messages for each turn:
 * - assistant messages (one per LLM turn, with tool_calls if any)
 * - tool result messages (one per tool call execution)
 *
 * The first assistant message is created by the caller before starting the flow.
 * Subsequent messages are created via createToolResultMessage / createAssistantMessage.
 */
export interface AgenticFlowCallbacks {
	/** Content chunk for the current assistant message */
	onChunk?: (chunk: string) => void;
	/** Reasoning content chunk for the current assistant message */
	onReasoningChunk?: (chunk: string) => void;
	/** Tool calls being streamed (partial, accumulating) for the current turn */
	onToolCallsStreaming?: (toolCalls: ApiChatCompletionToolCall[]) => void;
	/** Attachments extracted from tool results */
	onAttachments?: (messageId: string, extras: DatabaseMessageExtra[]) => void;
	/** Model name detected from response */
	onModel?: (model: string) => void;
	/** Current assistant turn's streaming is complete - save to DB */
	onAssistantTurnComplete?: (
		content: string,
		reasoningContent: string | undefined,
		timings: ChatMessageTimings | undefined,
		toolCalls: ApiChatCompletionToolCall[] | undefined
	) => Promise<void>;
	/** Create a tool result message in the DB tree */
	createToolResultMessage?: (
		toolCallId: string,
		content: string,
		extras?: DatabaseMessageExtra[]
	) => Promise<DatabaseMessage>;
	/** Create a new assistant message for the next agentic turn */
	createAssistantMessage?: () => Promise<DatabaseMessage>;
	/** Entire agentic flow is complete */
	onFlowComplete?: (timings?: ChatMessageTimings) => void;
	/** Error during flow */
	onError?: (error: Error) => void;
	/** Timing updates during streaming */
	onTimings?: (timings?: ChatMessageTimings, promptProgress?: ChatMessagePromptProgress) => void;
	/** An agentic turn (LLM + tool execution) completed - intermediate timing update */
	onTurnComplete?: (intermediateTimings: ChatMessageTimings) => void;
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

/**
 * A user message to be injected into the agentic loop between turns.
 */
export interface SteeringMessage {
	content: string;
	extras?: DatabaseMessageExtra[];
}
