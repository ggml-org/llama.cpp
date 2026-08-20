import type {
	ApiChatCompletionRequest,
	ApiChatCompletionToolCall,
	ApiChatMessageContentPart,
	ApiChatMessageData
} from './api';
import type {
	ChatMessageAgenticTimings,
	ChatMessagePromptProgress,
	ChatMessageTimings
} from './chat';
import type {
	DatabaseMessage,
	DatabaseMessageExtra,
	DatabaseMessageExtraAudioFile,
	DatabaseMessageExtraImageFile
} from './database';
import type { MessageRole } from '$lib/enums';
import { AgenticSectionType, ContinueIntentKind, ToolCallType } from '$lib/enums';

/**
 * Agentic orchestration configuration.
 */
export interface AgenticConfig {
	enabled: boolean;
	maxTurns: number;
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
	/** ID of the tool call whose output is currently being streamed back
	 *  (e.g. exec_shell_command outputting to /tools?stream=true). Lets the
	 *  matching tool renderer flip into live-update mode while chunks
	 *  arrive; cleared when the tool's terminal event lands. */
	executingToolCallId: string | null;
	/** Live LLM token totals of the running flow: completed turns plus the
	 *  in-flight turn's streamed counts; null when idle. */
	liveLlm: ChatMessageAgenticTimings['llm'] | null;
	/** ID of the flow's first assistant message (the one the UI groups the
	 *  whole run under); null when idle. */
	flowRootMessageId: string | null;
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
	onCompletionId?: (id: string) => void;
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
		extras?: DatabaseMessageExtra[],
		toolCwd?: string
	) => Promise<DatabaseMessage>;
	/** Update an already-created tool result message. Used while a streaming
	 *  tool (e.g. exec_shell_command) accumulates output chunks before its
	 *  terminal event; the same message is rewritten in place so the chat UI
	 *  sees the partial output live. */
	updateToolResultMessage?: (
		messageId: string,
		content: string,
		extras?: DatabaseMessageExtra[]
	) => Promise<void>;
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
	/** ID of the flow's first assistant message, used to keep its stats live */
	flowRootMessageId?: string;
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

/**
 * Represents a parsed section of agentic content for display
 */
export interface AgenticSection {
	type: AgenticSectionType;
	content: string;
	toolName?: string;
	toolArgs?: string;
	toolResult?: string;
	toolResultExtras?: DatabaseMessageExtra[];
	/** Working directory the tool call ran with (from the tool result
	 *  message), shown by the exec_shell_command renderer. */
	toolCwd?: string;
	/** ID of the model-side tool call (matches tool_calls[i].id). Lets
	 *  downstream consumers correlate a section with the agentic loop's
	 *  currently-executing tool, e.g. to drive live-streaming UI state
	 *  by matching against agenticStore.getExecutingToolCallId. */
	toolCallId?: string;
	wasInterrupted?: boolean;
}

/**
 * Represents a tool result line that may reference an image attachment
 */
export type ToolResultLine = {
	text: string;
	media?: DatabaseMessageExtraImageFile | DatabaseMessageExtraAudioFile;
};

/**
 * Classification of how a Continue click on an assistant message should resume
 * generation. The caller dispatches the resume path based on this value.
 *
 *   append_text  -> the target is a plain text turn, resume with
 *                   continue_final_message and rehydrate the persisted
 *                   tool_calls and attachments through the regular DB to API
 *                   message converter.
 *   rerun_turn   -> the target carries tool_calls that were never resolved by
 *                   tool result messages. The agentic stream was cut mid turn,
 *                   so we drop the target and rerun the loop from the previous
 *                   history. truncateAfter is the last kept index, inclusive.
 *   next_turn    -> the target's tool_calls were already resolved by trailing
 *                   tool results. Hand the history up to and including the
 *                   last consecutive tool result back to the agentic loop so it
 *                   starts the next turn naturally. truncateAfter points at
 *                   that last tool result.
 */
export type ContinueIntent =
	| { kind: ContinueIntentKind.APPEND_TEXT }
	| { kind: ContinueIntentKind.RERUN_TURN; truncateAfter: number }
	| { kind: ContinueIntentKind.NEXT_TURN; truncateAfter: number };
