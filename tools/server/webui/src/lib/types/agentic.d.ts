import type { MessageRole } from '$lib/enums';
import type { ApiChatCompletionRequest, ApiChatMessageContentPart } from './api';

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
