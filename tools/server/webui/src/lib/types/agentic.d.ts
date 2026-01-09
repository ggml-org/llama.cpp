import type { ApiChatCompletionRequest, ApiChatMessageContentPart } from './api';

/**
 * Agentic orchestration configuration.
 */
export interface AgenticConfig {
	enabled: boolean;
	maxTurns: number;
	maxToolPreviewLines: number;
	filterReasoningAfterFirstTurn: boolean;
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
			role: 'system' | 'user';
			content: string | ApiChatMessageContentPart[];
	  }
	| {
			role: 'assistant';
			content?: string | ApiChatMessageContentPart[];
			tool_calls?: AgenticToolCallPayload[];
	  }
	| {
			role: 'tool';
			tool_call_id: string;
			content: string;
	  };

export type AgenticAssistantMessage = Extract<AgenticMessage, { role: 'assistant' }>;
export type AgenticToolCallList = NonNullable<AgenticAssistantMessage['tool_calls']>;

export type AgenticChatCompletionRequest = Omit<ApiChatCompletionRequest, 'messages'> & {
	messages: AgenticMessage[];
	stream: true;
	tools?: ApiChatCompletionRequest['tools'];
};
