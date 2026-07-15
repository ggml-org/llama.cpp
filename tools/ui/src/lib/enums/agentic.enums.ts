/**
 * OpenAI-compatible tool call type.
 */
export enum ToolCallType {
	FUNCTION = 'function'
}

/**
 * Types of sections in agentic content display.
 */
export enum AgenticSectionType {
	TEXT = 'text',
	TOOL_CALL = 'tool_call',
	TOOL_CALL_PENDING = 'tool_call_pending',
	TOOL_CALL_STREAMING = 'tool_call_streaming',
	REASONING = 'reasoning',
	REASONING_PENDING = 'reasoning_pending'
}

/**
 * How a Continue click on an assistant message resumes generation.
 */
export enum ContinueIntentKind {
	APPEND_TEXT = 'append_text',
	RERUN_TURN = 'rerun_turn',
	NEXT_TURN = 'next_turn'
}

/**
 * Renderer tier for a tool-result blob shown in the default tool-call block.
 */
export enum ToolResultKind {
	JSON = 'json',
	MARKDOWN = 'markdown',
	TEXT = 'text'
}

/**
 * Line classification for the unified-diff renderer of `edit_file` results.
 */
export enum DiffLineKind {
	CONTEXT = 'context',
	ADD = 'add',
	REMOVE = 'remove'
}
