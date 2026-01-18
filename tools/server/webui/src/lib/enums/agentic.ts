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
