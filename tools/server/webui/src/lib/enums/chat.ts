export enum ChatMessageStatsView {
	GENERATION = 'generation',
	READING = 'reading',
	TOOLS = 'tools',
	SUMMARY = 'summary'
}

/**
 * Message roles for chat messages.
 */
export enum MessageRole {
	USER = 'user',
	ASSISTANT = 'assistant',
	SYSTEM = 'system'
}

/**
 * Message types for different content kinds.
 */
export enum MessageType {
	ROOT = 'root',
	TEXT = 'text',
	THINK = 'think',
	SYSTEM = 'system'
}
