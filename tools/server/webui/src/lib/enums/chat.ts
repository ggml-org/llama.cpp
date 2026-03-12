export enum ChatMessageStatsView {
	GENERATION = 'generation',
	READING = 'reading',
	TOOLS = 'tools',
	SUMMARY = 'summary'
}

/**
 * Reasoning format options for API requests.
 */
export enum ReasoningFormat {
	NONE = 'none',
	AUTO = 'auto'
}

/**
 * Message roles for chat messages.
 */
export enum MessageRole {
	USER = 'user',
	ASSISTANT = 'assistant',
	SYSTEM = 'system',
	TOOL = 'tool'
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

/**
 * Content part types for API chat message content.
 */
export enum ContentPartType {
	TEXT = 'text',
	IMAGE_URL = 'image_url',
	INPUT_AUDIO = 'input_audio'
}

/**
 * Error dialog types for displaying server/timeout errors.
 */
export enum ErrorDialogType {
	TIMEOUT = 'timeout',
	SERVER = 'server'
}
