export enum ChatMessageStatsView {
	GENERATION = 'generation',
	READING = 'reading',
	SUMMARY = 'summary',
	TOOLS = 'tools'
}

export enum ChatMessageStatisticsMode {
	GENERATION = 'generation',
	READING = 'reading',
	SWITCHABLE = 'switchable'
}

/**
 * Connection state of a streamed completion, drives the resume status indicator.
 */
export enum StreamConnectionState {
	LOST = 'lost',
	RESUMING = 'resuming',
	STREAMING = 'streaming'
}

/**
 * Reasoning format options for API requests.
 */
export enum ReasoningFormat {
	AUTO = 'auto',
	NONE = 'none'
}

/**
 * Message roles for chat messages.
 */
export enum MessageRole {
	ASSISTANT = 'assistant',
	SYSTEM = 'system',
	TOOL = 'tool',
	USER = 'user'
}

/**
 * Message types for different content kinds.
 */
export enum MessageType {
	COMPACTION = 'compaction',
	ROOT = 'root',
	SYSTEM = 'system',
	TEXT = 'text',
	THINK = 'think'
}

/**
 * Content part types for API chat message content.
 */
export enum ContentPartType {
	IMAGE_URL = 'image_url',
	INPUT_AUDIO = 'input_audio',
	INPUT_VIDEO = 'input_video',
	TEXT = 'text'
}

/**
 * Error dialog types for displaying server/timeout errors.
 */
export enum ErrorDialogType {
	SERVER = 'server',
	TIMEOUT = 'timeout'
}

export enum ConversationSelectionMode {
	EXPORT = 'export',
	IMPORT = 'import'
}

/**
 * PDF view mode options for previewing PDF attachments.
 */
export enum PdfViewMode {
	PAGES = 'pages',
	TEXT = 'text'
}

export enum ChatFormCommandAction {
	COMPACT = 'compact',
	CWD = 'cwd',
	MODEL = 'model',
	PROMPT = 'prompt'
}

export enum FileMentionEntryType {
	DIRECTORY = 'directory',
	FILE = 'file'
}

/**
 * Kinds of tokens the chat-form-input-rich produces.
 */
export enum ChatFormInputRichTokenKind {
	BADGE = 'badge',
	CODE_BLOCK = 'code_block',
	CODE_INLINE = 'code_inline',
	TEXT = 'text'
}
