/**
 * Attachment type enum for database message extras
 */
export enum AttachmentType {
	AUDIO = 'AUDIO',
	IMAGE = 'IMAGE',
	LEGACY_CONTEXT = 'context', // Legacy attachment type for backward compatibility
	MCP_PROMPT = 'MCP_PROMPT',
	MCP_RESOURCE = 'MCP_RESOURCE',
	PDF = 'PDF',
	TEXT = 'TEXT',
	VIDEO = 'VIDEO'
}

/**
 * Unique identifiers for attachment menu items in the chat form action dropdowns.
 * Used to select which file upload or attachment action is triggered.
 */
export enum AttachmentMenuItemId {
	AUDIO = 'audio',
	IMAGES = 'images',
	MCP_PROMPT = 'mcp-prompt',
	MCP_RESOURCES = 'mcp-resources',
	PDF = 'pdf',
	SYSTEM_MESSAGE = 'system-message',
	TEXT = 'text',
	VIDEO = 'video'
}

/**
 * Defines when an attachment menu item should be enabled.
 */
export enum AttachmentItemEnabledWhen {
	ALWAYS = 'always',
	HAS_AUDIO_MODALITY = 'hasAudioModality',
	HAS_VIDEO_MODALITY = 'hasVideoModality',
	HAS_VISION_MODALITY = 'hasVisionModality'
}

/**
 * Defines the callback action triggered when an attachment menu item is clicked.
 */
export enum AttachmentAction {
	FILE_UPLOAD = 'onFileUpload',
	MCP_PROMPT_CLICK = 'onMcpPromptClick',
	MCP_RESOURCES_CLICK = 'onMcpResourcesClick',
	SYSTEM_PROMPT_CLICK = 'onSystemPromptClick'
}

/**
 * Human-readable labels used when embedding attachments in outgoing messages.
 */
export enum AttachmentLabel {
	FILE = 'File',
	MCP_PROMPT = 'MCP Prompt',
	MCP_RESOURCE = 'MCP Resource',
	PDF_FILE = 'PDF File'
}

/**
 * Visibility conditions for attachment menu items.
 */
export enum AttachmentItemVisibleWhen {
	HAS_MCP_PROMPTS_SUPPORT = 'hasMcpPromptsSupport',
	HAS_MCP_RESOURCES_SUPPORT = 'hasMcpResourcesSupport'
}
