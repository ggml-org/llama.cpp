/**
 * Attachment type enum for database message extras
 */
export enum AttachmentType {
	AUDIO = 'AUDIO',
	IMAGE = 'IMAGE',
	MCP_PROMPT = 'MCP_PROMPT',
	MCP_RESOURCE = 'MCP_RESOURCE',
	PDF = 'PDF',
	TEXT = 'TEXT',
	LEGACY_CONTEXT = 'context' // Legacy attachment type for backward compatibility
}

/**
 * Unique identifiers for attachment menu items in the chat form action dropdowns.
 * Used to select which file upload or attachment action is triggered.
 */
export enum AttachmentMenuItemId {
	IMAGES = 'images',
	AUDIO = 'audio',
	TEXT = 'text',
	PDF = 'pdf',
	SYSTEM_MESSAGE = 'system-message',
	MCP_PROMPT = 'mcp-prompt',
	MCP_RESOURCES = 'mcp-resources'
}

/**
 * Defines when an attachment menu item should be enabled.
 */
export enum AttachmentItemEnabledWhen {
	ALWAYS = 'always',
	HAS_VISION_MODALITY = 'hasVisionModality',
	HAS_AUDIO_MODALITY = 'hasAudioModality'
}

/**
 * Defines the callback action triggered when an attachment menu item is clicked.
 */
export enum AttachmentAction {
	FILE_UPLOAD = 'onFileUpload',
	SYSTEM_PROMPT_CLICK = 'onSystemPromptClick',
	MCP_PROMPT_CLICK = 'onMcpPromptClick',
	MCP_RESOURCES_CLICK = 'onMcpResourcesClick'
}

/**
 * Visibility conditions for attachment menu items.
 */
export enum AttachmentItemVisibleWhen {
	HAS_MCP_PROMPTS_SUPPORT = 'hasMcpPromptsSupport',
	HAS_MCP_RESOURCES_SUPPORT = 'hasMcpResourcesSupport'
}
