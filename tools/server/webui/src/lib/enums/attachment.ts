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
