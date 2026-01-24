/**
 * Attachment type enum for database message extras
 */
export enum AttachmentType {
	AUDIO = 'AUDIO',
	IMAGE = 'IMAGE',
	MCP_PROMPT = 'MCP_PROMPT',
	PDF = 'PDF',
	TEXT = 'TEXT',
	LEGACY_CONTEXT = 'context' // Legacy attachment type for backward compatibility
}
