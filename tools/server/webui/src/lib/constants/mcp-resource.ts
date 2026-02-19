import { MimeTypeImage } from '$lib/enums';

// File extension patterns for resource type detection
export const IMAGE_FILE_EXTENSION_REGEX = /\.(png|jpg|jpeg|gif|svg|webp)$/i;
export const CODE_FILE_EXTENSION_REGEX =
	/\.(js|ts|json|yaml|yml|xml|html|css|py|rs|go|java|cpp|c|h|rb|sh|toml)$/i;
export const TEXT_FILE_EXTENSION_REGEX = /\.(txt|md|log)$/i;

// URI protocol prefix pattern
export const PROTOCOL_PREFIX_REGEX = /^[a-z]+:\/\//;

// File extension regex for display name extraction
export const FILE_EXTENSION_REGEX = /\.[^.]+$/;

// Separator regex for splitting display names (kebab-case/snake_case)
export const DISPLAY_NAME_SEPARATOR_REGEX = /[-_]/;

// Regex for matching base64-encoded data URIs
export const DATA_URI_BASE64_REGEX = /^data:([^;]+);base64,([A-Za-z0-9+/]+=*)$/;

// Prefix for MCP attachment filenames
export const MCP_ATTACHMENT_NAME_PREFIX = 'mcp-attachment';

// Prefix for MCP resource attachment IDs
export const MCP_RESOURCE_ATTACHMENT_ID_PREFIX = 'res';

// Default file extension for unknown image types
export const DEFAULT_IMAGE_EXTENSION = 'img';

/**
 * Mapping from image MIME types to file extensions.
 * Used for generating attachment filenames from MIME types.
 */
export const IMAGE_MIME_TO_EXTENSION: Record<string, string> = {
	[MimeTypeImage.JPEG]: 'jpg',
	[MimeTypeImage.JPG]: 'jpg',
	[MimeTypeImage.PNG]: 'png',
	[MimeTypeImage.GIF]: 'gif',
	[MimeTypeImage.WEBP]: 'webp'
} as const;
