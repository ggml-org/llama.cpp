import type { MCPServerSettingsEntry, MCPResourceContent, MCPResourceInfo } from '$lib/types';
import {
	MCPTransportType,
	MCPLogLevel,
	UrlPrefix,
	MimeTypePrefix,
	MimeTypeIncludes,
	UriPattern,
	MimeTypeText
} from '$lib/enums';
import { DEFAULT_MCP_CONFIG, MCP_SERVER_ID_PREFIX } from '$lib/constants/mcp';
import {
	IMAGE_FILE_EXTENSION_REGEX,
	CODE_FILE_EXTENSION_REGEX,
	TEXT_FILE_EXTENSION_REGEX,
	PROTOCOL_PREFIX_REGEX,
	FILE_EXTENSION_REGEX,
	DISPLAY_NAME_SEPARATOR_REGEX
} from '$lib/constants/mcp-resource';
import {
	Database,
	File,
	FileText,
	Image,
	Code,
	Info,
	AlertTriangle,
	XCircle
} from '@lucide/svelte';
import type { Component } from 'svelte';
import type { MimeTypeUnion } from '$lib/types/common';

/**
 * Detects the MCP transport type from a URL.
 * WebSocket URLs (ws:// or wss://) use 'websocket', others use 'streamable_http'.
 */
export function detectMcpTransportFromUrl(url: string): MCPTransportType {
	const normalized = url.trim().toLowerCase();

	return normalized.startsWith(UrlPrefix.WEBSOCKET) ||
		normalized.startsWith(UrlPrefix.WEBSOCKET_SECURE)
		? MCPTransportType.WEBSOCKET
		: MCPTransportType.STREAMABLE_HTTP;
}

/**
 * Parses MCP server settings from a JSON string or array.
 * requestTimeoutSeconds is not user-configurable in the UI, so we always use the default value.
 * @param rawServers - The raw servers to parse
 * @returns An empty array if the input is invalid.
 */
export function parseMcpServerSettings(rawServers: unknown): MCPServerSettingsEntry[] {
	if (!rawServers) return [];

	let parsed: unknown;

	if (typeof rawServers === 'string') {
		const trimmed = rawServers.trim();
		if (!trimmed) return [];

		try {
			parsed = JSON.parse(trimmed);
		} catch (error) {
			console.warn('[MCP] Failed to parse mcpServers JSON, ignoring value:', error);

			return [];
		}
	} else {
		parsed = rawServers;
	}

	if (!Array.isArray(parsed)) return [];

	return parsed.map((entry, index) => {
		const url = typeof entry?.url === 'string' ? entry.url.trim() : '';
		const headers = typeof entry?.headers === 'string' ? entry.headers.trim() : undefined;
		const id =
			typeof (entry as { id?: unknown })?.id === 'string' && (entry as { id?: string }).id?.trim()
				? (entry as { id: string }).id.trim()
				: `${MCP_SERVER_ID_PREFIX}-${index + 1}`;

		return {
			id,
			enabled: Boolean((entry as { enabled?: unknown })?.enabled),
			url,
			name: (entry as { name?: string })?.name,
			requestTimeoutSeconds: DEFAULT_MCP_CONFIG.requestTimeoutSeconds,
			headers: headers || undefined,
			useProxy: Boolean((entry as { useProxy?: unknown })?.useProxy)
		} satisfies MCPServerSettingsEntry;
	});
}

/**
 * Get the appropriate icon component for a log level
 *
 * @param level - MCP log level
 * @returns Lucide icon component
 */
export function getMcpLogLevelIcon(level: MCPLogLevel): Component {
	switch (level) {
		case MCPLogLevel.ERROR:
			return XCircle;
		case MCPLogLevel.WARN:
			return AlertTriangle;
		default:
			return Info;
	}
}

/**
 * Get the appropriate CSS class for a log level
 *
 * @param level - MCP log level
 * @returns Tailwind CSS class string
 */
export function getMcpLogLevelClass(level: MCPLogLevel): string {
	switch (level) {
		case MCPLogLevel.ERROR:
			return 'text-destructive';
		case MCPLogLevel.WARN:
			return 'text-yellow-600 dark:text-yellow-500';
		default:
			return 'text-muted-foreground';
	}
}

/**
 * Check if a MIME type represents an image.
 *
 * @param mimeType - The MIME type to check
 * @returns True if the MIME type starts with 'image/'
 */
export function isImageMimeType(mimeType?: MimeTypeUnion): boolean {
	return mimeType?.startsWith(MimeTypePrefix.IMAGE) ?? false;
}

/**
 * Parse a resource URI into path segments, stripping the protocol prefix.
 *
 * @param uri - The resource URI to parse
 * @returns Array of non-empty path segments
 */
export function parseResourcePath(uri: string): string[] {
	try {
		const withoutProtocol = uri.replace(PROTOCOL_PREFIX_REGEX, '');
		return withoutProtocol.split('/').filter((p) => p.length > 0);
	} catch {
		return [uri];
	}
}

/**
 * Convert a path part into a human-readable display name.
 * Strips file extensions and converts kebab-case/snake_case to Title Case.
 *
 * @param pathPart - The path segment to convert
 * @returns Human-readable display name
 */
export function getDisplayName(pathPart: string): string {
	const withoutExt = pathPart.replace(FILE_EXTENSION_REGEX, '');
	return withoutExt
		.split(DISPLAY_NAME_SEPARATOR_REGEX)
		.map((word) => word.charAt(0).toUpperCase() + word.slice(1))
		.join(' ');
}

/**
 * Get the display name from a resource, extracting the last path segment from the URI.
 *
 * @param resource - The MCP resource info
 * @returns Display name string
 */
export function getResourceDisplayName(resource: MCPResourceInfo): string {
	try {
		const parts = parseResourcePath(resource.uri);
		return parts[parts.length - 1] || resource.name || resource.uri;
	} catch {
		return resource.name || resource.uri;
	}
}

/**
 * Determine if a MIME type and/or URI represents code content.
 *
 * @param mimeType - Optional MIME type string
 * @param uri - Optional URI string
 * @returns True if the content is code
 */
export function isCodeResource(mimeType?: MimeTypeUnion, uri?: string): boolean {
	const mime = mimeType?.toLowerCase() || '';
	const u = uri?.toLowerCase() || '';
	return (
		mime.includes(MimeTypeIncludes.JSON) ||
		mime.includes(MimeTypeIncludes.JAVASCRIPT) ||
		mime.includes(MimeTypeIncludes.TYPESCRIPT) ||
		CODE_FILE_EXTENSION_REGEX.test(u)
	);
}

/**
 * Determine if a MIME type and/or URI represents image content.
 *
 * @param mimeType - Optional MIME type string
 * @param uri - Optional URI string
 * @returns True if the content is an image
 */
export function isImageResource(mimeType?: MimeTypeUnion, uri?: string): boolean {
	const mime = mimeType?.toLowerCase() || '';
	const u = uri?.toLowerCase() || '';
	return mime.startsWith(MimeTypePrefix.IMAGE) || IMAGE_FILE_EXTENSION_REGEX.test(u);
}

/**
 * Get the appropriate Lucide icon component for an MCP resource based on its MIME type and URI.
 *
 * @param mimeType - Optional MIME type of the resource
 * @param uri - Optional URI of the resource
 * @returns Lucide icon component
 */
export function getResourceIcon(mimeType?: MimeTypeUnion, uri?: string): Component {
	const mime = mimeType?.toLowerCase() || '';
	const u = uri?.toLowerCase() || '';

	if (mime.startsWith(MimeTypePrefix.IMAGE) || IMAGE_FILE_EXTENSION_REGEX.test(u)) {
		return Image;
	}

	if (
		mime.includes(MimeTypeIncludes.JSON) ||
		mime.includes(MimeTypeIncludes.JAVASCRIPT) ||
		mime.includes(MimeTypeIncludes.TYPESCRIPT) ||
		CODE_FILE_EXTENSION_REGEX.test(u)
	) {
		return Code;
	}

	if (mime.includes(MimeTypePrefix.TEXT) || TEXT_FILE_EXTENSION_REGEX.test(u)) {
		return FileText;
	}

	if (u.includes(UriPattern.DATABASE_KEYWORD) || u.includes(UriPattern.DATABASE_SCHEME)) {
		return Database;
	}

	return File;
}

/**
 * Extract text content from MCP resource content array.
 *
 * @param content - Array of MCP resource content items
 * @returns Joined text content string
 */
export function getResourceTextContent(content: MCPResourceContent[] | null | undefined): string {
	if (!content) return '';
	return content
		.filter((c): c is { uri: string; mimeType?: MimeTypeUnion; text: string } => 'text' in c)
		.map((c) => c.text)
		.join('\n\n');
}

/**
 * Extract blob content from MCP resource content array.
 *
 * @param content - Array of MCP resource content items
 * @returns Array of blob content items
 */
export function getResourceBlobContent(
	content: MCPResourceContent[] | null | undefined
): Array<{ uri: string; mimeType?: MimeTypeUnion; blob: string }> {
	if (!content) return [];

	return content.filter(
		(c): c is { uri: string; mimeType?: MimeTypeUnion; blob: string } => 'blob' in c
	);
}

/**
 * Trigger a file download from text content.
 *
 * @param text - The text content to download
 * @param mimeType - MIME type for the blob
 * @param filename - Suggested filename
 */
export function downloadResourceContent(
	text: string,
	mimeType: MimeTypeUnion = MimeTypeText.PLAIN,
	filename: string = 'resource.txt'
): void {
	const blob = new Blob([text], { type: mimeType });
	const url = URL.createObjectURL(blob);
	const a = document.createElement('a');
	a.href = url;
	a.download = filename;
	document.body.appendChild(a);
	a.click();
	document.body.removeChild(a);
	URL.revokeObjectURL(url);
}
