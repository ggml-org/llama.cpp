import { extractRootDomain } from './url';
import {
	AlertTriangle,
	Code,
	Database,
	File,
	FileText,
	Image,
	Info,
	XCircle
} from '@lucide/svelte';
import {
	CODE_FILE_EXTENSION_REGEX,
	DEFAULT_RESOURCE_FILENAME,
	DISPLAY_NAME_SEPARATOR_REGEX,
	EXPECTED_THEMED_ICON_PAIR_COUNT,
	FILE_EXTENSION_REGEX,
	IMAGE_FILE_EXTENSION_REGEX,
	MCP_ALLOWED_ICON_MIME_TYPES,
	MCP_SERVER_ID_PREFIX,
	MCP_SSE,
	MIME_TYPE_PREFIXES,
	MIME_TYPE_SUBSTRINGS,
	PATH_SEPARATOR,
	PROTOCOL_PREFIX_REGEX,
	RESOURCE_TEXT_CONTENT_SEPARATOR,
	TEXT_FILE_EXTENSION_REGEX,
	URI_PATTERNS
} from '$lib/constants';
import {
	ColorMode,
	HealthCheckStatus,
	MCPLogLevel,
	MCPTransportType,
	MimeTypeText,
	UrlProtocol
} from '$lib/enums';
import type {
	HealthCheckState,
	MCPResourceContent,
	MCPResourceIcon,
	MCPResourceInfo,
	MCPServerDisplayInfo,
	MCPServerSettingsEntry
} from '$lib/types';
import type { MimeTypeUnion } from '$lib/types/common';
import type { Component } from 'svelte';

/**
 * Detects the MCP transport type from a URL.
 * WebSocket URLs (ws:// or wss://) use 'websocket', others use 'streamable_http'.
 */
export function detectMcpTransportFromUrl(url: string): MCPTransportType {
	const normalized = url.trim().toLowerCase();

	if (
		normalized.startsWith(UrlProtocol.WEBSOCKET) ||
		normalized.startsWith(UrlProtocol.WEBSOCKET_SECURE)
	) {
		return MCPTransportType.WEBSOCKET;
	}

	if (
		normalized.endsWith(MCP_SSE.ENDPOINT) ||
		normalized.endsWith(MCP_SSE.ENDPOINT_SLASH) ||
		normalized.includes(MCP_SSE.ENDPOINT_QUERY)
	) {
		return MCPTransportType.SSE;
	}

	return MCPTransportType.STREAMABLE_HTTP;
}

/**
 * Parses MCP server settings from a JSON string or array.
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
			displayName: (entry as { displayName?: string })?.displayName,
			enabled: Boolean((entry as { enabled?: unknown })?.enabled),
			headers: headers || undefined,
			id,
			name: (entry as { name?: string })?.name,
			url,
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
	return mimeType?.startsWith(MIME_TYPE_PREFIXES.IMAGE) ?? false;
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

		return withoutProtocol.split(PATH_SEPARATOR).filter((p) => p.length > 0);
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
		mime.includes(MIME_TYPE_SUBSTRINGS.JSON) ||
		mime.includes(MIME_TYPE_SUBSTRINGS.JAVASCRIPT) ||
		mime.includes(MIME_TYPE_SUBSTRINGS.TYPESCRIPT) ||
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

	return mime.startsWith(MIME_TYPE_PREFIXES.IMAGE) || IMAGE_FILE_EXTENSION_REGEX.test(u);
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

	if (mime.startsWith(MIME_TYPE_PREFIXES.IMAGE) || IMAGE_FILE_EXTENSION_REGEX.test(u)) {
		return Image;
	}

	if (
		mime.includes(MIME_TYPE_SUBSTRINGS.JSON) ||
		mime.includes(MIME_TYPE_SUBSTRINGS.JAVASCRIPT) ||
		mime.includes(MIME_TYPE_SUBSTRINGS.TYPESCRIPT) ||
		CODE_FILE_EXTENSION_REGEX.test(u)
	) {
		return Code;
	}

	if (mime.includes(MIME_TYPE_PREFIXES.TEXT) || TEXT_FILE_EXTENSION_REGEX.test(u)) {
		return FileText;
	}

	if (u.includes(URI_PATTERNS.DATABASE_KEYWORD) || u.includes(URI_PATTERNS.DATABASE_SCHEME)) {
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
		.join(RESOURCE_TEXT_CONTENT_SEPARATOR);
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
	filename: string = DEFAULT_RESOURCE_FILENAME
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

/**
 * Validates that an icon URI uses a safe scheme (https: or data:).
 */
function isValidMcpIconUri(src: string): boolean {
	try {
		if (src.startsWith(UrlProtocol.DATA)) return true;

		const url = new URL(src);

		return url.protocol === UrlProtocol.HTTPS;
	} catch {
		return false;
	}
}

/**
 * Selects the best icon URL from an MCP icons array.
 * Follows security guidelines from the MCP specification:
 * - Only allows https: and data: URIs
 * - Filters to supported MIME types
 *
 * Selection priority:
 * 1. Icon matching the current color scheme (dark/light)
 * 2. Universal icon (no theme specified); if exactly 2, assumes [0]=light, [1]=dark
 * 3. First valid icon as last resort
 */
export function getMcpIconUrl(icons: MCPResourceIcon[] | undefined, isDark = false): string | null {
	if (!icons?.length) return null;

	const validIcons = icons.filter((icon) => {
		if (!icon.src || !isValidMcpIconUri(icon.src)) return false;

		if (icon.mimeType && !MCP_ALLOWED_ICON_MIME_TYPES.has(icon.mimeType)) return false;

		return true;
	});

	if (validIcons.length === 0) return null;

	const preferredTheme = isDark ? ColorMode.DARK : ColorMode.LIGHT;
	// 1. Prefer icon explicitly matching the current color scheme
	const themedIcon = validIcons.find((icon) => icon.theme === preferredTheme);

	if (themedIcon) return themedIcon.src;

	// 2. Handle universal icons (no theme specified)
	const universalIcons = validIcons.filter((icon) => !icon.theme);

	if (universalIcons.length === EXPECTED_THEMED_ICON_PAIR_COUNT) {
		// Heuristic: two theme-less icons → assume [0] = light, [1] = dark
		return universalIcons[isDark ? 1 : 0].src;
	}

	if (universalIcons.length > 0) {
		return universalIcons[0].src;
	}

	// 3. Last resort: use opposite-theme icon
	return validIcons[0].src;
}

/**
 * Construct a fallback favicon URL from the MCP server URL.
 * e.g. https://mcp.example.com/sse -> https://example.com/favicon.ico
 */
export function getMcpServerFaviconFallback(serverUrl: string): string | null {
	try {
		const url = new URL(serverUrl);
		const rootDomain = extractRootDomain(url);

		if (!rootDomain) return null;

		const origin = `${url.protocol}//${rootDomain}`;
		const candidates = ['favicon.ico', 'favicon.png'];

		for (const path of candidates) {
			const faviconUrl = `${origin}/${path}`;

			if (isValidMcpIconUri(faviconUrl)) {
				return faviconUrl;
			}
		}
	} catch {
		// Invalid URL, return null
	}

	return null;
}

/**
 * Resolves the raw label for a server: user-defined display name first,
 * then server-reported title or name when the health check succeeded,
 * then the configured name (admin baseline or legacy data), then URL.
 */
function getMcpServerBaseLabel(
	server: MCPServerDisplayInfo,
	healthState?: HealthCheckState
): string {
	if (server.displayName) return server.displayName;

	if (healthState?.status === HealthCheckStatus.SUCCESS)
		return (
			healthState.serverInfo?.title || healthState.serverInfo?.name || server.name || server.url
		);

	return server.name || server.url;
}

/**
 * Returns the display label for a server, suffixed with a positional
 * counter when several configured servers resolve to the same base label
 * (e.g. two endpoints of the same host reporting an identical name).
 * Numbering follows config order, so it is stable across renders.
 */
export function getMcpServerLabel(
	server: MCPServerDisplayInfo,
	servers: MCPServerDisplayInfo[],
	healthChecks: Record<string, HealthCheckState>
): string {
	const label = getMcpServerBaseLabel(server, healthChecks[server.id]);
	const twins = servers.filter((s) => getMcpServerBaseLabel(s, healthChecks[s.id]) === label);

	if (twins.length < 2) return label;

	const position = twins.findIndex((s) => s.id === server.id);

	return position < 0 ? label : `${label} (${position + 1})`;
}
