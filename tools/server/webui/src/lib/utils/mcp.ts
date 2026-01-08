import type { MCPTransportType } from '$lib/types/mcp';

/**
 * Represents a key-value pair for HTTP headers.
 */
export type HeaderPair = { key: string; value: string };

/**
 * Detects the MCP transport type from a URL.
 * WebSocket URLs (ws:// or wss://) use 'websocket', others use 'streamable_http'.
 */
export function detectMcpTransportFromUrl(url: string): MCPTransportType {
	const normalized = url.trim().toLowerCase();
	return normalized.startsWith('ws://') || normalized.startsWith('wss://')
		? 'websocket'
		: 'streamable_http';
}

/**
 * Generates a valid MCP server ID from user input.
 * Returns the trimmed ID if valid, otherwise generates 'server-{index+1}'.
 */
export function generateMcpServerId(id: unknown, index: number): string {
	if (typeof id === 'string' && id.trim()) {
		return id.trim();
	}
	return `server-${index + 1}`;
}

/**
 * Extracts a human-readable server name from a URL.
 * Strips common prefixes like 'www.' and 'mcp.' and capitalizes the result.
 */
export function extractServerNameFromUrl(url: string): string {
	try {
		const parsedUrl = new URL(url);
		const host = parsedUrl.hostname.replace(/^(www\.|mcp\.)/, '');
		const name = host.split('.')[0] || 'Unknown';
		return name.charAt(0).toUpperCase() + name.slice(1);
	} catch {
		return 'New Server';
	}
}

/**
 * Gets a favicon URL for an MCP server using Google's favicon service.
 * Returns null if the URL is invalid.
 */
export function getFaviconUrl(serverUrl: string): string | null {
	try {
		const parsedUrl = new URL(serverUrl);
		const hostnameParts = parsedUrl.hostname.split('.');
		const rootDomain =
			hostnameParts.length >= 2 ? hostnameParts.slice(-2).join('.') : parsedUrl.hostname;
		return `https://www.google.com/s2/favicons?domain=${rootDomain}&sz=32`;
	} catch {
		return null;
	}
}

/**
 * Parses a JSON string of headers into an array of key-value pairs.
 * Returns empty array if the JSON is invalid or empty.
 */
export function parseHeadersToArray(headersJson: string): HeaderPair[] {
	if (!headersJson?.trim()) return [];
	try {
		const parsed = JSON.parse(headersJson);
		if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
			return Object.entries(parsed).map(([key, value]) => ({
				key,
				value: String(value)
			}));
		}
	} catch {
		// Invalid JSON, return empty
	}
	return [];
}

/**
 * Serializes an array of header key-value pairs to a JSON string.
 * Filters out pairs with empty keys and returns empty string if no valid pairs.
 */
export function serializeHeaders(pairs: HeaderPair[]): string {
	const validPairs = pairs.filter((p) => p.key.trim());
	if (validPairs.length === 0) return '';
	const obj: Record<string, string> = {};
	for (const pair of validPairs) {
		obj[pair.key.trim()] = pair.value;
	}
	return JSON.stringify(obj);
}
