import type { MCPClientConfig, MCPServerConfig, MCPServerSettingsEntry } from '$lib/types';
import type { SettingsConfigType } from '$lib/types/settings';
import type { McpServerOverride } from '$lib/types/database';
import { MCPTransportType } from '$lib/enums';
import { DEFAULT_MCP_CONFIG } from '$lib/constants/mcp';
import { normalizePositiveNumber } from '$lib/utils/number';

/**
 * Detects the MCP transport type from a URL.
 * WebSocket URLs (ws:// or wss://) use 'websocket', others use 'streamable_http'.
 */
export function detectMcpTransportFromUrl(url: string): MCPTransportType {
	const normalized = url.trim().toLowerCase();

	return normalized.startsWith('ws://') || normalized.startsWith('wss://')
		? MCPTransportType.Websocket
		: MCPTransportType.StreamableHttp;
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
 * Gets a display name for an MCP server.
 * Returns server.name if set, otherwise extracts name from URL.
 */
export function getServerDisplayName(server: MCPServerSettingsEntry): string {
	if (server.name) return server.name;

	return extractServerNameFromUrl(server.url);
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
export function parseHeadersToArray(headersJson: string): { key: string; value: string }[] {
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
		return [];
	}

	return [];
}

/**
 * Serializes an array of header key-value pairs to a JSON string.
 * Filters out pairs with empty keys and returns empty string if no valid pairs.
 */
export function serializeHeaders(pairs: { key: string; value: string }[]): string {
	const validPairs = pairs.filter((p) => p.key.trim());

	if (validPairs.length === 0) return '';

	const obj: Record<string, string> = {};

	for (const pair of validPairs) {
		obj[pair.key.trim()] = pair.value;
	}

	return JSON.stringify(obj);
}

/**
 * Parses MCP server settings from a JSON string or array.
 * @param rawServers - The raw servers to parse
 * @param fallbackRequestTimeoutSeconds - The fallback request timeout seconds
 * @returns An empty array if the input is invalid.
 */
export function parseMcpServerSettings(
	rawServers: unknown,
	fallbackRequestTimeoutSeconds = DEFAULT_MCP_CONFIG.requestTimeoutSeconds
): MCPServerSettingsEntry[] {
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
		const requestTimeoutSeconds = normalizePositiveNumber(
			(entry as { requestTimeoutSeconds?: unknown })?.requestTimeoutSeconds,
			fallbackRequestTimeoutSeconds
		);

		const url = typeof entry?.url === 'string' ? entry.url.trim() : '';
		const headers = typeof entry?.headers === 'string' ? entry.headers.trim() : undefined;

		return {
			id: generateMcpServerId((entry as { id?: unknown })?.id, index),
			enabled: Boolean((entry as { enabled?: unknown })?.enabled),
			url,
			requestTimeoutSeconds,
			headers: headers || undefined
		} satisfies MCPServerSettingsEntry;
	});
}

/**
 * Builds an MCP server configuration from a server settings entry.
 * @param entry - The server settings entry to build the configuration from
 * @param connectionTimeoutMs - The connection timeout in milliseconds
 * @returns The built server configuration, or undefined if the entry is invalid
 */
function buildServerConfig(
	entry: MCPServerSettingsEntry,
	connectionTimeoutMs = DEFAULT_MCP_CONFIG.connectionTimeoutMs
): MCPServerConfig | undefined {
	if (!entry?.url) {
		return undefined;
	}

	let headers: Record<string, string> | undefined;
	if (entry.headers) {
		try {
			const parsed = JSON.parse(entry.headers);
			if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
				headers = parsed as Record<string, string>;
			}
		} catch {
			console.warn('[MCP] Failed to parse custom headers JSON, ignoring:', entry.headers);
		}
	}

	return {
		url: entry.url,
		transport: detectMcpTransportFromUrl(entry.url),
		handshakeTimeoutMs: connectionTimeoutMs,
		requestTimeoutMs: Math.round(entry.requestTimeoutSeconds * 1000),
		headers
	};
}

/**
 * Checks if a server is enabled considering per-chat overrides.
 * Per-chat override takes precedence over global setting.
 * Pure helper function - no side effects.
 */
export function checkServerEnabled(
	server: MCPServerSettingsEntry,
	perChatOverrides?: McpServerOverride[]
): boolean {
	if (perChatOverrides) {
		const override = perChatOverrides.find((o) => o.serverId === server.id);
		if (override !== undefined) {
			return override.enabled;
		}
	}

	return server.enabled;
}

/**
 * Builds MCP client configuration from settings.
 * Returns undefined if no valid servers are configured.
 * @param config - Global settings configuration
 * @param perChatOverrides - Optional per-chat server overrides
 */
export function buildMcpClientConfig(
	config: SettingsConfigType,
	perChatOverrides?: McpServerOverride[]
): MCPClientConfig | undefined {
	const rawServers = parseMcpServerSettings(config.mcpServers);

	if (!rawServers.length) {
		return undefined;
	}

	const servers: Record<string, MCPServerConfig> = {};
	for (const [index, entry] of rawServers.entries()) {
		if (!checkServerEnabled(entry, perChatOverrides)) continue;

		const normalized = buildServerConfig(entry);
		if (normalized) {
			servers[generateMcpServerId(entry.id, index)] = normalized;
		}
	}

	if (Object.keys(servers).length === 0) {
		return undefined;
	}

	return {
		protocolVersion: DEFAULT_MCP_CONFIG.protocolVersion,
		capabilities: DEFAULT_MCP_CONFIG.capabilities,
		clientInfo: DEFAULT_MCP_CONFIG.clientInfo,
		requestTimeoutMs: Math.round(DEFAULT_MCP_CONFIG.requestTimeoutSeconds * 1000),
		servers
	};
}
