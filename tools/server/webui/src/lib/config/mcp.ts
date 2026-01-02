import type { MCPClientConfig, MCPServerConfig, MCPServerSettingsEntry } from '$lib/types/mcp';
import type { SettingsConfigType } from '$lib/types/settings';
import { DEFAULT_MCP_CONFIG } from '$lib/constants/mcp';
import { detectMcpTransportFromUrl, generateMcpServerId } from '$lib/utils/mcp';
import { normalizePositiveNumber } from '$lib/utils/number';

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

function buildServerConfig(
	entry: MCPServerSettingsEntry,
	connectionTimeoutMs = DEFAULT_MCP_CONFIG.connectionTimeoutMs
): MCPServerConfig | undefined {
	if (!entry?.url) {
		return undefined;
	}

	// Parse custom headers if provided
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
 * Builds MCP client configuration from settings.
 * Returns undefined if no valid servers are configured.
 */
export function buildMcpClientConfig(config: SettingsConfigType): MCPClientConfig | undefined {
	const rawServers = parseMcpServerSettings(config.mcpServers);

	if (!rawServers.length) {
		return undefined;
	}

	const servers: Record<string, MCPServerConfig> = {};
	for (const [index, entry] of rawServers.entries()) {
		if (!entry.enabled) continue;

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

export function hasEnabledMcpServers(config: SettingsConfigType): boolean {
	return Boolean(buildMcpClientConfig(config));
}
