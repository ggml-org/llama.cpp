import type {
	MCPClientCapabilities,
	MCPClientConfig,
	MCPClientInfo,
	MCPServerConfig
} from '../mcp/types';
import type { SettingsConfigType } from '$lib/types/settings';

/**
 * Raw MCP server configuration entry stored in settings.
 */
export type MCPServerSettingsEntry = {
	id: string;
	enabled: boolean;
	url: string;
	requestTimeoutSeconds: number;
};

const defaultMcpConfig = {
	protocolVersion: '2025-06-18',
	capabilities: { tools: { listChanged: true } } as MCPClientCapabilities,
	clientInfo: { name: 'llama-webui-mcp', version: 'dev' } as MCPClientInfo,
	requestTimeoutSeconds: 300, // 5 minutes for long-running tools
	connectionTimeoutMs: 10_000 // 10 seconds for connection establishment
};

export function getDefaultMcpConfig() {
	return defaultMcpConfig;
}

export function detectMcpTransportFromUrl(url: string): 'websocket' | 'streamable_http' {
	const normalized = url.trim().toLowerCase();
	return normalized.startsWith('ws://') || normalized.startsWith('wss://')
		? 'websocket'
		: 'streamable_http';
}

function normalizeRequestTimeoutSeconds(value: unknown, fallback: number): number {
	const parsed = typeof value === 'string' ? Number.parseFloat(value) : Number(value);
	if (!Number.isFinite(parsed) || parsed <= 0) {
		return fallback;
	}

	return parsed;
}

function sanitizeId(id: unknown, index: number): string {
	if (typeof id === 'string' && id.trim()) {
		return id.trim();
	}

	return `server-${index + 1}`;
}

function sanitizeUrl(url: unknown): string {
	if (typeof url === 'string') {
		return url.trim();
	}

	return '';
}

export function parseMcpServerSettings(
	rawServers: unknown,
	fallbackRequestTimeoutSeconds = defaultMcpConfig.requestTimeoutSeconds
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
		const requestTimeoutSeconds = normalizeRequestTimeoutSeconds(
			(entry as { requestTimeoutSeconds?: unknown })?.requestTimeoutSeconds,
			fallbackRequestTimeoutSeconds
		);

		const url = sanitizeUrl((entry as { url?: unknown })?.url);

		return {
			id: sanitizeId((entry as { id?: unknown })?.id, index),
			enabled: Boolean((entry as { enabled?: unknown })?.enabled),
			url,
			requestTimeoutSeconds
		} satisfies MCPServerSettingsEntry;
	});
}

function buildServerConfig(
	entry: MCPServerSettingsEntry,
	connectionTimeoutMs = defaultMcpConfig.connectionTimeoutMs
): MCPServerConfig | undefined {
	if (!entry?.url) {
		return undefined;
	}

	return {
		url: entry.url,
		transport: detectMcpTransportFromUrl(entry.url),
		handshakeTimeoutMs: connectionTimeoutMs,
		requestTimeoutMs: Math.round(entry.requestTimeoutSeconds * 1000)
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
			servers[sanitizeId(entry.id, index)] = normalized;
		}
	}

	if (Object.keys(servers).length === 0) {
		return undefined;
	}

	return {
		protocolVersion: defaultMcpConfig.protocolVersion,
		capabilities: defaultMcpConfig.capabilities,
		clientInfo: defaultMcpConfig.clientInfo,
		requestTimeoutMs: Math.round(defaultMcpConfig.requestTimeoutSeconds * 1000),
		servers
	};
}

export function hasEnabledMcpServers(config: SettingsConfigType): boolean {
	return Boolean(buildMcpClientConfig(config));
}
