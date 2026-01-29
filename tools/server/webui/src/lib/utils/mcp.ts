import type { MCPServerSettingsEntry } from '$lib/types';
import { MCPTransportType, MCPLogLevel, UrlPrefix } from '$lib/enums';
import { DEFAULT_MCP_CONFIG } from '$lib/constants/mcp';
import { Info, AlertTriangle, XCircle } from '@lucide/svelte';
import type { Component } from 'svelte';

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
				: `server-${index + 1}`;

		return {
			id,
			enabled: Boolean((entry as { enabled?: unknown })?.enabled),
			url,
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
