import type { MCPTransportType } from '$lib/types/mcp';

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
