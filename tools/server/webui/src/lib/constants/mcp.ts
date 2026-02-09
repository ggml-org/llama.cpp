import { Zap, Globe, Radio } from '@lucide/svelte';
import { MCPTransportType } from '$lib/enums';
import type { ClientCapabilities, Implementation } from '$lib/types';
import type { Component } from 'svelte';

export const DEFAULT_MCP_CONFIG = {
	protocolVersion: '2025-06-18',
	capabilities: { tools: { listChanged: true } } as ClientCapabilities,
	clientInfo: { name: 'llama-webui-mcp', version: 'dev' } as Implementation,
	requestTimeoutSeconds: 300, // 5 minutes for long-running tools
	connectionTimeoutMs: 10_000 // 10 seconds for connection establishment
} as const;

export const MCP_SERVER_ID_PREFIX = 'LlamaCpp-WebUI-MCP-Server-';
export const DEFAULT_CLIENT_VERSION = '1.0.0';
export const DEFAULT_IMAGE_MIME_TYPE = 'image/png';

export const MCP_RECONNECT_INITIAL_DELAY = 1000;
export const MCP_RECONNECT_BACKOFF_MULTIPLIER = 2;
export const MCP_RECONNECT_MAX_DELAY = 30000;

/** Human-readable labels for MCP transport types */
export const MCP_TRANSPORT_LABELS: Record<MCPTransportType, string> = {
	[MCPTransportType.WEBSOCKET]: 'WebSocket',
	[MCPTransportType.STREAMABLE_HTTP]: 'HTTP',
	[MCPTransportType.SSE]: 'SSE'
};

/** Icon components for MCP transport types */
export const MCP_TRANSPORT_ICONS: Record<MCPTransportType, Component> = {
	[MCPTransportType.WEBSOCKET]: Zap,
	[MCPTransportType.STREAMABLE_HTTP]: Globe,
	[MCPTransportType.SSE]: Radio
};
