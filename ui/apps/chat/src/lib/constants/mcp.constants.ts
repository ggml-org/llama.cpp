import { Globe, Radio, Zap } from '@lucide/svelte';
import { MCPTransportType } from '$lib/enums';
import { MimeTypeImage } from '$lib/enums/files.enums';
import type { ClientCapabilities, Implementation } from '$lib/types';
import type { Component } from 'svelte';

export const DEFAULT_CLIENT_VERSION = '1.0.0';
export const MCP_CLIENT_NAME = 'llama-ui-mcp';
export const DEFAULT_IMAGE_MIME_TYPE = MimeTypeImage.PNG;

/** MIME types considered safe for rendering MCP server icons */
export const MCP_ALLOWED_ICON_MIME_TYPES = new Set([
	MimeTypeImage.PNG,
	MimeTypeImage.JPEG,
	MimeTypeImage.JPG,
	MimeTypeImage.SVG,
	MimeTypeImage.WEBP,
	MimeTypeImage.ICO,
	MimeTypeImage.ICO_MICROSOFT
]);

/**
 * MCP specification version this client targets.
 * Update when the upstream MCP spec introduces a new stable version:
 * https://spec.modelcontextprotocol.io/
 */
export const MCP_PROTOCOL_VERSION = '2025-06-18';

export const DEFAULT_MCP_CONFIG = {
	capabilities: { tools: { listChanged: true } } as ClientCapabilities,
	clientInfo: { name: MCP_CLIENT_NAME, version: DEFAULT_CLIENT_VERSION } as Implementation,
	connectionTimeoutMs: 10_000, // 10 seconds for connection establishment
	protocolVersion: MCP_PROTOCOL_VERSION,
	requestTimeoutSeconds: 300 // 5 minutes for long-running tools
} as const;

export const MCP_SERVER_ID_PREFIX = 'LlamaUI-MCP-Server';

/** Backoff policy for reconnecting to a dropped MCP server. */
export const MCP_RECONNECT = {
	/** Per-attempt timeout for a single reconnection attempt before giving up and backing off. */
	ATTEMPT_TIMEOUT_MS: 15_000,
	BACKOFF_MULTIPLIER: 2,
	INITIAL_DELAY: 1000,
	MAX_DELAY: 30000
};

/** Maximum number of MCP server avatars to display in the chat form */
export const MAX_DISPLAYED_MCP_AVATARS = 4;

/** Expected count when two theme-less icons represent a light/dark pair */
export const EXPECTED_THEMED_ICON_PAIR_COUNT = 2;

/** CORS proxy connection settings */
export const CORS_PROXY = {
	/** Header prefix for headers that should be forwarded by the CORS proxy */
	HEADER_PREFIX: 'x-llama-server-proxy-header-',
	/** CORS proxy URL query parameter name */
	URL_PARAM: 'url'
} as const;

/** Standard SSE endpoint path indicators */
export const MCP_SSE = {
	ENDPOINT: '/sse',
	ENDPOINT_QUERY: '/sse?',
	ENDPOINT_SLASH: '/sse/'
} as const;

/** Human-readable labels for MCP transport types */
export const MCP_TRANSPORT_LABELS: Record<MCPTransportType, string> = {
	[MCPTransportType.SSE]: 'SSE',
	[MCPTransportType.STREAMABLE_HTTP]: 'HTTP',
	[MCPTransportType.WEBSOCKET]: 'WebSocket'
};

/** Icon components for MCP transport types */
export const MCP_TRANSPORT_ICONS: Record<MCPTransportType, Component> = {
	[MCPTransportType.SSE]: Radio,
	[MCPTransportType.STREAMABLE_HTTP]: Globe,
	[MCPTransportType.WEBSOCKET]: Zap
};
