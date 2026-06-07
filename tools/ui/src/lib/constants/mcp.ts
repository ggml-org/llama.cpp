import { Zap, Globe, Radio } from '@lucide/svelte';
import { MCPTransportType } from '$lib/enums';
import type { ClientCapabilities, Implementation } from '$lib/types';
import type { Component } from 'svelte';
import { MimeTypeImage } from '$lib/enums/files';

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
	protocolVersion: MCP_PROTOCOL_VERSION,
	capabilities: { tools: { listChanged: true } } as ClientCapabilities,
	clientInfo: { name: MCP_CLIENT_NAME, version: DEFAULT_CLIENT_VERSION } as Implementation,
	requestTimeoutSeconds: 300, // 5 minutes for long-running tools
	connectionTimeoutMs: 10_000 // 10 seconds for connection establishment
} as const;

export const MCP_SERVER_ID_PREFIX = 'LlamaUI-MCP-Server';

export const MCP_RECONNECT_INITIAL_DELAY = 1000;
export const MCP_RECONNECT_BACKOFF_MULTIPLIER = 2;
export const MCP_RECONNECT_MAX_DELAY = 30000;
/** Per-attempt timeout for a single reconnection attempt before giving up and backing off. */
export const MCP_RECONNECT_ATTEMPT_TIMEOUT_MS = 15_000;

/** Maximum number of MCP server avatars to display in the chat form */
export const MAX_DISPLAYED_MCP_AVATARS = 4;

/** Expected count when two theme-less icons represent a light/dark pair */
export const EXPECTED_THEMED_ICON_PAIR_COUNT = 2;

/** CORS proxy URL query parameter name */
export const CORS_PROXY_URL_PARAM = 'url';

/** Number of trailing characters to keep visible when partially redacting mcp-session-id */
export const MCP_SESSION_ID_VISIBLE_CHARS = 5;

/** Partial-redaction rules for MCP headers: header name -> visible trailing chars */
export const MCP_PARTIAL_REDACT_HEADERS = new Map<string, number>([
	['mcp-session-id', MCP_SESSION_ID_VISIBLE_CHARS]
]);

/** Header names whose values should be redacted in diagnostic logs */
export const REDACTED_HEADERS = new Set([
	'authorization',
	'api-key',
	'cookie',
	'mcp-session-id',
	'proxy-authorization',
	'set-cookie',
	'x-auth-token',
	'x-api-key'
]);

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
