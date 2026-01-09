import type { ClientCapabilities, Implementation } from '$lib/types/mcp';

export const DEFAULT_MCP_CONFIG = {
	protocolVersion: '2025-06-18',
	capabilities: { tools: { listChanged: true } } as ClientCapabilities,
	clientInfo: { name: 'llama-webui-mcp', version: 'dev' } as Implementation,
	requestTimeoutSeconds: 300, // 5 minutes for long-running tools
	connectionTimeoutMs: 10_000 // 10 seconds for connection establishment
} as const;
