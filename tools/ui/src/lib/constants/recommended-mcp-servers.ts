import { DEFAULT_MCP_CONFIG } from './mcp';
import type { MCPServerSettingsEntry } from '$lib/types';

/**
 * Pre-defined recommended MCP servers.
 *
 * Servers are enabled by default, but they are not turned on for individual
 * conversations until the user explicitly enables them (so their tools are
 * disabled by default).
 */
export const RECOMMENDED_MCP_SERVERS: MCPServerSettingsEntry[] = [
	{
		id: 'exa-web-search',
		name: 'Exa Web Search',
		url: 'https://mcp.exa.ai/mcp',
		enabled: true,
		requestTimeoutSeconds: DEFAULT_MCP_CONFIG.requestTimeoutSeconds
	},
	{
		id: 'huggingface-mcp',
		name: 'Hugging Face',
		url: 'https://huggingface.co/mcp',
		enabled: true,
		requestTimeoutSeconds: DEFAULT_MCP_CONFIG.requestTimeoutSeconds
	}
];

export const RECOMMENDED_MCP_SERVER_IDS = new Set(
	RECOMMENDED_MCP_SERVERS.map((server) => server.id)
);
