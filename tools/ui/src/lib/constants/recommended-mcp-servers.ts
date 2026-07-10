import { DEFAULT_MCP_CONFIG } from './mcp';
import type { RecommendedMCPServer } from '$lib/types';

/**
 * Hard-coded suggested MCP servers.
 *
 * Shown only inside the "Add New Server" dialog as opt-in cards. These
 * entries are rendered without triggering any connection or health check,
 * so showing them never causes a request to leave the user's machine.
 * Only when the user explicitly picks one and clicks the dialog's own
 * Add button does the URL end up in the configured server list and start
 * contacting the upstream server.
 */
export const RECOMMENDED_MCP_SERVERS: RecommendedMCPServer[] = [
	{
		id: 'exa-web-search',
		name: 'Exa Web Search',
		description: 'Search the web and retrieve relevant content.',
		url: 'https://mcp.exa.ai/mcp',
		enabled: true,
		requestTimeoutSeconds: DEFAULT_MCP_CONFIG.requestTimeoutSeconds
	},
	{
		id: 'huggingface-mcp',
		name: 'Hugging Face',
		description: 'Browse AI models, datasets, spaces and more.',
		url: 'https://huggingface.co/mcp',
		enabled: true,
		requestTimeoutSeconds: DEFAULT_MCP_CONFIG.requestTimeoutSeconds
	}
];
