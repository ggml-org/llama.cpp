import type { RecommendedMCPServer } from '$lib/types';

/**
 * Hard-coded suggested MCP servers.
 *
 * Shown inside the "Add New Server" dialog as opt-in cards. These entries
 * are rendered without triggering any connection or health check, so
 * showing them never causes a request to leave the user's machine. Only
 * when the user explicitly picks one and clicks the dialog's own Add
 * button does the URL end up in the configured server list and start
 * contacting the upstream server.
 *
 * Favicons are bundled in `static/recommended-mcp/` and resolved via
 * static asset paths so the card never reaches the upstream domain
 * until the user adds the server.
 */
export const RECOMMENDED_MCP_SERVERS: RecommendedMCPServer[] = [
	{
		id: 'exa-web-search',
		name: 'Exa',
		description: 'Search the web and fetch full page content as clean markdown.',
		url: 'https://mcp.exa.ai/mcp',
		iconUrl: '/recommended-mcp/exa.ico'
	},
	{
		id: 'huggingface-mcp',
		name: 'Hugging Face',
		description: 'Search and browse AI models, datasets, spaces, and docs on the Hugging Face Hub.',
		url: 'https://huggingface.co/mcp',
		iconUrl: '/recommended-mcp/huggingface.ico'
	},
	{
		id: 'github',
		name: 'GitHub',
		description: 'Search repositories, issues, pull requests and interact with code on GitHub.',
		url: 'https://api.githubcopilot.com/mcp',
		iconUrlLight: '/recommended-mcp/github-light.png',
		iconUrlDark: '/recommended-mcp/github-dark.png',
		needsAuthorization: true
	},
	{
		id: 'context7',
		name: 'Context7',
		description: 'Browse up-to-date documentation and code examples for libraries and frameworks.',
		url: 'https://mcp.context7.com/mcp',
		iconUrl: '/recommended-mcp/context7.png'
	}
];
