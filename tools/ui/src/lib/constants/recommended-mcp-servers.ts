import type { RecommendedMCPServer } from '$lib/types';

// Suggested MCP servers shown as opt-in cards in the "Add New Server" dialog.
// Rendering these cards never reaches the upstream domain - favicons come
// from local bundles in static/recommended-mcp/ and the URL is only used
// after the user clicks Add.
export const RECOMMENDED_MCP_SERVERS: RecommendedMCPServer[] = [
	{
		id: 'exa',
		name: 'Exa',
		description: 'Search the web and fetch full page content as clean markdown.',
		url: 'https://mcp.exa.ai/mcp',
		iconUrl: '/recommended-mcp/exa.ico'
	},
	{
		id: 'huggingface',
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
