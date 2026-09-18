/**
 * A single parsed entry from a web-search MCP tool result.
 */
export type SearchResult = {
	title: string;
	url: string;
	published?: string;
	author?: string;
	highlights?: string;
};
