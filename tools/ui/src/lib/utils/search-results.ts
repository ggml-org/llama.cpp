/**
 * Parsers for MCP web-search tool responses shaped like:
 *
 *     Title: <text>
 *     URL: <https url>
 *     Published: <iso date or N/A>
 *     Author: <name or N/A>
 *     Highlights:
 *     <multi-line excerpt>
 *     ---
 *     Title: <next result>
 *     ...
 *
 * The model is content-driven (any tool emitting `Title:` / `URL:` lines
 * separated by `---` qualifies), so it adapts to other web-search MCP
 * servers without hardcoding tool names.
 */

export type SearchResult = {
	title: string;
	url: string;
	published?: string;
	author?: string;
	highlights?: string;
};

const SEPARATOR_LINE_RE = /^\s*---\s*$/;
const URL_SCHEME_RE = /^https?:\/\//i;

type FieldKey = 'title' | 'url' | 'published' | 'author';
const FIELD_PREFIXES: ReadonlyArray<{ key: FieldKey; prefix: string }> = [
	{ key: 'title', prefix: 'Title:' },
	{ key: 'url', prefix: 'URL:' },
	{ key: 'published', prefix: 'Published:' },
	{ key: 'author', prefix: 'Author:' }
];

/**
 * Split a tool result string into individual search-result chunks by
 * scanning line-by-line for `---` separator rows. Handles multi-line
 * safely (line-aware, not regex on the full string) so trailing /
 * leading / consecutive separators are not lost.
 */
function splitChunks(text: string): string[] {
	const lines = text.split(/\r?\n/);
	const chunks: string[] = [];
	let buffer: string[] = [];
	for (const line of lines) {
		if (SEPARATOR_LINE_RE.test(line)) {
			if (buffer.length > 0) {
				chunks.push(buffer.join('\n'));
				buffer = [];
			}
		} else {
			buffer.push(line);
		}
	}
	if (buffer.length > 0) chunks.push(buffer.join('\n'));
	return chunks;
}

/**
 * Parse a single chunk into a SearchResult. Returns null when the chunk
 * has neither a title nor a URL — those are required for an entry to be
 * actionable (otherwise it is almost certainly malformed or a stray
 * separator line).
 */
function parseChunk(chunk: string): SearchResult | null {
	const trimmed = chunk.trim();
	if (!trimmed) return null;

	const lines = chunk.split(/\r?\n/);

	const fields: Record<FieldKey, string | undefined> = {
		title: undefined,
		url: undefined,
		published: undefined,
		author: undefined
	};
	const highlightLines: string[] = [];
	let inHighlights = false;

	for (const line of lines) {
		if (!inHighlights && line.trim() === 'Highlights:') {
			inHighlights = true;
			continue;
		}

		if (inHighlights) {
			highlightLines.push(line);
			continue;
		}

		for (const { key, prefix } of FIELD_PREFIXES) {
			if (!line.startsWith(prefix)) continue;
			const value = line.slice(prefix.length).trim();
			if (value && value !== 'N/A') {
				fields[key] = value;
			}
			break;
		}
	}

	if (!fields.title || !fields.url || !URL_SCHEME_RE.test(fields.url)) return null;

	const highlights = highlightLines.join('\n').trim();

	const result: SearchResult = { title: fields.title, url: fields.url };
	if (fields.published) result.published = fields.published;
	if (fields.author) result.author = fields.author;
	if (highlights) result.highlights = highlights;
	return result;
}

/**
 * Extract a SearchResult[] from a tool-result string. Returns `[]` when
 * the input does not match the expected shape — useful for branching
 * between dedicated search-results rendering and the generic tool-call
 * block.
 */
export function extractSearchResults(text: string | undefined | null): SearchResult[] {
	if (!text) return [];

	const results: SearchResult[] = [];
	for (const chunk of splitChunks(text)) {
		const parsed = parseChunk(chunk);
		if (parsed) results.push(parsed);
	}
	return results;
}

/**
 * Best-effort extraction of the search query out of a tool call's JSON
 * argument blob. Currently looks for a `query` field (the convention
 * used by Exa and most web-search MCP servers); returns an empty string
 * if it cannot be located.
 */
export function extractSearchQuery(toolArgs: string | undefined | null): string {
	if (!toolArgs) return '';
	try {
		const parsed: unknown = JSON.parse(toolArgs);
		if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
			const candidate = (parsed as { query?: unknown }).query;
			if (typeof candidate === 'string') return candidate.trim();
		}
	} catch {
		return '';
	}
	return '';
}

/**
 * Resolve a best-effort favicon URL for a search result, derived from the
 * result's origin (`https://host/favicon.ico`). Returns `null` when the
 * URL is malformed, has no recognizable host, or uses a non-http(s)
 * scheme — callers should fall back to a generic globe icon.
 */
export function faviconForUrl(url: string): string | null {
	try {
		const parsed = new URL(url);
		if (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') return null;
		return `${parsed.protocol}//${parsed.host}/favicon.ico`;
	} catch {
		return null;
	}
}
