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

// Match either Unix or Windows line endings so chunking/parsing handles
// payloads written by either scheme without off-by-one mismatches.
const LINE_BREAK_RE = /\r?\n/;

// Sentinel the search-result wire format uses when a field is absent
// (e.g. `Author: N/A`). Treated identically to a missing field so the
// rendered card hides the row either way.
const NOT_AVAILABLE_VALUE = 'N/A';

// Section header that announces the start of the multi-line Highlights
// block. Everything from that line onward (until the next `---`
// separator or end of chunk) is captured verbatim as highlight text
// instead of being re-scanned for `Title:`/`URL:`/... field lines.
const HIGHLIGHTS_SECTION_HEADER = 'Highlights:';

// Field name conventionally used by web-search tools (Exa etc.) as the
// user-supplied query parameter. Extracted so future tool schemas that
// adopt the same convention stay grep-compatible with this parser.
const SEARCH_TOOL_QUERY_FIELD = 'query';

// URL schemes the favicon helper will resolve to a hosted favicon. Any
// other scheme (e.g. data:, blob:) intentionally returns null so the UI
// can fall back to a generic globe icon.
const RESOLVABLE_URL_PROTOCOLS: readonly string[] = ['https:', 'http:'];

// Conventional favicon path served by virtually every web host.
// Appended to the URL origin as a best-effort lookup target; ignore
// 404s at render time.
const FAVICON_PATH = '/favicon.ico';

// Wire-format field names emitted by the search-result parser. String
// values match the keys the chunk parser writes into the `fields` map
// (and that callers read off `SearchResult`), so `FieldKey.TITLE` is a
// drop-in for the literal `'title'`.
enum FieldKey {
	TITLE = 'title',
	URL = 'url',
	PUBLISHED = 'published',
	AUTHOR = 'author'
}
const FIELD_PREFIXES: ReadonlyArray<{ key: FieldKey; prefix: string }> = [
	{ key: FieldKey.TITLE, prefix: 'Title:' },
	{ key: FieldKey.URL, prefix: 'URL:' },
	{ key: FieldKey.PUBLISHED, prefix: 'Published:' },
	{ key: FieldKey.AUTHOR, prefix: 'Author:' }
];

/**
 * Split a tool result string into individual search-result chunks by
 * scanning line-by-line for `---` separator rows. Handles multi-line
 * safely (line-aware, not regex on the full string) so trailing /
 * leading / consecutive separators are not lost.
 */
function splitChunks(text: string): string[] {
	const lines = text.split(LINE_BREAK_RE);
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

	const lines = chunk.split(LINE_BREAK_RE);

	const fields: Record<FieldKey, string | undefined> = {
		[FieldKey.TITLE]: undefined,
		[FieldKey.URL]: undefined,
		[FieldKey.PUBLISHED]: undefined,
		[FieldKey.AUTHOR]: undefined
	};
	const highlightLines: string[] = [];
	let inHighlights = false;

	for (const line of lines) {
		if (!inHighlights && line.trim() === HIGHLIGHTS_SECTION_HEADER) {
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
			if (value && value !== NOT_AVAILABLE_VALUE) {
				fields[key] = value;
			}
			break;
		}
	}

	if (!fields[FieldKey.TITLE] || !fields[FieldKey.URL] || !URL_SCHEME_RE.test(fields[FieldKey.URL]))
		return null;

	const highlights = highlightLines.join('\n').trim();

	const result: SearchResult = {
		title: fields[FieldKey.TITLE],
		url: fields[FieldKey.URL]
	};
	if (fields[FieldKey.PUBLISHED]) result.published = fields[FieldKey.PUBLISHED];
	if (fields[FieldKey.AUTHOR]) result.author = fields[FieldKey.AUTHOR];
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
			const candidate = (parsed as Record<string, unknown>)[SEARCH_TOOL_QUERY_FIELD];
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
		if (!RESOLVABLE_URL_PROTOCOLS.includes(parsed.protocol)) return null;
		return `${parsed.protocol}//${parsed.host}${FAVICON_PATH}`;
	} catch {
		return null;
	}
}

// Web-search MCP servers broadly follow the `web_search` token convention
// for their primary tool, but the rich pill UI makes assumptions about
// both the request shape (single `query` string) and the response shape
// (Title:/URL:/Published:/Author:/Highlights blocks). Adding a tool here
// is a deliberate signal that the renderer is known to handle its output.
// Continued maintenance note: when broadening this list, verify both the
// tool schema and the response format against the supported spec above.
export const SUPPORTED_WEB_SEARCH_TOOL_NAMES: readonly string[] = ['web_search_exa'];

/**
 * True when the tool's name is in the explicit allow-list of web-search
 * tools above. Returned to the dispatcher so it can route the call's UI
 * early (before results arrive) without false-firing on non-web-search
 * tools that also happen to accept a `query` argument.
 */
export function isWebSearchToolName(toolName: string | undefined | null): boolean {
	if (!toolName) return false;
	return SUPPORTED_WEB_SEARCH_TOOL_NAMES.includes(toolName);
}
