/**
 * Pure helpers for the working-directory picker search.
 *
 * The picker is backed by the server's `file_glob_search` built-in tool.
 * Queries that start with `/` or `~` navigate the directory tree (search
 * the parent for the last segment); anything else glob-matches home-relative
 * entries. These helpers build the glob, normalize results and rank them
 * client-side; the component owns the network/state plumbing.
 */

import { PATH_SEPARATOR } from '$lib/constants/mcp-resource';
import { TRAILING_SLASHES_REGEX } from '$lib/constants/url';
import {
	GLOB_RANGE_CLOSE,
	GLOB_RANGE_OPEN,
	GLOB_SPECIAL_CHARS,
	GLOB_WILDCARD,
	HOME_TILDE
} from '$lib/constants';
import { lastPathSegment } from './path-display';

export interface GlobEntry {
	path: string;
	type: string;
}

export interface PathQuery {
	parent: string;
	last: string;
}

/** A query starting with `/` or `~` is path navigation, not a home-relative glob. */
export function splitPathQuery(query: string): PathQuery | null {
	if (!query.startsWith(PATH_SEPARATOR) && !query.startsWith(HOME_TILDE)) return null;
	const normalized = query.replace(TRAILING_SLASHES_REGEX, '');
	if (!normalized || normalized === HOME_TILDE) {
		return { parent: normalized === HOME_TILDE ? HOME_TILDE : PATH_SEPARATOR, last: '' };
	}
	const idx = normalized.lastIndexOf(PATH_SEPARATOR);
	if (idx === 0) return { parent: PATH_SEPARATOR, last: normalized.slice(1) };
	return { parent: normalized.slice(0, idx), last: normalized.slice(idx + 1) };
}

/** Build a case-insensitive glob that matches `query` anywhere within a name. */
export function buildCaseInsensitiveGlob(query: string): string {
	let out = GLOB_WILDCARD;
	for (const c of query) {
		const lo = c.toLowerCase();
		const up = c.toUpperCase();
		if (lo !== up) out += GLOB_RANGE_OPEN + lo + up + GLOB_RANGE_CLOSE;
		else if (!GLOB_SPECIAL_CHARS.includes(c)) out += c;
	}
	return out + GLOB_WILDCARD;
}

/** Exact basename first, then prefix, then substring; lower is better. */
const RANK_EXACT = 0;
const RANK_PREFIX = 1;
const RANK_SUBSTRING = 2;
const RANK_OTHER = 3;

function rankScore(path: string, query: string): number {
	const name = lastPathSegment(path).toLowerCase();
	const q = query.toLowerCase();
	if (name === q) return RANK_EXACT;
	if (name.startsWith(q)) return RANK_PREFIX;
	if (name.includes(q)) return RANK_SUBSTRING;
	return RANK_OTHER;
}

/** Sort entries by relevance, then shorter path, then alphabetically. */
export function rankEntries(entries: GlobEntry[], query: string): GlobEntry[] {
	return [...entries].sort(
		(a, b) =>
			rankScore(a.path, query) - rankScore(b.path, query) ||
			a.path.length - b.path.length ||
			a.path.localeCompare(b.path)
	);
}

/** Join a base path and a relative segment, avoiding duplicate slashes. */
export function joinPath(base: string, rel: string): string {
	if (!base) return rel;
	return base.replace(TRAILING_SLASHES_REGEX, '') + PATH_SEPARATOR + rel;
}

/** Split `text` into alternating segments at each case-insensitive `query` match. */
export function highlightMatch(text: string, query: string): { text: string; match: boolean }[] {
	if (!query) return [{ text, match: false }];
	const segments: { text: string; match: boolean }[] = [];
	const lowerText = text.toLowerCase();
	const lowerQuery = query.toLowerCase();
	let i = 0;
	while (i < text.length) {
		const idx = lowerText.indexOf(lowerQuery, i);
		if (idx < 0) {
			segments.push({ text: text.slice(i), match: false });
			break;
		}
		if (idx > i) segments.push({ text: text.slice(i, idx), match: false });
		segments.push({ text: text.slice(idx, idx + query.length), match: true });
		i = idx + query.length;
	}
	return segments;
}
