/**
 * Pure helpers for the working-directory picker search.
 *
 * The picker is backed by the server's `file_glob_search` built-in tool.
 * Queries that start with `/` or `~` navigate the directory tree (search
 * the parent for the last segment); anything else glob-matches home-relative
 * entries. These helpers build the glob, normalize results and rank them
 * client-side; the component owns the network/state plumbing.
 */

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
	if (!query.startsWith('/') && !query.startsWith('~')) return null;
	const normalized = query.replace(/\/+$/, '');
	if (!normalized || normalized === '~') {
		return { parent: normalized === '~' ? '~' : '/', last: '' };
	}
	const idx = normalized.lastIndexOf('/');
	if (idx === 0) return { parent: '/', last: normalized.slice(1) };
	return { parent: normalized.slice(0, idx), last: normalized.slice(idx + 1) };
}

/** Build a case-insensitive glob that matches `query` anywhere within a name. */
export function buildCaseInsensitiveGlob(query: string): string {
	let out = '*';
	for (const c of query) {
		const lo = c.toLowerCase();
		const up = c.toUpperCase();
		if (lo !== up) out += `[${lo}${up}]`;
		else if (!'*?[]'.includes(c)) out += c;
	}
	return out + '*';
}

/** Exact basename first, then prefix, then substring; lower is better. */
function rankScore(path: string, query: string): number {
	const name = lastPathSegment(path).toLowerCase();
	const q = query.toLowerCase();
	if (name === q) return 0;
	if (name.startsWith(q)) return 1;
	if (name.includes(q)) return 2;
	return 3;
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
	return base.replace(/\/+$/, '') + '/' + rel;
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
