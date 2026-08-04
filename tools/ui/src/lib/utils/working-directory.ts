/**
 * Pure helpers for the working-directory picker search.
 *
 * The picker is backed by the server's `file_glob_search` built-in tool.
 * Queries that start from a root (`/`, `C:\`, `\\host\share`) or from `~`
 * navigate the directory tree (search the parent for the last segment);
 * anything else glob-matches home-relative entries. Paths are carried with
 * `/` separators, which is what the server returns and what Windows accepts.
 * These helpers build the glob, normalize results and rank them
 * client-side; the component owns the network/state plumbing.
 */

import { PATH_SEPARATOR } from '$lib/constants/mcp-resource';
import { TRAILING_SLASHES_REGEX } from '$lib/constants/url';
import {
	DRIVE_PREFIX_REGEX,
	DRIVE_ROOT_REGEX,
	GLOB_RANGE_CLOSE,
	GLOB_RANGE_OPEN,
	GLOB_SPECIAL_CHARS,
	GLOB_WILDCARD,
	HOME_TILDE,
	LEADING_SLASHES_REGEX,
	UNC_ROOT_REGEX,
	WINDOWS_SEPARATOR
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

/**
 * Rewrite `\` into `/` when the query carries a Windows root. Elsewhere the
 * backslash is left alone: it is a legal filename character on POSIX.
 */
function toPosixSeparators(query: string): string {
	if (!DRIVE_PREFIX_REGEX.test(query) && !query.startsWith(WINDOWS_SEPARATOR)) return query;
	return query.split(WINDOWS_SEPARATOR).join(PATH_SEPARATOR);
}

/**
 * Length of the root prefix of `path`, or 0 when it has none. Covers the
 * POSIX root, a Windows drive (`C:/`) and a UNC share (`//host/share/`).
 */
export function rootPrefixLength(path: string): number {
	const unc = path.match(UNC_ROOT_REGEX);
	if (unc) return unc[0].length;
	const drive = path.match(DRIVE_ROOT_REGEX);
	if (drive) return drive[0].length;
	return path.startsWith(PATH_SEPARATOR) ? PATH_SEPARATOR.length : 0;
}

/** A query starting from a root or from `~` is path navigation, not a home-relative glob. */
export function splitPathQuery(query: string): PathQuery | null {
	const normalized = toPosixSeparators(query);
	const rootLength = rootPrefixLength(normalized);
	if (rootLength === 0 && !normalized.startsWith(HOME_TILDE)) return null;

	// a root keeps its trailing separator so it stays absolute on its own
	const root =
		rootLength > 0
			? normalized.slice(0, rootLength).replace(TRAILING_SLASHES_REGEX, '') + PATH_SEPARATOR
			: HOME_TILDE;

	const rest = normalized
		.slice(rootLength > 0 ? rootLength : HOME_TILDE.length)
		.replace(LEADING_SLASHES_REGEX, '')
		.replace(TRAILING_SLASHES_REGEX, '');

	const parentOf = (dirs: string) =>
		rootLength > 0 ? root + dirs : HOME_TILDE + PATH_SEPARATOR + dirs;

	if (!rest) return { parent: root, last: '' };

	const idx = rest.lastIndexOf(PATH_SEPARATOR);
	if (idx === -1) return { parent: root, last: rest };
	return { parent: parentOf(rest.slice(0, idx)), last: rest.slice(idx + 1) };
}

/** Build a case-insensitive glob that matches `query` anywhere within a name. */
export function buildCaseInsensitiveGlob(query: string): string {
	let out = GLOB_WILDCARD;
	for (const c of query) {
		const lo = c.toLowerCase();
		const up = c.toUpperCase();
		if (lo !== up) out += GLOB_RANGE_OPEN + lo + up + GLOB_RANGE_CLOSE;
		// glob metacharacters are escaped into a literal character class so a
		// query like "a*b" matches a literal '*' instead of becoming "ab"
		else if (GLOB_SPECIAL_CHARS.includes(c)) out += GLOB_RANGE_OPEN + c + GLOB_RANGE_CLOSE;
		else out += c;
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
