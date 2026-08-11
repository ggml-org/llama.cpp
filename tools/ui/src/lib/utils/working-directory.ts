/**
 * Pure helpers for the working-directory picker search, backed by the
 * server's `file_glob_search` tool. Queries starting from a root (`/`,
 * `C:\`, `\\host\share`) or `~` navigate the tree (search the parent for
 * the last segment); anything else glob-matches home-relative entries.
 */

import { lastPathSegment } from './path-display';
import {
	GLOB,
	HOME_TILDE,
	LEADING_SLASHES_REGEX,
	PATH_SEPARATOR,
	SEARCH,
	TRAILING_SLASHES_REGEX
} from '$lib/constants';

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
	if (!GLOB.DRIVE_PREFIX_REGEX.test(query) && !query.startsWith(GLOB.WINDOWS_SEPARATOR))
		return query;

	return query.split(GLOB.WINDOWS_SEPARATOR).join(PATH_SEPARATOR);
}

export function rootPrefixLength(path: string): number {
	const unc = path.match(GLOB.UNC_ROOT_REGEX);

	if (unc) return unc[0].length;

	const drive = path.match(GLOB.DRIVE_ROOT_REGEX);

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

	if (!rest) return { last: '', parent: root };

	const idx = rest.lastIndexOf(PATH_SEPARATOR);

	if (idx === -1) return { last: rest, parent: root };

	return { last: rest.slice(idx + 1), parent: parentOf(rest.slice(0, idx)) };
}

export function buildCaseInsensitiveGlob(query: string): string {
	let out = GLOB.WILDCARD;

	for (const c of query) {
		const lo = c.toLowerCase();
		const up = c.toUpperCase();

		if (lo !== up) out += GLOB.RANGE_OPEN + lo + up + GLOB.RANGE_CLOSE;
		// glob metacharacters are escaped into a literal character class so a
		// query like "a*b" matches a literal '*' instead of becoming "ab"
		else if (GLOB.SPECIAL_CHARS.includes(c)) out += GLOB.RANGE_OPEN + c + GLOB.RANGE_CLOSE;
		else out += c;
	}

	return out + GLOB.WILDCARD;
}

export interface GlobSearchArgs {
	path: string;
	include: string;
	maxDepth: number;
	rankQuery: string;
	/** Last segment of a path-navigation query (`~/dir/sub`), undefined for
	 * a plain home-relative glob. Lets callers act on the exact targeted
	 * segment (e.g. the WD picker "entering" a directory). */
	last?: string;
}

export function buildGlobSearchArgs(
	query: string,
	scopePath: string,
	searchDepth: number
): GlobSearchArgs {
	const pathQuery = splitPathQuery(query);
	const path = pathQuery ? pathQuery.parent : scopePath;
	const include = pathQuery
		? pathQuery.last
			? buildCaseInsensitiveGlob(pathQuery.last)
			: GLOB.WILDCARD
		: buildCaseInsensitiveGlob(query);
	const maxDepth = pathQuery ? SEARCH.PATH_NAV_MAX_DEPTH : searchDepth;

	return { include, last: pathQuery?.last, maxDepth, path, rankQuery: pathQuery?.last ?? query };
}

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

export function rankEntries(entries: GlobEntry[], query: string): GlobEntry[] {
	return [...entries].sort(
		(a, b) =>
			rankScore(a.path, query) - rankScore(b.path, query) ||
			a.path.length - b.path.length ||
			a.path.localeCompare(b.path)
	);
}

export function joinPath(base: string, rel: string): string {
	if (!base) return rel;

	return base.replace(TRAILING_SLASHES_REGEX, '') + PATH_SEPARATOR + rel;
}

export function highlightMatch(text: string, query: string): { text: string; match: boolean }[] {
	if (!query) return [{ match: false, text }];

	const segments: { text: string; match: boolean }[] = [];
	const lowerText = text.toLowerCase();
	const lowerQuery = query.toLowerCase();

	let i = 0;

	while (i < text.length) {
		const idx = lowerText.indexOf(lowerQuery, i);

		if (idx < 0) {
			segments.push({ match: false, text: text.slice(i) });

			break;
		}

		if (idx > i) segments.push({ match: false, text: text.slice(i, idx) });

		segments.push({ match: true, text: text.slice(idx, idx + query.length) });
		i = idx + query.length;
	}

	return segments;
}
