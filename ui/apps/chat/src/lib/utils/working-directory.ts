/**
 * Pure helpers for the working-directory picker search, backed by the
 * server's `file_glob_search` tool. Queries starting from a root (`/`,
 * `C:\`, `\\host\share`) or `~` navigate the tree (search the parent for
 * the last segment); anything else glob-matches entries under the scope
 * path, unless the caller anchors it in home with `toHomeRelativeQuery`.
 */

import { lastPathSegment } from './path-display';
import {
	GLOB,
	HIDDEN_NAME_PREFIX,
	HOME_TILDE,
	HOME_TILDE_PREFIX,
	LEADING_SLASHES_REGEX,
	PATH_SEPARATOR,
	SEARCH,
	TRAILING_SLASHES_REGEX
} from '#lib/constants/index.ts';
import type { GlobEntry, GlobSearchArgs } from '#lib/types/glob.js';

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

/**
 * The query with `\\` rewritten into `/` when the scope it is searched in is a
 * Windows path (`C:/Users/al`, `//host/share`), so `git\\llama` is a path there
 * too. Under a POSIX scope the backslash stays: it is a legal filename character.
 */
export function toScopeSeparators(query: string, scope: string): string {
	const isWindowsScope = GLOB.DRIVE_PREFIX_REGEX.test(scope) || GLOB.UNC_ROOT_REGEX.test(scope);

	if (!isWindowsScope) return query;

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

/**
 * Anchor a plain query in home, as if it were typed after `~/` (under the
 * resolved `home` when known). Queries that already start from a root (`/`,
 * `C:\`, `\\host\share`) or `~` are kept, so they still navigate the
 * absolute tree.
 */
export function toHomeRelativeQuery(query: string, home?: string | null): string {
	if (!query || splitPathQuery(query)) return query;

	return home ? joinPath(home, query) : HOME_TILDE_PREFIX + query;
}

/** Glob fragment matching `text` in any letter case, with glob metacharacters taken literally. */
function toCaseInsensitiveFragment(text: string): string {
	let out = '';

	for (const c of text) {
		const lo = c.toLowerCase();
		const up = c.toUpperCase();

		if (lo !== up) out += GLOB.RANGE_OPEN + lo + up + GLOB.RANGE_CLOSE;
		// glob metacharacters are escaped into a literal character class so a
		// query like "a*b" matches a literal '*' instead of becoming "ab"
		else if (GLOB.SPECIAL_CHARS.includes(c)) out += GLOB.RANGE_OPEN + c + GLOB.RANGE_CLOSE;
		else out += c;
	}

	return out;
}

export function buildCaseInsensitiveGlob(query: string): string {
	return GLOB.WILDCARD + toCaseInsensitiveFragment(query) + GLOB.WILDCARD;
}

/**
 * Globs that together match every name within one typo of `query`: each one
 * swaps a pair of neighbouring characters for a wildcard, which covers a
 * wrong, missing, extra or swapped character there. They match far more than
 * that, so the results are filtered with `fuzzyDistance`.
 */
export function buildFuzzyGlobs(query: string): string[] {
	const chars = [...query];

	if (chars.length < SEARCH.FUZZY_MIN_QUERY_LENGTH || chars.length > SEARCH.FUZZY_MAX_QUERY_LENGTH)
		return [];

	const globs = new Set<string>();

	for (let i = 0; i + 1 < chars.length; i++) {
		const kept = [chars.slice(0, i), chars.slice(i + 2)]
			.filter((part) => part.length > 0)
			.map((part) => toCaseInsensitiveFragment(part.join('')));

		globs.add(GLOB.WILDCARD + kept.join(GLOB.WILDCARD) + GLOB.WILDCARD);
	}

	return [...globs];
}

/**
 * Fewest single-character edits (a wrong, missing, extra or swapped
 * character) that turn `query` into some part of `text`, ignoring case.
 */
export function fuzzyDistance(text: string, query: string): number {
	const t = [...text.toLowerCase()];
	const q = [...query.toLowerCase()];

	// a match may start anywhere in the text, so the first row costs nothing
	let previous: number[] = new Array(t.length + 1).fill(0);
	let beforePrevious: number[] = [];

	for (let i = 1; i <= q.length; i++) {
		const row = [i];

		for (let j = 1; j <= t.length; j++) {
			const substitution = previous[j - 1] + (q[i - 1] === t[j - 1] ? 0 : 1);

			let best = Math.min(previous[j] + 1, row[j - 1] + 1, substitution);

			if (i > 1 && j > 1 && q[i - 1] === t[j - 2] && q[i - 2] === t[j - 1]) {
				best = Math.min(best, beforePrevious[j - 2] + 1);
			}

			row.push(best);
		}

		beforePrevious = previous;
		previous = row;
	}

	return Math.min(...previous);
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

/** True when the path is, or sits inside, a hidden folder (`.config/llama`, `git\\.cache`). */
export function isHiddenPath(path: string): boolean {
	return toPosixSeparators(path)
		.split(PATH_SEPARATOR)
		.some((segment) => segment.startsWith(HIDDEN_NAME_PREFIX));
}

/**
 * Regular entries come before hidden ones, unless the query itself asks for a
 * hidden name (`.config`). Within each group the best name match wins.
 */
export function rankEntries(entries: GlobEntry[], query: string): GlobEntry[] {
	const hiddenRank = (path: string) =>
		!query.startsWith(HIDDEN_NAME_PREFIX) && isHiddenPath(path) ? 1 : 0;

	return [...entries].sort(
		(a, b) =>
			hiddenRank(a.path) - hiddenRank(b.path) ||
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
