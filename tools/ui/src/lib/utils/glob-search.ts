/**
 * Shared `file_glob_search` runners with a short-lived result cache, so a
 * repeated query for the same (type, path, glob, depth) reuses the last
 * result instead of re-walking the tree.
 */

import { BuiltInTool, GlobSearchType } from '$lib/enums';
import { ToolsService } from '$lib/services/tools.service';
import {
	GLOB_WILDCARD,
	PATH_NAV_MAX_DEPTH,
	PATH_SEPARATOR,
	WINDOWS_SEPARATOR
} from '$lib/constants';
import { lastPathSegment } from './path-display';
import {
	buildGlobSearchArgs,
	joinPath,
	rankEntries,
	type GlobEntry,
	type GlobSearchArgs
} from './working-directory';

const SEARCH_CACHE_TTL_MS = 2000;

interface CacheEntry {
	results: GlobEntry[];
	base: string;
	at: number;
}

const searchCache = new Map<string, CacheEntry>();

export interface GlobSearchResult {
	base: string;
	entries: GlobEntry[];
	error?: string;
}

export async function runGlobSearch(
	args: GlobSearchArgs,
	type: GlobSearchType,
	limit: number,
	signal: AbortSignal
): Promise<GlobSearchResult> {
	const key = `${type}\u0000${args.path}\u0000${args.include}\u0000${args.maxDepth}\u0000${limit}`;
	const cached = searchCache.get(key);
	if (cached && Date.now() - cached.at < SEARCH_CACHE_TTL_MS) {
		return { base: cached.base, entries: cached.results };
	}

	const res = await ToolsService.executeToolRaw(
		BuiltInTool.FILE_GLOB_SEARCH,
		{ path: args.path, type, include: args.include, max_depth: args.maxDepth, limit },
		signal
	);

	if (typeof res.error === 'string') return { base: '', entries: [], error: res.error };

	const base = typeof res.base === 'string' ? res.base : '';
	const entries = Array.isArray(res.entries) ? (res.entries as GlobEntry[]) : [];
	const now = Date.now();
	// prune stale entries so the short-lived cache cannot grow unbounded
	for (const [k, v] of searchCache) {
		if (now - v.at >= SEARCH_CACHE_TTL_MS) searchCache.delete(k);
	}
	searchCache.set(key, { results: entries, base, at: now });
	return { base, entries };
}

export interface GlobEntryResult {
	path: string;
	name: string;
	type: string;
}

export interface GlobSearchChildOptions {
	type?: GlobSearchType;
	/** Descend only on a trailing path separator (mention picker); off for
	 * the WD picker, which descends on any exact match. */
	descendOnTrailingSeparator?: boolean;
	childMaxDepth?: number;
}

export interface GlobSearchChildResult {
	base: string;
	args: GlobSearchArgs;
	/** Outer ranked entries plus the walked directory's children (absolute). */
	entries: GlobEntryResult[];
	/** Absolute path of the directory whose children were appended. */
	exactDir?: string;
	error?: string;
}

function toEntryResult(e: GlobEntry, base: string): GlobEntryResult {
	return { path: joinPath(base, e.path), name: lastPathSegment(e.path), type: e.type };
}

/**
 * One ranked glob search that may also list the matched directory's
 * children, shared by the WD picker (descend on exact match) and the
 * mention picker (descend on a trailing `/` or `\`).
 */
export async function runGlobSearchWithChildren(
	query: string,
	scopePath: string,
	searchDepth: number,
	limit: number,
	signal: AbortSignal,
	options: GlobSearchChildOptions = {}
): Promise<GlobSearchChildResult> {
	const {
		type = GlobSearchType.ALL,
		descendOnTrailingSeparator = false,
		childMaxDepth = PATH_NAV_MAX_DEPTH
	} = options;

	const args = buildGlobSearchArgs(query, scopePath, searchDepth);
	const res = await runGlobSearch(args, type, limit, signal);
	if (res.error) return { base: res.base, args, entries: [], error: res.error };

	const ranked = rankEntries(res.entries, args.rankQuery);
	const entries = ranked.map((e) => toEntryResult(e, res.base));

	const last = args.last;
	if (last) {
		const wantsDescend = descendOnTrailingSeparator
			? query.endsWith(PATH_SEPARATOR) || query.endsWith(WINDOWS_SEPARATOR)
			: true;
		const exact = ranked.find(
			(e) => e.type === 'dir' && lastPathSegment(e.path).toLowerCase() === last.toLowerCase()
		);
		if (wantsDescend && exact) {
			const exactDir = joinPath(res.base, exact.path);
			const childRes = await runGlobSearch(
				{ path: exactDir, include: GLOB_WILDCARD, maxDepth: childMaxDepth, rankQuery: '' },
				type,
				limit,
				signal
			);
			if (!childRes.error) {
				const children = childRes.entries
					.map((e) => toEntryResult(e, childRes.base))
					.sort((a, b) => a.path.localeCompare(b.path));
				return { base: res.base, args, entries: [...entries, ...children], exactDir };
			}
		}
	}

	return { base: res.base, args, entries };
}
