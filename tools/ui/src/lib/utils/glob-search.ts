/**
 * Shared `file_glob_search` runner with a short-lived result cache.
 *
 * Both the working-directory picker and the file/folder mention picker
 * glob-search the server tree as the user types. They share this module so
 * a repeated query for the same (type, path, glob, depth) - e.g. opening one
 * picker right after the other on the same directory - reuses the last result
 * instead of re-walking the tree. Entries are considered fresh only within a
 * short TTL; anything older re-fetches.
 */

import { BuiltInTool, GlobSearchType } from '$lib/enums';
import { ToolsService } from '$lib/services/tools.service';
import type { GlobEntry, GlobSearchArgs } from './working-directory';

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

/**
 * Run (or serve from cache) one `file_glob_search` call. Returns the raw
 * server entries; callers map them to their own shape and rank locally.
 */
export async function runGlobSearch(
	args: GlobSearchArgs,
	type: GlobSearchType,
	limit: number,
	signal: AbortSignal
): Promise<GlobSearchResult> {
	const key = `${type}\u0000${args.path}\u0000${args.include}\u0000${args.maxDepth}`;
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
	searchCache.set(key, { results: entries, base, at: Date.now() });
	return { base, entries };
}
