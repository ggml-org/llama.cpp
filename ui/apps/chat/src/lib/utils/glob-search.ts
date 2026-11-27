/**
 * Shared `file_glob_search` runners with a short-lived result cache, so a
 * repeated query for the same (type, path, glob, depth) reuses the last
 * result instead of re-walking the tree.
 */

import { lastPathSegment } from './path-display';
import {
	buildFuzzyGlobs,
	buildGlobSearchArgs,
	fuzzyDistance,
	isHiddenPath,
	joinPath,
	rankEntries,
	splitPathQuery,
	toHomeRelativeQuery,
	toScopeSeparators
} from './working-directory';
import {
	GLOB,
	HIDDEN_NAME_PREFIX,
	HOME_TILDE,
	PATH_SEPARATOR,
	SEARCH
} from '#lib/constants/index.ts';
import { BuiltInTool, GlobSearchType } from '#lib/enums/index.ts';
import { ToolsService } from '#lib/services/tools.service.ts';
import type {
	GlobEntry,
	GlobEntryResult,
	GlobSearchArgs,
	GlobSearchChildOptions,
	GlobSearchChildResult,
	GlobSearchResult
} from '#lib/types/glob.js';

const SEARCH_CACHE_TTL_MS = 2000;

interface CacheEntry {
	results: GlobEntry[];
	base: string;
	at: number;
}

const searchCache = new Map<string, CacheEntry>();

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
		BuiltInTool.SERVER_FILE_GLOB_SEARCH,
		{ include: args.include, limit, max_depth: args.maxDepth, path: args.path, type },
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
	searchCache.set(key, { at: now, base, results: entries });

	return { base, entries };
}

function toEntryResult(e: GlobEntry, base: string): GlobEntryResult {
	return { name: lastPathSegment(e.path), path: joinPath(base, e.path), type: e.type };
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
	const search = (q: string) =>
		searchWithChildren(q, scopePath, searchDepth, limit, signal, options);

	query = toScopeSeparators(query, scopePath);

	const pathQuery = splitPathQuery(query);

	if (pathQuery) {
		const direct = await search(query);
		const canLookDeeper = !!pathQuery.last && searchDepth > SEARCH.PATH_NAV_MAX_DEPTH;

		if (direct.error || direct.entries.length > 0 || !canLookDeeper) return direct;

		// no such name in that folder (`~/llama.cpp`): look for it further down (`~/git/llama.cpp`)
		const deep = await runGlobSearch(
			{ ...direct.args, maxDepth: searchDepth },
			options.type ?? GlobSearchType.ALL,
			limit,
			signal
		);

		if (deep.error) return direct;

		const ranked = rankEntries(deep.entries, direct.args.rankQuery);

		return { ...direct, base: deep.base, entries: ranked.map((e) => toEntryResult(e, deep.base)) };
	}

	if (!query.includes(PATH_SEPARATOR)) return search(query);

	// A relative path (`ui/lib/voi`) is also followed from the scope itself: the
	// server's glob matches it only below some other folder, never at the top.
	const [direct, deep] = await Promise.all([search(joinPath(scopePath, query)), search(query)]);

	// nothing found directly: the deep search speaks, its error included
	if (direct.error || (direct.entries.length === 0 && deep.error)) return deep;

	const directPaths = new Set(direct.entries.map((entry) => entry.path));
	const below = deep.entries.filter((entry) => !directPaths.has(entry.path));

	return { ...direct, entries: [...direct.entries, ...below] };
}

async function searchWithChildren(
	query: string,
	scopePath: string,
	searchDepth: number,
	limit: number,
	signal: AbortSignal,
	options: GlobSearchChildOptions
): Promise<GlobSearchChildResult> {
	const {
		childMaxDepth = SEARCH.PATH_NAV_MAX_DEPTH,
		descendOnTrailingSeparator = false,
		type = GlobSearchType.ALL
	} = options;
	const args = buildGlobSearchArgs(query, scopePath, searchDepth);
	const res = await runGlobSearch(args, type, limit, signal);

	if (res.error) return { args, base: res.base, entries: [], error: res.error };

	const ranked = rankEntries(res.entries, args.rankQuery);
	const result = { args, base: res.base, entries: ranked.map((e) => toEntryResult(e, res.base)) };
	const wantsDescend = descendOnTrailingSeparator
		? query.endsWith(PATH_SEPARATOR) || query.endsWith(GLOB.WINDOWS_SEPARATOR)
		: true;

	if (!args.last || !wantsDescend) return result;

	return withExactDirChildren(result, ranked, args.last, type, limit, childMaxDepth, signal);
}

/** Append the children of the directory named exactly `name`, when the results hold one. */
async function withExactDirChildren(
	result: GlobSearchChildResult,
	ranked: GlobEntry[],
	name: string,
	type: GlobSearchType,
	limit: number,
	childMaxDepth: number,
	signal: AbortSignal
): Promise<GlobSearchChildResult> {
	const exact = ranked.find(
		(e) => e.type === 'dir' && lastPathSegment(e.path).toLowerCase() === name.toLowerCase()
	);

	if (!exact) return result;

	const exactDir = joinPath(result.base, exact.path);
	const childRes = await runGlobSearch(
		{ include: GLOB.WILDCARD, maxDepth: childMaxDepth, path: exactDir, rankQuery: '' },
		type,
		limit,
		signal
	);

	if (childRes.error) return result;

	// alphabetical, with hidden entries after the regular ones
	const children = childRes.entries
		.map((e) => toEntryResult(e, childRes.base))
		.sort(
			(a, b) =>
				Number(isHiddenPath(a.name)) - Number(isHiddenPath(b.name)) || a.path.localeCompare(b.path)
		);

	return { ...result, entries: [...result.entries, ...children], exactDir };
}

/**
 * Folders within a typo of the query's name, for when the exact search finds
 * nothing. The depths are walked in turn, nearest first, and the walk stops at
 * the first one that finds something: every glob is a walk of its own on the
 * server, and its result cap could crowd the nearest folders out of a deeper one.
 */
async function runFuzzyFolderSearch(
	args: GlobSearchArgs,
	depths: number[],
	signal: AbortSignal
): Promise<GlobSearchChildResult> {
	const globs = buildFuzzyGlobs(args.rankQuery);
	const close = new Map<string, GlobEntry>();

	let base = '';
	let error: string | undefined;

	for (const maxDepth of depths) {
		const results = await Promise.all(
			globs.map((include) =>
				runGlobSearch({ ...args, include, maxDepth }, GlobSearchType.DIR, SEARCH.LIMIT, signal)
			)
		);

		base = results.find((res) => !res.error)?.base ?? base;
		// a walk whose every glob failed reports why, rather than finding nothing
		error = results.every((res) => res.error) ? results[0]?.error : undefined;

		if (error) break;

		for (const entry of results.flatMap((res) => res.entries)) {
			if (fuzzyDistance(lastPathSegment(entry.path), args.rankQuery) <= SEARCH.FUZZY_MAX_EDITS) {
				close.set(entry.path, entry);
			}
		}

		if (close.size > 0) break;
	}

	if (error && close.size === 0) return { args, base, entries: [], error };

	const entries = rankEntries([...close.values()], args.rankQuery);

	return { args, base, entries: entries.map((e) => toEntryResult(e, base)) };
}

/**
 * The working-directory picker's search. A query with a path in it (`git/llama`,
 * `~/git`, `/usr/lo`) navigates: it matches the last segment among the parent's
 * children. A plain name (`llama`) is looked up anywhere under home, nearest
 * folders first: `onPartial` gets the shallow results while the deeper walk
 * runs. Either way a search that finds nothing is retried allowing for a typo.
 */
export async function runFolderSearch(
	query: string,
	home: string | null,
	signal: AbortSignal,
	onPartial?: (result: GlobSearchChildResult) => void
): Promise<GlobSearchChildResult> {
	const scope = home ?? HOME_TILDE;
	const type = GlobSearchType.DIR;

	query = toScopeSeparators(query, scope);

	if (splitPathQuery(query) || query.includes(PATH_SEPARATOR)) {
		const res = await runGlobSearchWithChildren(
			toHomeRelativeQuery(query, home),
			scope,
			SEARCH.MAX_DEPTH,
			SEARCH.LIMIT,
			signal,
			{ type }
		);

		if (res.error || res.entries.length > 0) return res;

		return runFuzzyFolderSearch(res.args, [SEARCH.PATH_NAV_MAX_DEPTH], signal);
	}

	const args = buildGlobSearchArgs(query, scope, SEARCH.SHALLOW_DEPTH);
	// started alongside the shallow search and read after it: a failure here (the search
	// being cancelled, say) must not go unhandled, and costs only the deeper results
	const deepSearch = runGlobSearch(
		{ ...args, maxDepth: SEARCH.MAX_DEPTH },
		type,
		SEARCH.LIMIT,
		signal
	).catch((): GlobSearchResult => ({ base: '', entries: [] }));
	const shallow = await runGlobSearch(args, type, SEARCH.LIMIT, signal);

	if (shallow.error) return { args, base: shallow.base, entries: [], error: shallow.error };

	const near = rankEntries(shallow.entries, query);
	// only a near folder is entered on an exact name: a deeper namesake is just a result
	const collect = (ranked: GlobEntry[], base: string) => {
		const result = { args, base, entries: ranked.map((e) => toEntryResult(e, base)) };

		return withExactDirChildren(
			result,
			near,
			query,
			type,
			SEARCH.LIMIT,
			SEARCH.PATH_NAV_MAX_DEPTH,
			signal
		);
	};

	if (near.length > 0) onPartial?.(await collect(near, shallow.base));

	const deep = await deepSearch;
	const nearPaths = new Set(near.map((entry) => entry.path));
	// deeper folders go below the near ones, so the list does not reshuffle when they arrive
	const far = rankEntries(
		deep.entries.filter((entry) => !nearPaths.has(entry.path)),
		query
	);

	if (near.length + far.length === 0) {
		return runFuzzyFolderSearch(args, [SEARCH.SHALLOW_DEPTH, SEARCH.FUZZY_MAX_DEPTH], signal);
	}

	// hidden folders go last, below even the deeper regular ones
	const isRegular = (entry: GlobEntry) =>
		query.startsWith(HIDDEN_NAME_PREFIX) || !isHiddenPath(entry.path);
	const found = [...near, ...far];

	return collect(
		[...found.filter(isRegular), ...found.filter((entry) => !isRegular(entry))],
		shallow.base
	);
}
