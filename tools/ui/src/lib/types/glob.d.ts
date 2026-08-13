import type { GlobSearchType } from '$lib/enums';

/**
 * A single directory entry returned by the server's `file_glob_search`
 * tool.
 */
export interface GlobEntry {
	path: string;
	type: string;
}

/**
 * Query arguments for a `file_glob_search` run.
 */
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

/**
 * Ranked result of a glob search against a base path.
 */
export interface GlobSearchResult {
	base: string;
	entries: GlobEntry[];
	error?: string;
}

/**
 * A glob entry resolved to an absolute path with its display name.
 */
export interface GlobEntryResult {
	path: string;
	name: string;
	type: string;
}

/**
 * Options controlling how a search descends into a matched directory.
 */
export interface GlobSearchChildOptions {
	type?: GlobSearchType;
	/** Descend only on a trailing path separator (mention picker); off for
	 * the WD picker, which descends on any exact match. */
	descendOnTrailingSeparator?: boolean;
	childMaxDepth?: number;
}

/**
 * Result of a glob search that may also list a matched directory's
 * children.
 */
export interface GlobSearchChildResult {
	base: string;
	args: GlobSearchArgs;
	/** Outer ranked entries plus the walked directory's children (absolute). */
	entries: GlobEntryResult[];
	/** Absolute path of the directory whose children were appended. */
	exactDir?: string;
	error?: string;
}
