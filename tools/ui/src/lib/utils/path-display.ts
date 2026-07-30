import type { ApiFilesystemRoot } from '$lib/types';

/**
 * Last non-empty slash-delimited segment of `path`, with trailing
 * slashes stripped. Returns the input unchanged when no `/` is present.
 */
export function lastPathSegment(p: string): string {
	const trimmed = p.replace(/\/+$/, '');
	const idx = trimmed.lastIndexOf('/');
	return idx === -1 ? trimmed : trimmed.slice(idx + 1);
}

/**
 * Resolve the default browse root from a list of configured roots.
 * Mirrors the picker logic: the entry flagged `default` is preferred,
 * otherwise the first entry. Returns `null` when roots haven't loaded
 * yet or the list is empty.
 */
export function resolveDefaultBrowseRoot(
	roots: ApiFilesystemRoot[] | null | undefined
): string | null {
	if (!roots || roots.length === 0) return null;
	const def = roots.find((r) => r.default);
	return def?.path ?? roots[0].path;
}

/**
 * Abbreviate `path` to `~/...` when it sits under the default browse
 * root (typically $HOME), or to `~` when the path equals the root.
 * Falls back to `lastPathSegment(path)` during fetch and when the
 * path is under a non-default root. `~` semantics are reserved for
 * the implicit home, mirroring how shells render it.
 */
export function abbreviateWorkingDir(
	path: string | null | undefined,
	roots: ApiFilesystemRoot[] | null | undefined
): string {
	if (!path) return '';
	const root = resolveDefaultBrowseRoot(roots);
	if (!root) return lastPathSegment(path);
	if (path === root) return '~';
	if (path.startsWith(root + '/')) return '~/' + path.slice(root.length + 1);
	return lastPathSegment(path);
}
