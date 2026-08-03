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
 * Abbreviate `path` to `~/...` when it sits under `home`, or to `~` when
 * it equals `home`. Falls back to `lastPathSegment(path)` when home is
 * unknown or the path is outside it. `~` semantics are reserved for the
 * home directory, mirroring how shells render it.
 */
export function abbreviateWorkingDir(
	path: string | null | undefined,
	home: string | null | undefined
): string {
	if (!path) return '';
	if (!home) return lastPathSegment(path);
	if (path === home) return '~';
	if (path.startsWith(home + '/')) return '~/' + path.slice(home.length + 1);
	return lastPathSegment(path);
}

/**
 * Replace a leading `home` prefix in `path` with `~`. Unlike
 * abbreviateWorkingDir, paths outside `home` (or an unknown home) are
 * returned unchanged - used for tool-call path displays where the full
 * path matters.
 */
export function abbreviateHome(path: string, home: string | null | undefined): string {
	if (!home) return path;
	if (path === home) return '~';
	if (path.startsWith(home + '/')) return '~/' + path.slice(home.length + 1);
	return path;
}

export const CWD_CHANGED_PREFIX = 'Set working directory to ';
export const CWD_CLEARED_TEXT = 'Working directory cleared';

const CWD_CHANGED_PREFIX_LEGACY = 'CWD is changed to: ';
const CWD_CLEARED_TEXT_LEGACY = 'CWD is cleared';

export interface CwdMessageInfo {
	// absolute server-side path, null when the cwd was cleared
	path: string | null;
	// display form shown in the UI (e.g. ~/Documents)
	display: string;
}

/**
 * Format a synthetic cwd-change message. The text mirrors what the UI
 * renders for it; the path travels as `[file:///abs/path](display)` so
 * both the absolute and the short form are visible to the model and
 * parseable back by the UI.
 */
export function formatCwdMessage(cwd: string, home: string | null): string {
	const display = abbreviateWorkingDir(cwd, home);
	return `${CWD_CHANGED_PREFIX}[file://${cwd}](${display}).`;
}

/**
 * Parse a synthetic cwd message back into its parts. Also accepts the
 * legacy formats (`CWD is changed to: ...`, with the link parts swapped)
 * and a plain path after the prefix. Returns null when `content` is not
 * a cwd message.
 */
export function parseCwdMessage(content: string): CwdMessageInfo | null {
	const trimmed = content.trim();
	if (trimmed === CWD_CLEARED_TEXT || trimmed === CWD_CLEARED_TEXT_LEGACY) {
		return { path: null, display: '' };
	}
	if (trimmed.startsWith(CWD_CHANGED_PREFIX)) {
		const rest = trimmed.slice(CWD_CHANGED_PREFIX.length);
		// not anchored to the end: guidance may follow the link
		const link = rest.match(/^\[file:\/\/([\s\S]*)\]\(([\s\S]*)\)/);
		if (link) return { path: link[1], display: link[2] };
		return { path: rest, display: rest };
	}
	if (trimmed.startsWith(CWD_CHANGED_PREFIX_LEGACY)) {
		const rest = trimmed.slice(CWD_CHANGED_PREFIX_LEGACY.length);
		const link = rest.match(/^\[([\s\S]*)\]\(file:\/\/([\s\S]*)\)$/);
		if (link) return { path: link[2], display: link[1] };
		return { path: rest, display: rest };
	}
	return null;
}
