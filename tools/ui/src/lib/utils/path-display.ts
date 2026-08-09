import {
	CWD_CHANGED_PREFIX,
	CWD_CLEARED_TEXT,
	CWD_LINK_REGEX,
	FILE_URI_PREFIX,
	HOME_TILDE,
	HOME_TILDE_PREFIX
} from '$lib/constants';
import { PATH_SEPARATOR } from '$lib/constants/mcp-resource';
import { TRAILING_SLASHES_REGEX } from '$lib/constants/url';

export function lastPathSegment(p: string): string {
	const trimmed = p.replace(TRAILING_SLASHES_REGEX, '');
	const idx = trimmed.lastIndexOf(PATH_SEPARATOR);

	return idx === -1 ? trimmed : trimmed.slice(idx + 1);
}

// `~/...` under `home`; falls back to the basename when home is unknown
// or the path is outside it.
export function abbreviateWorkingDir(
	path: string | null | undefined,
	home: string | null | undefined
): string {
	if (!path) return '';

	if (!home) return lastPathSegment(path);

	if (path === home) return HOME_TILDE;

	if (path.startsWith(home + PATH_SEPARATOR))
		return HOME_TILDE_PREFIX + path.slice(home.length + 1);

	return lastPathSegment(path);
}

// Unlike abbreviateWorkingDir, paths outside `home` are returned
// unchanged - used where the full path matters.
export function abbreviateHome(path: string, home: string | null | undefined): string {
	if (!home) return path;

	if (path === home) return HOME_TILDE;

	if (path.startsWith(home + PATH_SEPARATOR))
		return HOME_TILDE_PREFIX + path.slice(home.length + 1);

	return path;
}

export { CWD_CHANGED_PREFIX, CWD_CLEARED_TEXT } from '$lib/constants';

export interface CwdMessageInfo {
	// absolute server-side path, null when the cwd was cleared
	path: string | null;
	// display form shown in the UI (e.g. ~/Documents)
	display: string;
}

/**
 * Format a synthetic cwd-change message. The path travels as
 * `[file:///abs/path](display)` so both the absolute and short form are
 * visible to the model and parseable back by the UI.
 */
export function formatCwdMessage(cwd: string, home: string | null): string {
	const display = abbreviateWorkingDir(cwd, home);

	return `${CWD_CHANGED_PREFIX}[${FILE_URI_PREFIX}${cwd}](${display}).`;
}

/**
 * Parse a synthetic cwd message back into its parts. The caller must
 * already know the message is synthetic (via the persisted `isSynthetic`
 * flag); this only extracts the path.
 */
export function parseCwdMessage(content: string): CwdMessageInfo | null {
	const trimmed = content.trim();

	if (trimmed === CWD_CLEARED_TEXT) {
		return { display: '', path: null };
	}

	if (trimmed.startsWith(CWD_CHANGED_PREFIX)) {
		const rest = trimmed.slice(CWD_CHANGED_PREFIX.length);
		// not anchored to the end: guidance may follow the link
		const link = rest.match(CWD_LINK_REGEX);

		if (link) return { display: link[2], path: link[1] };

		return { display: rest, path: rest };
	}

	return null;
}
