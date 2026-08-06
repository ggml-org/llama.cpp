import { abbreviateHome } from './path-display';
import {
	MENTION_BADGE_FILE_ICON_PATHS,
	MENTION_BADGE_FOLDER_ICON_PATHS
} from '$lib/constants/mention-badge';

export {
	MENTION_BADGE_CLASSNAME,
	MENTION_BADGE_ICON_CLASSNAME,
	MENTION_BADGE_SVG_ATTRIBUTES,
	MENTION_BADGE_FILE_ICON_PATHS,
	MENTION_BADGE_FOLDER_ICON_PATHS
} from '$lib/constants/mention-badge';

// `)` is allowed in a path only when not followed by whitespace or `[`,
// so macOS paths parse while adjacent badges still terminate the match.
const FILE_MENTION_LINK_SOURCE = String.raw`\[([^\]\n]+?)\]\(file:\/\/((?:[^)\n]|\)(?![\s[]))+)\)`;

export function fileMentionLinkRe(flags = ''): RegExp {
	return new RegExp(FILE_MENTION_LINK_SOURCE, flags);
}

export function containsFileMentionLink(value: string): boolean {
	return fileMentionLinkRe().test(value);
}

/**
 * Encode a filesystem path into a `[name](file://path)` destination.
 * Spaces/parens break CommonMark (the badge renders as raw source), so
 * each segment is escaped; leading/trailing slashes are kept (trailing
 * marks a directory in the picker).
 */
export function encodeFileLinkPath(path: string): string {
	return path
		.split('/')
		.map((segment) => encodeURIComponent(segment))
		.join('/');
}

/**
 * Inverse of `encodeFileLinkPath`; malformed escape sequences fall back
 * to the input unchanged.
 */
export function decodeFileLinkPath(path: string): string {
	try {
		return path
			.split('/')
			.map((segment) => decodeURIComponent(segment))
			.join('/');
	} catch {
		return path;
	}
}

// Directories are marked by a trailing `/` in the link target; this
// survives copy/paste, so pick the glyph from the raw path.
export function getMentionBadgeIconPaths(path: string): readonly string[] {
	return path.endsWith('/') ? MENTION_BADGE_FOLDER_ICON_PATHS : MENTION_BADGE_FILE_ICON_PATHS;
}

/**
 * Visible label: `name`, or the decoded full path (trailing `/` stripped)
 * when `showFullPath`; a known `home` abbreviates to `~`.
 */
export function getMentionBadgeLabel(
	name: string,
	path: string,
	showFullPath: boolean,
	home?: string | null
): string {
	if (!showFullPath) return name;
	const decoded = decodeFileLinkPath(path.replace(/\/+$/, ''));
	if (!decoded) return name;
	return abbreviateHome(decoded, home);
}
