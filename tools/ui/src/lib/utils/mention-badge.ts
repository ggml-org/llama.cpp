import { abbreviateHome, lastPathSegment } from './path-display';
import {
	MENTION_BADGE_FILE_ICON_PATHS,
	MENTION_BADGE_FOLDER_ICON_PATHS
} from '$lib/constants/mention-badge';
import { FILE_URI_PREFIX } from '$lib/constants';
import { FileMentionEntryType } from '$lib/enums';
import type { FileMentionEntry } from '$lib/types';

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
 * Escape each path segment for a `[name](file://path)` destination;
 * spaces/parens would otherwise break CommonMark. Keeps the trailing
 * slash that marks a directory.
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

// A trailing `/` in the link target marks a directory; it survives
// copy/paste, so pick the glyph from the raw path.
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

/**
 * Build the markdown link that replaces a mention token. Entry `path` is
 * already rooted, so `file://` + `/abs` yields the canonical `file:///`;
 * directories keep a trailing `/`. Cursor lands right after the trailing
 * space so typing can continue. Returns null when no token matches.
 */
export function buildMentionInsertion(
	entry: FileMentionEntry,
	value: string,
	token: { start: number; end: number }
): { newValue: string; caretOffset: number } | null {
	if (token.start < 0 || token.end > value.length || token.start > token.end) return null;
	// Strip the entry's directory marker so it is not doubled below.
	const cleanedPath = entry.path.replace(/\/+$/, '');
	const pathWithSeparator =
		entry.type === FileMentionEntryType.DIRECTORY ? `${cleanedPath}/` : cleanedPath;
	const basename = lastPathSegment(cleanedPath) || entry.name;
	const insertion = `[${basename}](${FILE_URI_PREFIX}${encodeFileLinkPath(pathWithSeparator)}) `;
	const newValue = value.slice(0, token.start) + insertion + value.slice(token.end);
	return { newValue, caretOffset: token.start + insertion.length };
}
