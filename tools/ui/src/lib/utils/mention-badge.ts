import { abbreviateHome, lastPathSegment } from './path-display';
import { FILE_URI_PREFIX } from '$lib/constants';
import {
	MENTION_BADGE_FILE_ICON_PATHS,
	MENTION_BADGE_FOLDER_ICON_PATHS
} from '$lib/constants/mention-badge';
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

// Escape each path segment for a markdown link destination (spaces/parens
// break CommonMark); keeps the trailing slash that marks a directory.
export function encodeFileLinkPath(path: string): string {
	return path
		.split('/')
		.map((segment) => encodeURIComponent(segment))
		.join('/');
}

// Malformed escape sequences fall back to the input unchanged.
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

export function getMentionBadgeIconPaths(path: string): readonly string[] {
	return path.endsWith('/') ? MENTION_BADGE_FOLDER_ICON_PATHS : MENTION_BADGE_FILE_ICON_PATHS;
}

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
 * already rooted, so `file://` + `/abs` yields the canonical `file:///`.
 * Null when the token is invalid.
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

	return { caretOffset: token.start + insertion.length, newValue };
}
