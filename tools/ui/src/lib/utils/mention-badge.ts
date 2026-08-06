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

// Recognizes `[name](file://path)` markdown mention links. The path allows
// `)` only when not followed by whitespace or `[` so the closing `)` of an
// adjacent badge still terminates the match, while macOS paths like
// `Screenshot (1).png` and folders named `Foo (Stuff)/bar` parse correctly.
const FILE_MENTION_LINK_SOURCE = String.raw`\[([^\]\n]+?)\]\(file:\/\/((?:[^)\n]|\)(?![\s[]))+)\)`;

export function fileMentionLinkRe(flags = ''): RegExp {
	return new RegExp(FILE_MENTION_LINK_SOURCE, flags);
}

export function containsFileMentionLink(value: string): boolean {
	return fileMentionLinkRe().test(value);
}

/**
 * Encode a filesystem path for safe embedding in a `[name](file://path)`
 * markdown link destination.
 *
 * A destination containing a space (or `()`, `<`, `>` etc.) is not valid
 * CommonMark; remark parses the whole line as plain text, so the mention
 * shows as raw source in a rendered message bubble instead of a badge.
 * Encoding each `/`-separated segment escapes those characters while
 * keeping the leading/trailing slashes (the trailing `/` marks a
 * directory for the picker).
 */
export function encodeFileLinkPath(path: string): string {
	return path
		.split('/')
		.map((segment) => encodeURIComponent(segment))
		.join('/');
}

/**
 * Inverse of `encodeFileLinkPath`. Decodes a `/`-separated encoded path
 * back to a human-readable filesystem path, tolerating malformed escape
 * sequences by falling back to the input unchanged.
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

/**
 * Pick the icon paths for a mention badge based on its `path`. The
 * mention picker encodes directories with a trailing `/` in the
 * `file://` link target; this convention survives copy/paste so the
 * icon stays correct even when a badge is reconstructed from a
 * pasted markdown source.
 */
export function getMentionBadgeIconPaths(path: string): readonly string[] {
	return path.endsWith('/') ? MENTION_BADGE_FOLDER_ICON_PATHS : MENTION_BADGE_FILE_ICON_PATHS;
}

/**
 * Compute the visible label for a mention badge. Defaults to the
 * file/folder name; with `showFullPath` it renders the full decoded
 * path (a trailing `/` marker for folders is stripped so the label
 * stays readable). A known `home` abbreviates its prefix to `~`,
 * e.g. `~/src/main.rs`, matching the working-directory chip.
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
