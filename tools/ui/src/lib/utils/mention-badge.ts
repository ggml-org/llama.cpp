import { abbreviateHome } from './path-display';

/**
 * Visual contract shared between `MentionBadge.svelte` and the two
 * DOM-only paths (the contenteditable tokenizer + the rehype plugin
 * that renders `file://` anchors inside `MarkdownContent`). Svelte
 * cannot be mounted at the per-keystroke tokenizer hot path, nor
 * from within a hast tree, so both paths emit the badge with the
 * exact same class string literal as the component. Tailwind's
 * content scanner picks the literal up in all three sources, which
 * is what keeps the styles in sync without runtime mounting.
 */

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

export const MENTION_BADGE_CLASSNAME =
	'inline-flex w-fit shrink-0 items-center gap-1 whitespace-nowrap rounded-md border border-border/50 bg-foreground/5 px-1.5 py-0.5 text-xs font-mono text-foreground hover:bg-foreground/10 dark:bg-foreground/10 dark:text-secondary-foreground';

export const MENTION_BADGE_ICON_CLASSNAME = 'h-3 w-3 shrink-0';

/**
 * SVG attributes shared by the DOM-built and hast-built badge icons so
 * both match `MentionBadge.svelte`'s lucide component. The tokenizer
 * applies them via `setAttribute`, the rehype plugin spreads them onto
 * the hast `<svg>` `properties`; string values are valid for both.
 */
export const MENTION_BADGE_SVG_ATTRIBUTES: Readonly<Record<string, string>> = {
	xmlns: 'http://www.w3.org/2000/svg',
	viewBox: '0 0 24 24',
	fill: 'none',
	stroke: 'currentColor',
	'stroke-width': '2',
	'stroke-linecap': 'round',
	'stroke-linejoin': 'round',
	'aria-hidden': 'true'
};

/**
 * SVG path strings for the badge's inline icon. Each entry becomes
 * one `<path>` child of the wrapper `<svg>` so the DOM-built and
 * hast-built badges are visually identical to `MentionBadge.svelte`'s
 * lucide component. Paths match `lucide-svelte`'s current `File` and
 * `Folder` glyphs. Used by both `contenteditable-tokenizer.ts`
 * (which calls `createElementNS`) and
 * `MarkdownContent/plugins/rehype/file-badge.ts` (which builds a
 * hast `<svg>` node).
 */
export const MENTION_BADGE_FILE_ICON_PATHS: readonly string[] = [
	'M6 22a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h8a2.4 2.4 0 0 1 1.704.706l3.588 3.588A2.4 2.4 0 0 1 20 8v12a2 2 0 0 1-2 2z',
	'M14 2v5a1 1 0 0 0 1 1h5'
];

export const MENTION_BADGE_FOLDER_ICON_PATHS: readonly string[] = [
	'M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z'
];

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
