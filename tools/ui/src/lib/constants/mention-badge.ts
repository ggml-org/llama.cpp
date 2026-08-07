/**
 * Visual contract for message @-mention badges. Svelte cannot be mounted
 * from a hast tree, so the rehype file-badge plugin emits the shared class
 * string below; keeping it here as a literal lets Tailwind's source
 * scanner generate the utility classes.
 */
export const MENTION_BADGE_CLASSNAME =
	'inline-flex w-fit shrink-0 items-center gap-1 whitespace-nowrap rounded-md border border-border/50 bg-foreground/5 px-1.5 py-0.5 text-xs font-mono text-foreground hover:bg-foreground/10 dark:bg-foreground/10 dark:text-secondary-foreground';

export const MENTION_BADGE_ICON_CLASSNAME = 'h-3 w-3 shrink-0';

/**
 * SVG attributes shared by the hast-built badge icons; the rehype plugin
 * spreads them onto the `<svg>` `properties`.
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
 * SVG path strings for the badge's inline icon; each entry becomes one
 * `<path>` child of the wrapper `<svg>`. Paths match `lucide-svelte`'s
 * current `File` and `Folder` glyphs.
 */
export const MENTION_BADGE_FILE_ICON_PATHS: readonly string[] = [
	'M6 22a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h8a2.4 2.4 0 0 1 1.704.706l3.588 3.588A2.4 2.4 0 0 1 20 8v12a2 2 0 0 1-2 2z',
	'M14 2v5a1 1 0 0 0 1 1h5'
];

export const MENTION_BADGE_FOLDER_ICON_PATHS: readonly string[] = [
	'M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z'
];
