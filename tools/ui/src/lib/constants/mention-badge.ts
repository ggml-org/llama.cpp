/**
 * Shared visual contract between the two DOM-only badge paths (the
 * contenteditable tokenizer + the rehype plugin). Svelte cannot be
 * mounted at the per-keystroke tokenizer hot path nor from a hast tree,
 * so both emit the badge with the same class string literal; Tailwind's
 * scanner picks it up in both sources.
 */
export const MENTION_BADGE_CLASSNAME =
	'inline-flex w-fit shrink-0 items-center gap-1 whitespace-nowrap rounded-md border border-border/50 bg-foreground/5 px-1.5 py-0.5 text-xs font-mono text-foreground hover:bg-foreground/10 dark:bg-foreground/10 dark:text-secondary-foreground';

export const MENTION_BADGE_ICON_CLASSNAME = 'h-3 w-3 shrink-0';

/**
 * SVG attributes shared by the DOM-built and hast-built badge icons.
 * The tokenizer applies them via `setAttribute`, the rehype plugin
 * spreads them onto the hast `<svg>` `properties`; string values are
 * valid for both.
 */
export const MENTION_BADGE_SVG_ATTRIBUTES: Readonly<Record<string, string>> = {
	'aria-hidden': 'true',
	fill: 'none',
	stroke: 'currentColor',
	'stroke-linecap': 'round',
	'stroke-linejoin': 'round',
	'stroke-width': '2',
	viewBox: '0 0 24 24',
	xmlns: 'http://www.w3.org/2000/svg'
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
