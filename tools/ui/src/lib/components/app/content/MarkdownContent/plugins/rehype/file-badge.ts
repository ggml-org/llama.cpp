/**
 * Rehype plugin that rewrites `file://` markdown anchors into the inline
 * mention chip, sharing the class string with the contenteditable
 * tokenizer via `$lib/constants`.
 *
 * The chip is presentational: `file://` navigation is blocked from
 * http(s) pages, so the anchor becomes a plain `<span>` (no link role,
 * no tab stop); the full path stays available on `title`.
 */

import {
	FILE_URI_PREFIX,
	MENTION_BADGE_CLASSNAME,
	MENTION_BADGE_ICON_CLASSNAME,
	MENTION_BADGE_SVG_ATTRIBUTES,
	PATH_SEPARATOR,
	SETTINGS_KEYS
} from '$lib/constants';
import { settingsStore, toolsStore } from '$lib/stores';
import { decodeFileLinkPath, getMentionBadgeIconPaths, getMentionBadgeLabel } from '$lib/utils';
import type { Element, Root } from 'hast';
import type { Plugin } from 'unified';
import { visit } from 'unist-util-visit';

// Trailing path separators mark a directory and are kept out of the label.
const TRAILING_SEPARATOR_REGEX = /\/+$/;

function decodeHrefPath(href: string): string {
	const stripped = href.startsWith(FILE_URI_PREFIX) ? href.slice(FILE_URI_PREFIX.length) : href;

	return decodeFileLinkPath(stripped);
}

function labelFromFileUrl(href: string): string {
	const decoded = decodeHrefPath(href);
	const trimmed = decoded.replace(TRAILING_SEPARATOR_REGEX, '');
	const slash = trimmed.lastIndexOf(PATH_SEPARATOR);

	return slash === -1 ? trimmed : trimmed.slice(slash + 1);
}

// A trailing `/` in the target marks a directory and selects the folder
// icon, matching the convention the mention picker inserts with.
function iconElement(href: string): Element {
	return {
		children: getMentionBadgeIconPaths(href).map((d) => ({
			children: [],
			properties: { d },
			tagName: 'path',
			type: 'element'
		})),
		properties: {
			...MENTION_BADGE_SVG_ATTRIBUTES,
			className: MENTION_BADGE_ICON_CLASSNAME.split(' ').filter(Boolean)
		},
		tagName: 'svg',
		type: 'element'
	};
}

export const rehypeFileBadge: Plugin<[], Root> = () => {
	return (tree: Root) => {
		visit(tree, 'element', (node: Element) => {
			if (node.tagName !== 'a') return;

			const props = node.properties ?? {};
			const href = typeof props.href === 'string' ? props.href : null;

			if (!href || !href.startsWith(FILE_URI_PREFIX)) return;

			const label = labelFromFileUrl(href);
			const titleAttr = typeof props.title === 'string' ? props.title : href;
			const decodedPath = decodeHrefPath(href);

			node.tagName = 'span';
			node.properties = {
				className: MENTION_BADGE_CLASSNAME.split(' ').filter(Boolean),
				title: titleAttr.startsWith(FILE_URI_PREFIX) ? decodedPath : titleAttr
			};
			node.children = [
				iconElement(href),
				{
					children: [
						{
							type: 'text',
							value: getMentionBadgeLabel(
								label,
								decodedPath,
								settingsStore.getConfig(SETTINGS_KEYS.SHOW_FULL_PATH_IN_MENTIONS),
								toolsStore.serverHome
							)
						}
					],
					properties: { className: ['shrink-0', 'truncate'] },
					tagName: 'span',
					type: 'element'
				}
			];
		});
	};
};
