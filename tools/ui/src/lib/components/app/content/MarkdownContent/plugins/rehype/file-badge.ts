/**
 * Rehype plugin that rewrites `file://` markdown anchors into the
 * inline chip used by `MentionBadge.svelte`. The badge class string
 * is shared with the Svelte component (and with the contenteditable
 * tokenizer) via `@mention-badge`, so the model gutter in the chat
 * input, the assistant reply, and the user-bubble preview all show
 * the same chip.
 *
 * Why a `rehype` step (not a markdown-it rule): the AST after
 * remark-rehype already represents `[label](file://path)` as a
 * single `<a>` element with a text-child label, which is exactly
 * the structural shape we need to rewrite. Going earlier would
 * require manipulating raw text and re-parsing.
 */

import {
	MENTION_BADGE_CLASSNAME,
	MENTION_BADGE_ICON_CLASSNAME,
	getMentionBadgeIconPaths
} from '$lib/utils';
import type { Plugin } from 'unified';
import type { Root, Element } from 'hast';
import { visit } from 'unist-util-visit';

/**
 * Derive a friendly label from a `file://` URL. We strip the
 * `file://` prefix and any trailing separator so `[tools/]`
 * rather than `[tools//]`.
 */
function labelFromFileUrl(href: string): string {
	const stripped = href.startsWith('file://') ? href.slice('file://'.length) : href;
	const trimmed = stripped.replace(/\/+$/, '');
	const slash = trimmed.lastIndexOf('/');
	return slash === -1 ? trimmed : trimmed.slice(slash + 1);
}

/**
 * Build the inline icon as a hast `<svg>` element tree. Mirrors the
 * lucide component picked by `MentionBadge.svelte` so the Svelte-
 * rendered and DOM-built paths produce visually identical output.
 *
 * @param href - The `file://` link target; a trailing `/` marks a
 * directory and selects the folder icon, matching the convention
 * the mention picker uses when it inserts the badge.
 */
function iconElement(href: string): Element {
	return {
		type: 'element',
		tagName: 'svg',
		properties: {
			xmlns: 'http://www.w3.org/2000/svg',
			viewBox: '0 0 24 24',
			fill: 'none',
			stroke: 'currentColor',
			'stroke-width': 2,
			'stroke-linecap': 'round',
			'stroke-linejoin': 'round',
			'aria-hidden': 'true',
			className: MENTION_BADGE_ICON_CLASSNAME.split(' ').filter(Boolean)
		},
		children: getMentionBadgeIconPaths(href).map((d) => ({
			type: 'element',
			tagName: 'path',
			properties: { d },
			children: []
		}))
	};
}

export const rehypeFileBadge: Plugin<[], Root> = () => {
	return (tree: Root) => {
		visit(tree, 'element', (node: Element) => {
			if (node.tagName !== 'a') return;

			const props = node.properties ?? {};
			const href = typeof props.href === 'string' ? props.href : null;

			if (!href || !href.startsWith('file://')) return;

			const label = labelFromFileUrl(href);
			const titleAttr = typeof props.title === 'string' ? props.title : href;

			node.tagName = 'span';
			node.properties = {
				className: MENTION_BADGE_CLASSNAME.split(' ').filter(Boolean),
				role: 'link',
				tabIndex: 0,
				'data-href': href,
				title: titleAttr
			};
			node.children = [
				iconElement(href),
				{
					type: 'element',
					tagName: 'span',
					properties: { className: ['shrink-0', 'truncate'] },
					children: [{ type: 'text', value: label }]
				}
			];
		});
	};
};
