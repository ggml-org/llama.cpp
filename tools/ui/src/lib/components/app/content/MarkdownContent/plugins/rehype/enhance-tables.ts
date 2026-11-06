/**
 * Rehype plugin to wrap tables in a horizontal scroll container.
 *
 * A bare <table> keeps its content-driven minimum width, which propagates up
 * the layout and can stretch the chat column past the window. Wrapping in
 * div.table-wrapper makes the wrapper the scroll container (styled in
 * markdown-content.css), so wide tables scroll in place instead.
 */

import type { Element, ElementContent, Root } from 'hast';
import type { Plugin } from 'unified';
import { visit } from 'unist-util-visit';

export const rehypeEnhanceTables: Plugin<[], Root> = () => {
	return (tree: Root) => {
		visit(tree, 'element', (node: Element, index, parent) => {
			if (node.tagName !== 'table' || !parent || index === undefined) return;

			// already wrapped (e.g. nested tables in raw HTML input)
			const parentClass = parent.type === 'element' ? parent.properties?.className : undefined;

			if (Array.isArray(parentClass) && parentClass.includes('table-wrapper')) return;

			const wrapper: Element = {
				children: [node as ElementContent],
				properties: { className: ['table-wrapper'] },
				tagName: 'div',
				type: 'element'
			};

			parent.children[index] = wrapper;
		});
	};
};
