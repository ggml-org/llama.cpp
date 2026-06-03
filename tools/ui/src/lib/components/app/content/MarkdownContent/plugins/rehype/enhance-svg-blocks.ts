/**
 * Rehype plugin to enhance svg diagram blocks with wrapper, header, and action buttons.
 *
 * Wraps <pre class="svg-diagram"> elements with a container that includes:
 * - Language label ("svg")
 * - Copy button (copies svg source to clipboard)
 * - Preview button (opens fullscreen preview dialog)
 *
 * Operates directly on the HAST tree and reuses the shared code-block builders.
 */

import type { Plugin } from 'unified';
import type { Root, Element, ElementContent } from 'hast';
import { visit } from 'unist-util-visit';
import { SVG_WRAPPER_CLASS, SVG_SCROLL_CONTAINER_CLASS } from '$lib/constants';
import {
	createBlockHeader,
	createCopyButton,
	createPreviewButton,
	createWrapper,
	generateBlockId
} from './code-block-utils';

declare global {
	interface Window {
		idxSvgBlock?: number;
	}
}

export const rehypeEnhanceSvgBlocks: Plugin<[], Root> = () => {
	return (tree: Root) => {
		visit(tree, 'element', (node: Element, index, parent) => {
			if (node.tagName !== 'pre' || !parent || index === undefined) return;

			const className = node.properties?.className;
			if (!Array.isArray(className)) return;

			const isSvg = className.some((cls) => typeof cls === 'string' && cls === 'svg-diagram');

			if (!isSvg) return;

			const svgId = generateBlockId('svg', 'idxSvgBlock');

			// Extract the svg source (text content of the pre element)
			const svgSource = node.children
				.map((child) => {
					if (child.type === 'text') return child.value;
					return '';
				})
				.join('');

			// Store the svg source in data attribute for copy and render
			node.properties = {
				...node.properties,
				'data-svg-source': svgSource,
				'data-svg-id': svgId
			};

			const actions = [
				createCopyButton(svgId, 'data-svg-id', 'Copy svg source'),
				createPreviewButton(svgId, 'data-svg-id', 'Preview svg')
			];

			const header = createBlockHeader('svg', svgId, 'data-svg-id', actions);
			const wrapper = createWrapper(header, node, SVG_WRAPPER_CLASS, SVG_SCROLL_CONTAINER_CLASS, {
				'data-svg-id': svgId
			});

			// Replace pre with wrapper in parent
			(parent.children as ElementContent[])[index] = wrapper;
		});
	};
};
