/**
 * Rehype plugin to enhance svg blocks with wrapper, header, and action buttons.
 *
 * Wraps <pre class="svg-block"> elements with a container that includes:
 * - Language label ("svg")
 * - Copy button (copies svg source to clipboard)
 * - Preview button (opens fullscreen preview dialog)
 *
 * Operates directly on the HAST tree and reuses the shared code-block builders.
 */

import {
	createBlockHeader,
	createCopyButton,
	createPreviewButton,
	createSourceView,
	createToggleSourceButton,
	createWrapper,
	generateBlockId
} from './code-block-utils';
import type { DiagramPreData } from './pre-transform';
import {
	DIAGRAM_VIEW_MODE_ATTR,
	DIAGRAM_VIEW_RENDERED,
	SVG_BLOCK_CLASS,
	SVG_ID_ATTR,
	SVG_LANGUAGE,
	SVG_SCROLL_CONTAINER_CLASS,
	SVG_SOURCE_ATTR,
	SVG_WRAPPER_CLASS
} from '$lib/constants';
import type { Element, ElementContent, Root } from 'hast';
import type { Plugin } from 'unified';
import { visit } from 'unist-util-visit';

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

			const isSvg = className.some((cls) => typeof cls === 'string' && cls === SVG_BLOCK_CLASS);

			if (!isSvg) return;

			const svgId = generateBlockId(SVG_LANGUAGE, 'idxSvgBlock');
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
				[SVG_ID_ATTR]: svgId,
				[SVG_SOURCE_ATTR]: svgSource
			};

			const actions = [
				createCopyButton(svgId, SVG_ID_ATTR, 'Copy svg source'),
				createToggleSourceButton(svgId, SVG_ID_ATTR, 'Toggle svg source'),
				createPreviewButton(svgId, SVG_ID_ATTR, 'Preview svg')
			];
			const header = createBlockHeader(SVG_LANGUAGE, svgId, SVG_ID_ATTR, actions);
			const preservedCode = (node.data as DiagramPreData | undefined)?.sourceCode;
			const sourceView = createSourceView(preservedCode, svgSource, SVG_LANGUAGE);
			const wrapper = createWrapper(
				header,
				node,
				SVG_WRAPPER_CLASS,
				SVG_SCROLL_CONTAINER_CLASS,
				{
					[DIAGRAM_VIEW_MODE_ATTR]: DIAGRAM_VIEW_RENDERED,
					[SVG_ID_ATTR]: svgId
				},
				[sourceView]
			);

			// Replace pre with wrapper in parent
			(parent.children as ElementContent[])[index] = wrapper;
		});
	};
};
