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

import type { Plugin } from 'unified';
import type { Root, Element, ElementContent } from 'hast';
import { visit } from 'unist-util-visit';
import {
	SVG_WRAPPER_CLASS,
	SVG_SCROLL_CONTAINER_CLASS,
	SVG_BLOCK_CLASS,
	SVG_LANGUAGE,
	SVG_SOURCE_ATTR,
	SVG_ID_ATTR,
	DIAGRAM_VIEW_MODE_ATTR,
	DIAGRAM_VIEW_RENDERED
} from '$lib/constants';
import type { DiagramPreData } from './pre-transform';
import {
	createBlockHeader,
	createCopyButton,
	createPreviewButton,
	createToggleSourceButton,
	createSourceView,
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
				[SVG_SOURCE_ATTR]: svgSource,
				[SVG_ID_ATTR]: svgId
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
					[SVG_ID_ATTR]: svgId,
					[DIAGRAM_VIEW_MODE_ATTR]: DIAGRAM_VIEW_RENDERED
				},
				[sourceView]
			);

			// Replace pre with wrapper in parent
			(parent.children as ElementContent[])[index] = wrapper;
		});
	};
};
