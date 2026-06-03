/**
 * Rehype plugin to enhance mermaid diagram blocks with wrapper, header, and action buttons.
 *
 * Wraps <pre class="mermaid"> elements with a container that includes:
 * - Language label ("mermaid")
 * - Copy button (copies mermaid syntax to clipboard)
 * - Preview button (opens fullscreen preview dialog)
 *
 * This operates directly on the HAST tree for better performance,
 * avoiding the need to stringify and re-parse HTML.
 */

import type { Plugin } from 'unified';
import type { Root, Element, ElementContent } from 'hast';
import { visit } from 'unist-util-visit';
import { MERMAID_WRAPPER_CLASS, MERMAID_SCROLL_CONTAINER_CLASS } from '$lib/constants';
import {
	createBlockHeader,
	createCopyButton,
	createPreviewButton,
	createWrapper,
	generateBlockId
} from './code-block-utils';

declare global {
	interface Window {
		idxMermaidBlock?: number;
	}
}

/**
 * Rehype plugin to enhance mermaid diagram blocks with wrapper, header, and action buttons.
 * This plugin wraps <pre class="mermaid"> elements with a container that includes:
 * - Language label ("mermaid")
 * - Copy button
 * - Preview button
 */
export const rehypeEnhanceMermaidBlocks: Plugin<[], Root> = () => {
	return (tree: Root) => {
		visit(tree, 'element', (node: Element, index, parent) => {
			if (node.tagName !== 'pre' || !parent || index === undefined) return;

			const className = node.properties?.className;
			if (!Array.isArray(className)) return;

			const isMermaid = className.some((cls) => typeof cls === 'string' && cls === 'mermaid');

			if (!isMermaid) return;

			const mermaidId = generateBlockId('mermaid', 'idxMermaidBlock');

			// Extract the mermaid syntax (text content of the pre element)
			const diagramText = node.children
				.map((child) => {
					if (child.type === 'text') return child.value;
					return '';
				})
				.join('');

			// Store the mermaid syntax in data attribute for copy functionality
			node.properties = {
				...node.properties,
				'data-mermaid-syntax': diagramText,
				'data-mermaid-id': mermaidId
			};

			const actions = [
				createCopyButton(mermaidId, 'data-mermaid-id', 'Copy mermaid syntax'),
				createPreviewButton(mermaidId, 'data-mermaid-id', 'Preview diagram')
			];

			const header = createBlockHeader('mermaid', mermaidId, 'data-mermaid-id', actions);
			const wrapper = createWrapper(
				header,
				node,
				MERMAID_WRAPPER_CLASS,
				MERMAID_SCROLL_CONTAINER_CLASS,
				{ 'data-mermaid-id': mermaidId }
			);

			// Replace pre with wrapper in parent
			(parent.children as ElementContent[])[index] = wrapper;
		});
	};
};
