/**
 * Rehype plugin to enhance code blocks with wrapper, header, and action buttons.
 *
 * Wraps <pre><code> elements with a container that includes:
 * - Language label
 * - Copy button
 * - Preview button (for HTML code blocks)
 *
 * This operates directly on the HAST tree for better performance,
 * avoiding the need to stringify and re-parse HTML.
 */

import {
	createBlockHeader,
	createCopyButton,
	createPreviewButton,
	createWrapper,
	generateBlockId
} from './code-block-utils';
import { CODE_BLOCK_CLASS } from '$lib/constants';
import type { Element, ElementContent, Root } from 'hast';
import type { Plugin } from 'unified';
import { visit } from 'unist-util-visit';

declare global {
	interface Window {
		idxCodeBlock?: number;
	}
}

function extractLanguage(codeElement: Element): string {
	const className = codeElement.properties?.className;

	if (!Array.isArray(className)) return 'text';

	for (const cls of className) {
		if (typeof cls === 'string' && cls.startsWith('language-')) {
			return cls.replace('language-', '');
		}
	}

	return 'text';
}

/**
 * Rehype plugin to enhance code blocks with wrapper, header, and action buttons.
 * This plugin wraps <pre><code> elements with a container that includes:
 * - Language label
 * - Copy button
 * - Preview button (for HTML code blocks)
 */
export const rehypeEnhanceCodeBlocks: Plugin<[], Root> = () => {
	return (tree: Root) => {
		visit(tree, 'element', (node: Element, index, parent) => {
			if (node.tagName !== 'pre' || !parent || index === undefined) return;

			const codeElement = node.children.find(
				(child): child is Element => child.type === 'element' && child.tagName === 'code'
			);

			if (!codeElement) return;

			const language = extractLanguage(codeElement);
			const codeId = generateBlockId('code', 'idxCodeBlock');

			codeElement.properties = {
				...codeElement.properties,
				'data-code-id': codeId
			};

			const actions: Element[] = [createCopyButton(codeId, 'data-code-id', 'Copy code')];

			if (language.toLowerCase() === 'html') {
				actions.push(createPreviewButton(codeId, 'data-code-id', 'Preview code'));
			}

			const header = createBlockHeader(language, codeId, 'data-code-id', actions);
			const wrapper = createWrapper(
				header,
				node,
				CODE_BLOCK_CLASS.WRAPPER,
				CODE_BLOCK_CLASS.SCROLL_CONTAINER
			);

			// Replace pre with wrapper in parent
			(parent.children as ElementContent[])[index] = wrapper;
		});
	};
};
