/**
 * Rehype plugin to enhance code blocks with wrapper, header, and action buttons.
 *
 * Wraps <pre><code> elements with a container that includes:
 * - Language label
 * - Copy button
 * - Download button
 * - Preview button (for HTML code blocks)
 *
 * This operates directly on the HAST tree for better performance,
 * avoiding the need to stringify and re-parse HTML.
 */

import {
	createBlockHeader,
	createCopyButton,
	createDownloadButton,
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

/**
 * Remark plugin to preserve the code block meta string.
 * remark-rehype drops the meta string by default, so we save it to data-meta
 * to access it later in rehypeEnhanceCodeBlocks.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const remarkPreserveCodeMeta: Plugin<[], any> = () => {
	return (tree) => {
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		visit(tree, 'code', (node: any) => {
			if (node.meta) {
				node.data = node.data || {};
				node.data.hProperties = node.data.hProperties || {};
				node.data.hProperties['data-meta'] = node.meta;
			}
		});
	};
};

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
 * Extracts a filename from the code block's info string if present.
 * Matches patterns like title="app.js", file=app.js, "app.js", `app.js`, or just app.js
 */
function extractFilenameFromInfo(node: Element, codeElement: Element): string | undefined {
	const meta = node.properties?.['data-meta'] ?? codeElement.properties?.['data-meta'];
	if (typeof meta !== 'string') return undefined;

	// Matches a file extension and name, optionally preceded by a key like name=, file=, or title=
	// and optionally wrapped in quotes or backticks.
	// Requires the extension to start with a letter to avoid capturing semantic versions like "1.0.0".
	const regex = /(?:^|\s)(?:[a-zA-Z0-9_-]+=)?["'`]?([^"'`\s]+\.[a-zA-Z][a-zA-Z0-9]{0,6})["'`]?(?:\s|$)/i;
	const match = meta.match(regex);
	return match ? match[1] : undefined;
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
			const filename = extractFilenameFromInfo(node, codeElement);
			const codeId = generateBlockId('code', 'idxCodeBlock');

			codeElement.properties = {
				...codeElement.properties,
				'data-code-id': codeId
			};

			const actions: Element[] = [
				createDownloadButton(codeId, 'data-code-id', 'Download'),
				createCopyButton(codeId, 'data-code-id', 'Copy code')
			];

			if (language.toLowerCase() === 'html') {
				actions.push(createPreviewButton(codeId, 'data-code-id', 'Preview code'));
			}

			const header = createBlockHeader(language, codeId, 'data-code-id', actions);
			const wrapper = createWrapper(
				header,
				node,
				CODE_BLOCK_CLASS.WRAPPER,
				CODE_BLOCK_CLASS.SCROLL_CONTAINER,
				{
					...(filename ? { 'data-filename': filename } : {})
				}
			);

			// Replace pre with wrapper in parent
			(parent.children as ElementContent[])[index] = wrapper;
		});
	};
};
