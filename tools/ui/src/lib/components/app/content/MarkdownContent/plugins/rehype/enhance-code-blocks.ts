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

import {
	CODE_BLOCK,
	CODE_BLOCK_CLASS,
	CODE_BLOCK_ATTR,
	CODE_BLOCK_TEXT
} from '$lib/constants';

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
				node.data.hProperties[CODE_BLOCK_ATTR.META_DATA] = node.meta;
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
	const meta = node.properties?.[CODE_BLOCK_ATTR.META_DATA] ?? codeElement.properties?.[CODE_BLOCK_ATTR.META_DATA];
	if (typeof meta !== 'string') return undefined;

	// Matches a file extension and name, optionally preceded by a key like name=, file=, or title=
	const regex = CODE_BLOCK.FILE_NAME_REGEX;
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
				[CODE_BLOCK_ATTR.CODE_ID]: codeId
			};

			const actions: Element[] = [
				createDownloadButton(codeId, CODE_BLOCK_ATTR.CODE_ID, CODE_BLOCK_TEXT.DOWNLOAD_BTN_TITLE),
				createCopyButton(codeId, CODE_BLOCK_ATTR.CODE_ID, CODE_BLOCK_TEXT.COPY_BTN_TITLE)
			];

			if (language.toLowerCase() === 'html') {
				actions.push(createPreviewButton(codeId, CODE_BLOCK_ATTR.CODE_ID, CODE_BLOCK_TEXT.PREVIEW_TITLE));
			}

			const header = createBlockHeader(language, codeId, CODE_BLOCK_ATTR.CODE_ID, actions);
			const wrapper = createWrapper(
				header,
				node,
				CODE_BLOCK_CLASS.WRAPPER,
				CODE_BLOCK_CLASS.SCROLL_CONTAINER,
				{
					...(filename ? { [CODE_BLOCK_ATTR.FILE_NAME] : filename } : {})
				}
			);

			// Replace pre with wrapper in parent
			(parent.children as ElementContent[])[index] = wrapper;
		});
	};
};
