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

import type { Plugin } from 'unified';
import type { Root, Element, ElementContent } from 'hast';
import { visit } from 'unist-util-visit';
import {
	CODE_BLOCK_SCROLL_CONTAINER_CLASS,
	CODE_BLOCK_WRAPPER_CLASS,
	CODE_BLOCK_HEADER_CLASS,
	CODE_BLOCK_ACTIONS_CLASS,
	CODE_LANGUAGE_CLASS,
	COPY_CODE_BTN_CLASS,
	PREVIEW_CODE_BTN_CLASS,
	RELATIVE_CLASS,
	COPY_ICON_SVG,
	PREVIEW_ICON_SVG
} from '$lib/constants';

declare global {
	interface Window {
		idxCodeBlock?: number;
	}
}

function createIconElement(svg: string): Element {
	return {
		type: 'element',
		tagName: 'span',
		properties: {},
		children: [{ type: 'raw', value: svg } as unknown as ElementContent]
	};
}

function createButton(className: string, title: string, iconSvg: string, codeId: string): Element {
	return {
		type: 'element',
		tagName: 'button',
		properties: {
			className: [className],
			'data-code-id': codeId,
			title,
			type: 'button'
		},
		children: [createIconElement(iconSvg)]
	};
}

function createCopyButton(codeId: string): Element {
	return createButton(COPY_CODE_BTN_CLASS, 'Copy code', COPY_ICON_SVG, codeId);
}

function createPreviewButton(codeId: string): Element {
	return createButton(PREVIEW_CODE_BTN_CLASS, 'Preview code', PREVIEW_ICON_SVG, codeId);
}

function createHeader(language: string, codeId: string): Element {
	const actions: Element[] = [createCopyButton(codeId)];

	if (language.toLowerCase() === 'html') {
		actions.push(createPreviewButton(codeId));
	}

	return {
		type: 'element',
		tagName: 'div',
		properties: { className: [CODE_BLOCK_HEADER_CLASS] },
		children: [
			{
				type: 'element',
				tagName: 'span',
				properties: { className: [CODE_LANGUAGE_CLASS] },
				children: [{ type: 'text', value: language }]
			},
			{
				type: 'element',
				tagName: 'div',
				properties: { className: [CODE_BLOCK_ACTIONS_CLASS] },
				children: actions
			}
		]
	};
}

function createScrollContainer(preElement: Element): Element {
	return {
		type: 'element',
		tagName: 'div',
		properties: { className: [CODE_BLOCK_SCROLL_CONTAINER_CLASS] },
		children: [preElement]
	};
}

function createWrapper(header: Element, preElement: Element): Element {
	return {
		type: 'element',
		tagName: 'div',
		properties: { className: [CODE_BLOCK_WRAPPER_CLASS, RELATIVE_CLASS] },
		children: [header, createScrollContainer(preElement)]
	};
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
 * Generates a unique code block ID using a global counter.
 */
function generateCodeId(): string {
	if (typeof window !== 'undefined') {
		return `code-${(window.idxCodeBlock = (window.idxCodeBlock ?? 0) + 1)}`;
	}
	// Fallback for SSR - use timestamp + random
	return `code-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
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
			const codeId = generateCodeId();

			codeElement.properties = {
				...codeElement.properties,
				'data-code-id': codeId
			};

			const header = createHeader(language, codeId);
			const wrapper = createWrapper(header, node);

			// Replace pre with wrapper in parent
			(parent.children as ElementContent[])[index] = wrapper;
		});
	};
};
