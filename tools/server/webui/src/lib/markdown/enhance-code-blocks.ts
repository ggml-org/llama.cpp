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

declare global {
	interface Window {
		idxCodeBlock?: number;
	}
}

const COPY_ICON_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-copy-icon lucide-copy"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>`;

const PREVIEW_ICON_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-eye lucide-eye-icon"><path d="M2.062 12.345a1 1 0 0 1 0-.69C3.5 7.73 7.36 5 12 5s8.5 2.73 9.938 6.655a1 1 0 0 1 0 .69C20.5 16.27 16.64 19 12 19s-8.5-2.73-9.938-6.655"/><circle cx="12" cy="12" r="3"/></svg>`;

/**
 * Creates an SVG element node from raw SVG string.
 * Since we can't parse HTML in HAST directly, we use the raw property.
 */
function createRawHtmlElement(html: string): Element {
	return {
		type: 'element',
		tagName: 'span',
		properties: {},
		children: [{ type: 'raw', value: html } as unknown as ElementContent]
	};
}

function createCopyButton(codeId: string): Element {
	return {
		type: 'element',
		tagName: 'button',
		properties: {
			className: ['copy-code-btn'],
			'data-code-id': codeId,
			title: 'Copy code',
			type: 'button'
		},
		children: [createRawHtmlElement(COPY_ICON_SVG)]
	};
}

function createPreviewButton(codeId: string): Element {
	return {
		type: 'element',
		tagName: 'button',
		properties: {
			className: ['preview-code-btn'],
			'data-code-id': codeId,
			title: 'Preview code',
			type: 'button'
		},
		children: [createRawHtmlElement(PREVIEW_ICON_SVG)]
	};
}

function createHeader(language: string, codeId: string): Element {
	const actions: Element[] = [createCopyButton(codeId)];

	if (language.toLowerCase() === 'html') {
		actions.push(createPreviewButton(codeId));
	}

	return {
		type: 'element',
		tagName: 'div',
		properties: { className: ['code-block-header'] },
		children: [
			{
				type: 'element',
				tagName: 'span',
				properties: { className: ['code-language'] },
				children: [{ type: 'text', value: language }]
			},
			{
				type: 'element',
				tagName: 'div',
				properties: { className: ['code-block-actions'] },
				children: actions
			}
		]
	};
}

function createWrapper(header: Element, preElement: Element): Element {
	return {
		type: 'element',
		tagName: 'div',
		properties: { className: ['code-block-wrapper'] },
		children: [header, preElement]
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
