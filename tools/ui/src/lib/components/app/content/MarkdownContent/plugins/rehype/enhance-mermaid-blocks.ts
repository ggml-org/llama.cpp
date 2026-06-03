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
import {
	CODE_BLOCK_HEADER_CLASS,
	CODE_BLOCK_ACTIONS_CLASS,
	CODE_LANGUAGE_CLASS,
	COPY_CODE_BTN_CLASS,
	PREVIEW_CODE_BTN_CLASS,
	RELATIVE_CLASS,
	MERMAID_WRAPPER_CLASS,
	MERMAID_SCROLL_CONTAINER_CLASS
} from '$lib/constants';

declare global {
	interface Window {
		idxMermaidBlock?: number;
	}
}

const COPY_ICON_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-copy-icon lucide-copy"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>`;

const PREVIEW_ICON_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-eye lucide-eye-icon"><path d="M2.062 12.345a1 1 0 0 1 0-.69C3.5 7.73 7.36 5 12 5s8.5 2.73 9.938 6.655a1 1 0 0 1 0 .69C20.5 16.27 16.64 19 12 19s-8.5-2.73-9.938-6.655"/><circle cx="12" cy="12" r="3"/></svg>`;

function createIconElement(svg: string): Element {
	// Use 'raw' for inline SVG - this works with rehype-stringify
	const rawContent = { type: 'raw' as const, value: svg };
	return {
		type: 'element',
		tagName: 'span',
		properties: { className: ['mermaid-btn-icon'] },
		children: [rawContent as unknown as ElementContent]
	};
}

function createButton(
	className: string,
	title: string,
	iconSvg: string,
	mermaidId: string
): Element {
	return {
		type: 'element',
		tagName: 'button',
		properties: {
			className: [className],
			'data-mermaid-id': mermaidId,
			title,
			type: 'button'
		},
		children: [createIconElement(iconSvg)]
	};
}

function createCopyButton(mermaidId: string): Element {
	return createButton(COPY_CODE_BTN_CLASS, 'Copy mermaid syntax', COPY_ICON_SVG, mermaidId);
}

function createPreviewButton(mermaidId: string): Element {
	return createButton(PREVIEW_CODE_BTN_CLASS, 'Preview diagram', PREVIEW_ICON_SVG, mermaidId);
}

function createHeader(mermaidId: string): Element {
	return {
		type: 'element',
		tagName: 'div',
		properties: { className: [CODE_BLOCK_HEADER_CLASS] },
		children: [
			{
				type: 'element',
				tagName: 'span',
				properties: { className: [CODE_LANGUAGE_CLASS] },
				children: [{ type: 'text', value: 'mermaid' }]
			},
			{
				type: 'element',
				tagName: 'div',
				properties: { className: [CODE_BLOCK_ACTIONS_CLASS] },
				children: [createCopyButton(mermaidId), createPreviewButton(mermaidId)]
			}
		]
	};
}

function createScrollContainer(preElement: Element): Element {
	return {
		type: 'element',
		tagName: 'div',
		properties: { className: [MERMAID_SCROLL_CONTAINER_CLASS] },
		children: [preElement]
	};
}

function createWrapper(header: Element, preElement: Element, mermaidId: string): Element {
	return {
		type: 'element',
		tagName: 'div',
		properties: {
			className: [MERMAID_WRAPPER_CLASS, RELATIVE_CLASS],
			'data-mermaid-id': mermaidId
		},
		children: [header, createScrollContainer(preElement)]
	};
}

/**
 * Generates a unique mermaid block ID using a global counter.
 */
function generateMermaidId(): string {
	if (typeof window !== 'undefined') {
		return `mermaid-${(window.idxMermaidBlock = (window.idxMermaidBlock ?? 0) + 1)}`;
	}
	// Fallback for SSR - use timestamp + random
	return `mermaid-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
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

			const mermaidId = generateMermaidId();

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

			const header = createHeader(mermaidId);
			const wrapper = createWrapper(header, node, mermaidId);

			// Replace pre with wrapper in parent
			(parent.children as ElementContent[])[index] = wrapper;
		});
	};
};
