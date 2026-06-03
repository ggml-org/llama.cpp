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
	MERMAID_SCROLL_CONTAINER_CLASS,
	COPY_ICON_SVG,
	PREVIEW_ICON_SVG
} from '$lib/constants';

declare global {
	interface Window {
		idxMermaidBlock?: number;
	}
}

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
