/**
 * Shared utilities for enhanced code blocks and mermaid diagram blocks.
 * Contains common HAST element creation functions to avoid code duplication.
 */

import type { Element, ElementContent } from 'hast';
import {
	CODE_BLOCK_HEADER_CLASS,
	CODE_BLOCK_ACTIONS_CLASS,
	CODE_BLOCK_SCROLL_CONTAINER_CLASS,
	CODE_LANGUAGE_CLASS,
	COPY_CODE_BTN_CLASS,
	PREVIEW_CODE_BTN_CLASS,
	TOGGLE_SOURCE_BTN_CLASS,
	DIAGRAM_SOURCE_CLASS,
	RELATIVE_CLASS,
	COPY_ICON_SVG,
	PREVIEW_ICON_SVG,
	CODE_ICON_SVG
} from '$lib/constants';

export interface BlockIdGenerator {
	(id: number): string;
}

/**
 * Creates an icon element with the given SVG content.
 */
export function createIconElement(svg: string): Element {
	return {
		type: 'element',
		tagName: 'span',
		properties: {},
		children: [{ type: 'raw', value: svg } as unknown as ElementContent]
	};
}

/**
 * Creates a button element with icon. Extra properties merge onto the button,
 * which lets a stateful button carry attributes like aria-pressed.
 */
export function createButton(
	className: string,
	title: string,
	iconSvg: string,
	id: string,
	idAttribute: string,
	extraProperties: Record<string, string> = {}
): Element {
	return {
		type: 'element',
		tagName: 'button',
		properties: {
			className: [className],
			[idAttribute]: id,
			title,
			type: 'button',
			...extraProperties
		},
		children: [createIconElement(iconSvg)]
	};
}

/**
 * Creates a copy button element.
 */
export function createCopyButton(id: string, idAttribute: string, title: string = 'Copy'): Element {
	return createButton(COPY_CODE_BTN_CLASS, title, COPY_ICON_SVG, id, idAttribute);
}

/**
 * Creates a preview button element.
 */
export function createPreviewButton(
	id: string,
	idAttribute: string,
	title: string = 'Preview'
): Element {
	return createButton(PREVIEW_CODE_BTN_CLASS, title, PREVIEW_ICON_SVG, id, idAttribute);
}

/**
 * Creates a button that toggles a diagram block between its rendered view and
 * its source view. aria-pressed starts false, the rendered view is the default.
 */
export function createToggleSourceButton(
	id: string,
	idAttribute: string,
	title: string = 'Toggle source'
): Element {
	return createButton(TOGGLE_SOURCE_BTN_CLASS, title, CODE_ICON_SVG, id, idAttribute, {
		'aria-pressed': 'false'
	});
}

/**
 * Creates a source view for a diagram block. It reuses the code block scroll
 * container so it matches the app code blocks, and wraps the highlighted code
 * element captured at transform time. A missing code element falls back to a
 * plain code node built from the raw source.
 */
export function createSourceView(
	codeElement: Element | undefined,
	source: string,
	language: string
): Element {
	const code: Element = codeElement ?? {
		type: 'element',
		tagName: 'code',
		properties: { className: ['hljs', `language-${language}`] },
		children: [{ type: 'text', value: source }]
	};
	return {
		type: 'element',
		tagName: 'div',
		properties: { className: [DIAGRAM_SOURCE_CLASS, CODE_BLOCK_SCROLL_CONTAINER_CLASS] },
		children: [
			{
				type: 'element',
				tagName: 'pre',
				properties: {},
				children: [code]
			}
		]
	};
}

/**
 * Creates a block header with language label and action buttons.
 */
export function createBlockHeader(
	language: string,
	id: string,
	idAttribute: string,
	actions: Element[],
	languageClassName: string = CODE_LANGUAGE_CLASS
): Element {
	return {
		type: 'element',
		tagName: 'div',
		properties: { className: [CODE_BLOCK_HEADER_CLASS] },
		children: [
			{
				type: 'element',
				tagName: 'span',
				properties: { className: [languageClassName] },
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

/**
 * Creates a scroll container element.
 */
export function createScrollContainer(preElement: Element, scrollContainerClass: string): Element {
	return {
		type: 'element',
		tagName: 'div',
		properties: { className: [scrollContainerClass] },
		children: [preElement]
	};
}

/**
 * Creates a wrapper element with header and scroll container. Extra children
 * append after the scroll container, which lets a block carry a source view
 * alongside its rendered output.
 */
export function createWrapper(
	header: Element,
	preElement: Element,
	wrapperClass: string,
	scrollContainerClass: string,
	additionalAttributes?: Record<string, string>,
	extraChildren: Element[] = []
): Element {
	return {
		type: 'element',
		tagName: 'div',
		properties: {
			className: [wrapperClass, RELATIVE_CLASS],
			...additionalAttributes
		} as Element['properties'],
		children: [header, createScrollContainer(preElement, scrollContainerClass), ...extraChildren]
	};
}

/**
 * Generates a unique block ID using a global counter.
 */
export function generateBlockId(prefix: string, windowKey: keyof Window): string {
	if (typeof window !== 'undefined') {
		const idx = window[windowKey] as number | undefined;
		const next = (idx ?? 0) + 1;
		(window as unknown as Record<string, number>)[windowKey] = next;
		return `${prefix}-${next}`;
	}
	// Fallback for SSR - use timestamp + random
	return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
}
