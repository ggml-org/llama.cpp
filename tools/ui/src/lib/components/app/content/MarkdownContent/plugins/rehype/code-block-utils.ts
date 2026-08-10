/**
 * Shared utilities for enhanced code blocks and mermaid diagram blocks.
 * Contains common HAST element creation functions to avoid code duplication.
 */

import {
	CODE_BLOCK_ACTIONS_CLASS,
	CODE_BLOCK_HEADER_CLASS,
	CODE_BLOCK_SCROLL_CONTAINER_CLASS,
	CODE_ICON_SVG,
	CODE_LANGUAGE_CLASS,
	COPY_CODE_BTN_CLASS,
	COPY_ICON_SVG,
	DIAGRAM_SOURCE_CLASS,
	PREVIEW_CODE_BTN_CLASS,
	PREVIEW_ICON_SVG,
	RELATIVE_CLASS,
	TOGGLE_SOURCE_BTN_CLASS
} from '$lib/constants';
import type { Element, ElementContent } from 'hast';

export interface BlockIdGenerator {
	(id: number): string;
}

/**
 * Creates an icon element with the given SVG content.
 */
export function createIconElement(svg: string): Element {
	return {
		children: [{ type: 'raw', value: svg } as unknown as ElementContent],
		properties: {},
		tagName: 'span',
		type: 'element'
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
		children: [createIconElement(iconSvg)],
		properties: {
			className: [className],
			[idAttribute]: id,
			title,
			type: 'button',
			...extraProperties
		},
		tagName: 'button',
		type: 'element'
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
		children: [{ type: 'text', value: source }],
		properties: { className: ['hljs', `language-${language}`] },
		tagName: 'code',
		type: 'element'
	};

	return {
		children: [
			{
				children: [code],
				properties: {},
				tagName: 'pre',
				type: 'element'
			}
		],
		properties: { className: [DIAGRAM_SOURCE_CLASS, CODE_BLOCK_SCROLL_CONTAINER_CLASS] },
		tagName: 'div',
		type: 'element'
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
		children: [
			{
				children: [{ type: 'text', value: language }],
				properties: { className: [languageClassName] },
				tagName: 'span',
				type: 'element'
			},
			{
				children: actions,
				properties: { className: [CODE_BLOCK_ACTIONS_CLASS] },
				tagName: 'div',
				type: 'element'
			}
		],
		properties: { className: [CODE_BLOCK_HEADER_CLASS] },
		tagName: 'div',
		type: 'element'
	};
}

/**
 * Creates a scroll container element.
 */
export function createScrollContainer(preElement: Element, scrollContainerClass: string): Element {
	return {
		children: [preElement],
		properties: { className: [scrollContainerClass] },
		tagName: 'div',
		type: 'element'
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
		children: [header, createScrollContainer(preElement, scrollContainerClass), ...extraChildren],
		properties: {
			className: [wrapperClass, RELATIVE_CLASS],
			...additionalAttributes
		} as Element['properties'],
		tagName: 'div',
		type: 'element'
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
