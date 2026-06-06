/**
 * Shared utilities for enhanced code blocks and mermaid diagram blocks.
 * Contains common HAST element creation functions to avoid code duplication.
 */

import type { Element, ElementContent } from 'hast';
import {
	CODE_BLOCK_HEADER_CLASS,
	CODE_BLOCK_ACTIONS_CLASS,
	CODE_LANGUAGE_CLASS,
	COPY_CODE_BTN_CLASS,
	PREVIEW_CODE_BTN_CLASS,
	RELATIVE_CLASS,
	COPY_ICON_SVG,
	PREVIEW_ICON_SVG
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
 * Creates a button element with icon.
 */
export function createButton(
	className: string,
	title: string,
	iconSvg: string,
	id: string,
	idAttribute: string
): Element {
	return {
		type: 'element',
		tagName: 'button',
		properties: {
			className: [className],
			[idAttribute]: id,
			title,
			type: 'button'
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
 * Creates a wrapper element with header and scroll container.
 */
export function createWrapper(
	header: Element,
	preElement: Element,
	wrapperClass: string,
	scrollContainerClass: string,
	additionalAttributes?: Record<string, string>
): Element {
	return {
		type: 'element',
		tagName: 'div',
		properties: {
			className: [wrapperClass, RELATIVE_CLASS],
			...additionalAttributes
		} as Element['properties'],
		children: [header, createScrollContainer(preElement, scrollContainerClass)]
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
