// Diagram blocks (mermaid, svg) in rendered markdown: drawing them after the HTML is in
// the page, and the actions of their header buttons. Shared by MarkdownContent and the
// live preview of MarkdownEditor, so a diagram looks and behaves the same in both.

import {
	DIAGRAM_VIEW_MODE_ATTR,
	DIAGRAM_VIEW_RENDERED,
	DIAGRAM_VIEW_SOURCE,
	MERMAID_BLOCK_CLASS,
	MERMAID_RENDERED_ATTR,
	MERMAID_SYNTAX_ATTR,
	MERMAID_WRAPPER_CLASS,
	SVG,
	TOGGLE_SOURCE_BTN_CLASS
} from '$lib/constants';
import { BooleanString } from '$lib/enums';
import { sanitizeSvg } from '$lib/utils/sanitize-svg';
import { mountSvgShadow } from '$lib/utils/svg-shadow';

export type DiagramAction = { copy: string } | { previewSvg: string };

/**
 * Renders the mermaid diagrams in `container` that haven't been rendered yet.
 * Marks nodes immediately to prevent duplicate renders during streaming.
 */
export async function renderMermaidDiagrams(container: HTMLElement, isDark: boolean) {
	const nodes = container.querySelectorAll<HTMLElement>(
		`pre.${MERMAID_BLOCK_CLASS}:not([${MERMAID_RENDERED_ATTR}])`
	);

	if (nodes.length === 0) return;

	// Mark nodes immediately to prevent duplicate renders if called again during streaming.
	// This avoids needing a guard that would block node discovery.
	nodes.forEach((node) => node.setAttribute(MERMAID_RENDERED_ATTR, BooleanString.TRUE));

	// lazy load the mermaid dependecy only when needed to reduce bundle size.
	const { default: mermaid } = await import('mermaid');

	mermaid.initialize({
		flowchart: {
			htmlLabels: true,
			useMaxWidth: false
		},
		gantt: {
			useMaxWidth: false
		},
		securityLevel: 'strict',
		sequence: {
			useMaxWidth: false
		},
		startOnLoad: false,
		theme: isDark ? 'dark' : 'default'
	});

	try {
		await mermaid.run({ nodes: Array.from(nodes) as unknown as NodeListOf<HTMLElement> });
	} catch (error) {
		console.error('Failed to render mermaid diagram:', error);
	}
}

/**
 * Renders the svg diagrams in `container` that haven't been rendered yet.
 * Sanitizes the source before injecting and marks each node so it renders once.
 * An empty sanitize result keeps the raw source as escaped text.
 */
export function renderSvgDiagrams(container: HTMLElement) {
	const nodes = container.querySelectorAll<HTMLElement>(
		`pre.${SVG.BLOCK_CLASS}:not([${SVG.RENDERED_ATTR}])`
	);

	nodes.forEach((node) => {
		node.setAttribute(SVG.RENDERED_ATTR, BooleanString.TRUE);

		const source = node.getAttribute(SVG.SOURCE_ATTR) ?? node.textContent ?? '';
		const clean = sanitizeSvg(source);

		if (clean) {
			node.textContent = '';
			const host = document.createElement('div');

			node.appendChild(host);
			mountSvgShadow(host, clean, SVG.INLINE_SHADOW_STYLE);
		}
	});
}

/**
 * Toggles a diagram block between its rendered view and its source view when `target`
 * is its toggle button; css drives the visibility from the wrapper mode.
 * Returns whether the target was a toggle button.
 */
export function toggleDiagramSource(target: HTMLElement): boolean {
	const toggleBtn = target.closest(`.${TOGGLE_SOURCE_BTN_CLASS}`);

	if (!toggleBtn) return false;

	const wrapper = toggleBtn.closest(`.${MERMAID_WRAPPER_CLASS}, .${SVG.WRAPPER_CLASS}`);

	if (!wrapper) return true;

	const isSource = wrapper.getAttribute(DIAGRAM_VIEW_MODE_ATTR) === DIAGRAM_VIEW_SOURCE;

	wrapper.setAttribute(
		DIAGRAM_VIEW_MODE_ATTR,
		isSource ? DIAGRAM_VIEW_RENDERED : DIAGRAM_VIEW_SOURCE
	);
	toggleBtn.setAttribute('aria-pressed', String(!isSource));

	return true;
}

/**
 * What a click on a diagram block's copy or preview button asks for: the text to copy
 * (mermaid syntax, svg source) or the svg to preview. Null when `target` is neither.
 */
export function getDiagramButtonAction(target: HTMLElement): DiagramAction | null {
	const isCopy = Boolean(target.closest('.copy-code-btn'));

	if (!isCopy && !target.closest('.preview-code-btn')) return null;

	const mermaidPre = target
		.closest(`.${MERMAID_WRAPPER_CLASS}`)
		?.querySelector<HTMLElement>(`pre.${MERMAID_BLOCK_CLASS}[${MERMAID_SYNTAX_ATTR}]`);

	if (mermaidPre) {
		if (isCopy) return { copy: mermaidPre.getAttribute(MERMAID_SYNTAX_ATTR) ?? '' };

		const svg = mermaidPre.querySelector('svg');

		return svg ? { previewSvg: svg.outerHTML } : null;
	}

	const svgPre = target
		.closest(`.${SVG.WRAPPER_CLASS}`)
		?.querySelector<HTMLElement>(`pre.${SVG.BLOCK_CLASS}[${SVG.SOURCE_ATTR}]`);

	if (!svgPre) return null;

	const source = svgPre.getAttribute(SVG.SOURCE_ATTR) ?? '';

	return isCopy ? { copy: source } : { previewSvg: sanitizeSvg(source) };
}
