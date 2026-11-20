// Diagram blocks (mermaid, svg) in rendered markdown: drawing them after the HTML is in
// the page, and the actions of their header buttons. Shared by MarkdownContent and the
// live preview of MarkdownEditor, so a diagram looks and behaves the same in both.

import {
	DIAGRAM_VIEW_MODE_ATTR,
	DIAGRAM_VIEW_RENDERED,
	DIAGRAM_VIEW_SOURCE,
	MERMAID_BLOCK_CLASS,
	MERMAID_ERROR_CLASS,
	MERMAID_ID_ATTR,
	MERMAID_RENDERED_ATTR,
	MERMAID_SCROLL_CONTAINER_CLASS,
	MERMAID_SVG_PADDING_X_REM,
	MERMAID_SYNTAX_ATTR,
	MERMAID_WRAPPER_CLASS,
	PELE_SVG_CLASS,
	PELE_THEME_VARIABLES,
	SVG,
	TOGGLE_SOURCE_BTN_CLASS
} from '#lib/constants/index.ts';
import { BooleanString } from '#lib/enums/index.ts';
import { sanitizeSvg } from '#lib/utils/sanitize-svg.ts';
import { mountSvgShadow } from '#lib/utils/svg-shadow.ts';

export type DiagramAction = { copy: string } | { previewSvg: string };

// The width a diagram can take in its block: the scroll container's content box, less the
// svg's own horizontal padding. Undefined when the block isn't laid out (detached, hidden).
function diagramWidth(node: HTMLElement): number | undefined {
	const scroll = node.closest<HTMLElement>(`.${MERMAID_SCROLL_CONTAINER_CLASS}`);

	if (!scroll?.clientWidth) return undefined;

	const box = getComputedStyle(scroll);
	const rem = parseFloat(getComputedStyle(document.documentElement).fontSize);
	const width =
		scroll.clientWidth -
		parseFloat(box.paddingLeft) -
		parseFloat(box.paddingRight) -
		MERMAID_SVG_PADDING_X_REM * rem;

	return width > 0 ? width : undefined;
}

// Pele wraps a node with a `click ... href` directive in a link. Diagrams come from model
// output, so they stay inert, as they were under Mermaid's strict security level: each link
// gives way to its children. The preview dialog and the downloaded file copy this svg.
function unwrapLinks(root: HTMLElement) {
	for (const link of root.querySelectorAll('a')) link.replaceWith(...link.childNodes);
}

/**
 * Renders the mermaid diagrams in `container` that haven't been rendered yet, with Pele.
 * Marks nodes immediately to prevent duplicate renders during streaming. Colors come
 * from the `--pele-*` variables in app.css, so a diagram follows the theme without a redraw.
 */
export async function renderMermaidDiagrams(container: HTMLElement) {
	const nodes = container.querySelectorAll<HTMLElement>(
		`pre.${MERMAID_BLOCK_CLASS}:not([${MERMAID_RENDERED_ATTR}])`
	);

	if (nodes.length === 0) return;

	// Mark nodes immediately to prevent duplicate renders if called again during streaming.
	// This avoids needing a guard that would block node discovery.
	nodes.forEach((node) => node.setAttribute(MERMAID_RENDERED_ATTR, BooleanString.TRUE));

	const { render } = await import('pele');

	// Pele measures labels with the font it is given, so wait until the page font is in
	await document.fonts?.ready;

	for (const node of nodes) {
		const source = node.getAttribute(MERMAID_SYNTAX_ATTR) ?? node.textContent ?? '';
		const id = node.getAttribute(MERMAID_ID_ATTR);

		try {
			const { svg } = render(source, {
				fontFamily: getComputedStyle(node).fontFamily,
				idPrefix: id ? id.replace(/[^\w-]/g, '_') : undefined,
				// narrow blocks (phones) may turn a left-to-right diagram top-to-bottom
				maxWidth: diagramWidth(node)
			});

			node.innerHTML = svg;
			unwrapLinks(node);
		} catch (error) {
			// the message quotes the diagram source, so it goes in as text
			const message = document.createElement('div');

			message.className = MERMAID_ERROR_CLASS;
			message.textContent = error instanceof Error ? error.message : String(error);
			node.replaceChildren(message);
		}
	}
}

/**
 * A Pele diagram takes its colors from the page's `--pele-*` variables. Writes their current
 * values into the svg, so a downloaded file keeps the colors it had on screen. Any other svg
 * comes back unchanged.
 */
export function withThemeColors(svgHtml: string): string {
	const svg = new DOMParser().parseFromString(svgHtml, 'image/svg+xml').documentElement;

	if (!(svg instanceof SVGElement) || !svg.classList.contains(PELE_SVG_CLASS)) return svgHtml;

	const theme = getComputedStyle(document.documentElement);

	for (const name of PELE_THEME_VARIABLES) {
		const value = theme.getPropertyValue(name).trim();

		if (value) svg.style.setProperty(name, value);
	}

	return new XMLSerializer().serializeToString(svg);
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
