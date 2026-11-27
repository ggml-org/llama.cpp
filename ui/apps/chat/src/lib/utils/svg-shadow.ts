const SVG_VIEWBOX_ATTR = 'viewBox';
const SVG_CSS_DPI = 96;
const SVG_DIMENSION_PATTERN = /^\s*(\d*\.?\d+)\s*(px|in|cm|mm|pt|pc)?\s*$/;
// user units (css px) per absolute length unit, at the css dpi
const SVG_ABSOLUTE_UNIT_PX: Record<string, number> = {
	'': 1,
	cm: SVG_CSS_DPI / 2.54,
	in: SVG_CSS_DPI,
	mm: SVG_CSS_DPI / 25.4,
	pc: SVG_CSS_DPI / 6,
	pt: SVG_CSS_DPI / 72,
	px: 1
};

/**
 * Reads an absolute svg width/height attribute (unitless, px, in, cm, mm, pt or pc)
 * as user units. Relative lengths such as % or em return null.
 */
export function readSvgDimension(value: string | null): number | null {
	const match = value?.match(SVG_DIMENSION_PATTERN);

	if (!match) return null;

	const size = Number(match[1]) * SVG_ABSOLUTE_UNIT_PX[match[2] ?? ''];

	return size > 0 ? size : null;
}

/**
 * Gives an svg with a fixed width and height but no viewBox a matching viewBox,
 * so css max-width scales the drawing instead of clipping it.
 */
function ensureSvgViewBox(svg: SVGSVGElement): void {
	if (svg.hasAttribute(SVG_VIEWBOX_ATTR)) return;

	const width = readSvgDimension(svg.getAttribute('width'));
	const height = readSvgDimension(svg.getAttribute('height'));

	if (width && height) svg.setAttribute(SVG_VIEWBOX_ATTR, `0 0 ${width} ${height}`);
}

/**
 * Mounts svg markup inside an open shadow root on the host element.
 * The shadow boundary scopes the svg <style> and its animations to the host,
 * so model authored css can not reach the surrounding page. The caller passes
 * markup that is already sanitized, this only isolates and sizes it.
 */
export function mountSvgShadow(host: HTMLElement, markup: string, style: string): void {
	const root = host.shadowRoot ?? host.attachShadow({ mode: 'open' });

	root.innerHTML = markup ? `<style>${style}</style>${markup}` : '';

	const svg = root.querySelector('svg');

	if (svg) ensureSvgViewBox(svg);
}
