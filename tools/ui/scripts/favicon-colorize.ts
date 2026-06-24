import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(HERE, '..');

const DEFAULT_LOGO = resolve(PROJECT_ROOT, 'src/lib/assets/logo.svg');
const DEFAULT_OUT_DIR = resolve(PROJECT_ROOT, 'static');
const DEFAULT_OUT_LIGHT = resolve(DEFAULT_OUT_DIR, 'favicon.svg');
const DEFAULT_OUT_DARK = resolve(DEFAULT_OUT_DIR, 'favicon-dark.svg');

const CURRENT_COLOR = 'currentColor';

export interface ColorizedFavicon {
	light: string;
	dark: string;
}

export interface WriteThemeFaviconsOptions {
	sourcePath?: string;
	lightOutPath?: string;
	darkOutPath?: string;
	/**
	 * Fraction of the icon (0..1) to leave as an even margin on each side.
	 * Applied by wrapping the inner content in a `<g transform="...">` so the
	 * source `src/lib/assets/logo.svg` is not modified. Pass 0 to disable.
	 */
	padding?: number;
}

/**
 * Replace every `currentColor` occurrence in the SVG with the given color.
 * Pure: no filesystem access, so it is straightforward to unit-test.
 */
export function colorizeFaviconSvg(
	svg: string,
	lightColor: string,
	darkColor: string
): ColorizedFavicon {
	return {
		light: svg.replaceAll(CURRENT_COLOR, lightColor),
		dark: svg.replaceAll(CURRENT_COLOR, darkColor)
	};
}

/**
 * Shrink the inner SVG content uniformly and re-center it so `padding` (a
 * 0..1 fraction) is reserved as equal margin on each side. Returns the input
 * unchanged for non-positive padding, missing/invalid `viewBox`, or unexpected
 * markup so the caller always gets a renderable SVG.
 */
export function padFaviconSvg(svg: string, padding: number): string {
	if (!(padding > 0) || padding >= 1) return svg;

	const viewBoxMatch = svg.match(/viewBox\s*=\s*["']([^"']+)["']/i);
	if (!viewBoxMatch) return svg;

	const parts = viewBoxMatch[1]
		.trim()
		.split(/[\s,]+/)
		.map(Number);
	if (parts.length !== 4 || parts.some((n) => !Number.isFinite(n))) return svg;

	const [, , width, height] = parts;
	if (width <= 0 || height <= 0) return svg;

	const scale = 1 - padding;
	const translateX = (padding * width) / 2;
	const translateY = (padding * height) / 2;

	const openTagStart = svg.search(/<svg\b/i);
	if (openTagStart === -1) return svg;
	const openTagEnd = svg.indexOf('>', openTagStart);
	if (openTagEnd === -1) return svg;
	const closeStart = svg.lastIndexOf('</svg');
	if (closeStart === -1 || closeStart <= openTagEnd) return svg;

	const openTag = svg.slice(0, openTagEnd + 1);
	const inner = svg.slice(openTagEnd + 1, closeStart);
	const closeTag = svg.slice(closeStart);

	const group = `<g transform="translate(${translateX} ${translateY}) scale(${scale})">`;
	return `${openTag}${group}${inner}</g>${closeTag}`;
}

/**
 * Read `src/lib/assets/logo.svg`, colorize it for both themes, and write
 * the results to the static directory so the PWA asset generator can consume
 * them. Paths can be overridden for tests.
 */
export function writeThemeFavicons(
	lightColor: string,
	darkColor: string,
	{
		sourcePath = DEFAULT_LOGO,
		lightOutPath = DEFAULT_OUT_LIGHT,
		darkOutPath = DEFAULT_OUT_DARK,
		padding = 0
	}: WriteThemeFaviconsOptions = {}
): void {
	const source = readFileSync(sourcePath, 'utf-8');
	const { light, dark } = colorizeFaviconSvg(source, lightColor, darkColor);
	mkdirSync(dirname(lightOutPath), { recursive: true });
	writeFileSync(lightOutPath, padFaviconSvg(light, padding));
	writeFileSync(darkOutPath, padFaviconSvg(dark, padding));
}
