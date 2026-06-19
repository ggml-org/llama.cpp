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
}

/**
 * Replace every `currentColor` occurrence in the SVG with the given color.
 * Pure: no filesystem access, so it is straightforward to unit-test.
 */
export function colorizeFaviconSvg(svg: string, lightColor: string, darkColor: string): ColorizedFavicon {
	return {
		light: svg.replaceAll(CURRENT_COLOR, lightColor),
		dark: svg.replaceAll(CURRENT_COLOR, darkColor)
	};
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
		darkOutPath = DEFAULT_OUT_DARK
	}: WriteThemeFaviconsOptions = {}
): void {
	const source = readFileSync(sourcePath, 'utf-8');
	const { light, dark } = colorizeFaviconSvg(source, lightColor, darkColor);
	mkdirSync(dirname(lightOutPath), { recursive: true });
	writeFileSync(lightOutPath, light);
	writeFileSync(darkOutPath, dark);
}
