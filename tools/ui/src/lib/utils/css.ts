import { SurfaceTheme } from '$lib/enums';

/**
 * Applies the light surface theme: Matte sets the theme-matte class on <html>, which the tokens
 * read (see tokens/semantic.css); Paper, the default, needs no class.
 */
export function applySurfaceTheme(theme: unknown): void {
	if (typeof document === 'undefined') return;

	document.documentElement.classList.toggle('theme-matte', theme === SurfaceTheme.MATTE);
}

/**
 * Converts a rem CSS value to pixels based on the document root font size.
 */
export function remToPx(rem: string): number {
	const val = parseFloat(rem);
	const fontSize = parseFloat(getComputedStyle(document.documentElement).fontSize);

	return val * fontSize;
}
