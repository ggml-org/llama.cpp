/**
 * Converts a rem CSS value to pixels based on the document root font size.
 */
export function remToPx(rem: string): number {
	const val = parseFloat(rem);
	const fontSize = parseFloat(getComputedStyle(document.documentElement).fontSize);

	return val * fontSize;
}
