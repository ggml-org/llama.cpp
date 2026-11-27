import { PDF_RENDERING } from '#lib/constants/index.ts';

/**
 * A PDF.js scale is the ratio of rendered pixels to PDF points: at 1.0 a page renders at
 * 72 DPI, so scale is DPI / 72.
 * @param dpi - the DPI to convert
 * @returns the PDF.js scale factor for the given DPI
 */
export function pdfDpiToScale(dpi: number): number {
	return dpi / PDF_RENDERING.POINTS_PER_INCH;
}

/**
 * Clamp a user-supplied DPI into the supported range. Anything that is not a finite number
 * (missing key, empty field, junk) falls back to the default the setting ships with.
 * @param value - the raw value from the settings config
 * @returns a DPI between MIN_DPI and MAX_DPI
 */
export function clampPdfDpi(value: unknown): number {
	const dpi = typeof value === 'string' ? (value.trim() === '' ? undefined : Number(value)) : value;

	if (typeof dpi !== 'number' || !Number.isFinite(dpi)) return PDF_RENDERING.DEFAULT_DPI;

	return Math.min(Math.max(Math.round(dpi), PDF_RENDERING.MIN_DPI), PDF_RENDERING.MAX_DPI);
}
