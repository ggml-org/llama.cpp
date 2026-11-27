import { PDF_RENDERING, SETTING_CONFIG_DEFAULT, SETTINGS_KEYS } from '#lib/constants/index.ts';
import { clampPdfDpi, pdfDpiToScale } from '#lib/utils/pdf-dpi.ts';
import { describe, expect, it } from 'vitest';

describe('pdf-dpi', () => {
	it('converts DPI to a PDF.js scale factor of pixels per PDF point', () => {
		expect(pdfDpiToScale(72)).toBe(1);
		expect(pdfDpiToScale(108)).toBeCloseTo(1.5, 10);
		expect(pdfDpiToScale(216)).toBe(3);
	});

	it('rounds, clamps and falls back to the default', () => {
		expect(clampPdfDpi(108)).toBe(108);
		expect(clampPdfDpi(216.4)).toBe(216);
		expect(clampPdfDpi(1000)).toBe(PDF_RENDERING.MAX_DPI);
		expect(clampPdfDpi(10)).toBe(PDF_RENDERING.MIN_DPI);
		expect(clampPdfDpi('216')).toBe(216);
		expect(clampPdfDpi('junk')).toBe(PDF_RENDERING.DEFAULT_DPI);
		expect(clampPdfDpi('')).toBe(PDF_RENDERING.DEFAULT_DPI);
		expect(clampPdfDpi('   ')).toBe(PDF_RENDERING.DEFAULT_DPI);
		expect(clampPdfDpi(undefined)).toBe(PDF_RENDERING.DEFAULT_DPI);
		expect(clampPdfDpi(null)).toBe(PDF_RENDERING.DEFAULT_DPI);
		expect(clampPdfDpi(NaN)).toBe(PDF_RENDERING.DEFAULT_DPI);
		expect(clampPdfDpi(Infinity)).toBe(PDF_RENDERING.DEFAULT_DPI);
	});
});

describe('PDF DPI setting', () => {
	it('ships a default that is inside its own range', () => {
		expect(SETTING_CONFIG_DEFAULT[SETTINGS_KEYS.PDF_DPI]).toBe(PDF_RENDERING.DEFAULT_DPI);
		expect(SETTING_CONFIG_DEFAULT[SETTINGS_KEYS.PDF_DPI]).toBeGreaterThanOrEqual(
			PDF_RENDERING.MIN_DPI
		);
		expect(SETTING_CONFIG_DEFAULT[SETTINGS_KEYS.PDF_DPI]).toBeLessThanOrEqual(
			PDF_RENDERING.MAX_DPI
		);
	});
});
