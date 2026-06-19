import { describe, expect, it } from 'vitest';
import { colorizeFaviconSvg, padFaviconSvg } from '../../scripts/favicon-colorize';

const SOURCE_SVG = [
	'<svg xmlns="http://www.w3.org/2000/svg">',
	'  <path d="M0 0" fill="currentColor"/>',
	'  <path d="M1 1" fill="#ff00aa"/>',
	'  <circle fill="currentColor"/>',
	'</svg>'
].join('\n');

describe('colorizeFaviconSvg', () => {
	it('substitutes every currentColor occurrence for the light variant', () => {
		const { light } = colorizeFaviconSvg(SOURCE_SVG, '#111111', '#fafafa');
		expect(light.match(/currentColor/g)).toBeNull();
		expect(light).toContain('fill="#111111"');
		expect(light).toContain('<circle fill="#111111"/>');
	});

	it('substitutes every currentColor occurrence for the dark variant', () => {
		const { dark } = colorizeFaviconSvg(SOURCE_SVG, '#111111', '#fafafa');
		expect(dark.match(/currentColor/g)).toBeNull();
		expect(dark).toContain('fill="#fafafa"');
		expect(dark).toContain('<circle fill="#fafafa"/>');
	});

	it('leaves non-currentColor colors untouched in both variants', () => {
		const { light, dark } = colorizeFaviconSvg(SOURCE_SVG, '#111111', '#fafafa');
		expect(light).toContain('fill="#ff00aa"');
		expect(dark).toContain('fill="#ff00aa"');
	});

	it('does not alter any other part of the SVG', () => {
		const { light, dark } = colorizeFaviconSvg(SOURCE_SVG, '#111111', '#fafafa');
		const stripColors = (s: string) => s.replaceAll('#111111', '').replaceAll('#fafafa', '');
		const expected = stripColors(SOURCE_SVG);
		expect(stripColors(light)).toBe(expected);
		expect(stripColors(dark)).toBe(expected);
	});

	it('returns the same SVG for light and dark when called with the same color', () => {
		const result = colorizeFaviconSvg(SOURCE_SVG, '#abcdef', '#abcdef');
		expect(result.light).toBe(result.dark);
	});

	it('returns the source unchanged when given a color that does not appear (no currentColor in source)', () => {
		const plain = '<svg><path fill="#000"/></svg>';
		const { light, dark } = colorizeFaviconSvg(plain, '#111111', '#fafafa');
		expect(light).toBe(plain);
		expect(dark).toBe(plain);
	});
});

describe('padFaviconSvg', () => {
	const SIZED_SVG =
		'<svg width="512" height="512" viewBox="0 0 512 512" fill="none" xmlns="http://www.w3.org/2000/svg">' +
		'<path d="M244.95 8L388.923 8Z" fill="currentColor"/>' +
		'</svg>';

	it('wraps inner content in a translate-then-scale group that matches padding', () => {
		const padded = padFaviconSvg(SIZED_SVG, 0.05);
		// scale = 1 - 0.05 = 0.95
		// translate = (0.05 * 512) / 2 = 12.8 on each axis
		expect(padded).toContain('transform="translate(12.8 12.8) scale(0.95)"');
		expect(padded).toContain('<g transform="translate(12.8 12.8) scale(0.95)">');
		expect(padded).toContain('<path d="M244.95 8L388.923 8Z" fill="currentColor"/>');
		expect(padded.endsWith('</g></svg>')).toBe(true);
	});

	it('preserves the outer <svg> tag attributes', () => {
		const padded = padFaviconSvg(SIZED_SVG, 0.1);
		expect(padded.startsWith('<svg width="512" height="512" viewBox="0 0 512 512"')).toBe(true);
	});

	it('returns the input unchanged for zero or negative padding', () => {
		expect(padFaviconSvg(SIZED_SVG, 0)).toBe(SIZED_SVG);
		expect(padFaviconSvg(SIZED_SVG, -0.1)).toBe(SIZED_SVG);
	});

	it('returns the input unchanged when padding would fully collapse the icon (>= 1)', () => {
		expect(padFaviconSvg(SIZED_SVG, 1)).toBe(SIZED_SVG);
		expect(padFaviconSvg(SIZED_SVG, 1.5)).toBe(SIZED_SVG);
	});

	it('returns the input unchanged when no viewBox is present', () => {
		const noViewBox = '<svg width="32" height="32"><path d="M0 0Z"/></svg>';
		expect(padFaviconSvg(noViewBox, 0.1)).toBe(noViewBox);
	});

	it('returns the input unchanged when viewBox values are not finite numbers', () => {
		const bad = '<svg viewBox="auto auto 0 0"><path/></svg>';
		expect(padFaviconSvg(bad, 0.1)).toBe(bad);
	});

	it('tolerates a non-square viewBox', () => {
		const wide = '<svg viewBox="0 0 100 50"><rect/></svg>';
		const padded = padFaviconSvg(wide, 0.1);
		// scale 0.9, translate (5, 2.5)
		expect(padded).toContain('transform="translate(5 2.5) scale(0.9)"');
	});
});
