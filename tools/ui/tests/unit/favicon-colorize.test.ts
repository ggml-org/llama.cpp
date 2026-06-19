import { describe, expect, it } from 'vitest';
import { colorizeFaviconSvg } from '../../scripts/favicon-colorize';

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
