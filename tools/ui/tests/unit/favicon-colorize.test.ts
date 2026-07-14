import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import {
	colorizeFaviconSvg,
	padFaviconSvg,
	writeThemeFavicons
} from '../../scripts/favicon-colorize';

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
		const stripColors = (s: string) =>
			s.replaceAll('#111111', '').replaceAll('#fafafa', '').replaceAll('currentColor', '');
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

describe('writeThemeFavicons', () => {
	const LOGO =
		'<svg width="512" height="512" viewBox="0 0 512 512" fill="none" xmlns="http://www.w3.org/2000/svg">' +
		'<path d="M244.95 8L388.923 8Z" fill="currentColor"/>' +
		'</svg>';

	let tmpDir: string;

	beforeEach(() => {
		tmpDir = mkdtempSync(join(tmpdir(), 'favicon-'));
	});

	afterEach(() => {
		rmSync(tmpDir, { recursive: true, force: true });
	});

	function setupSource() {
		const sourcePath = join(tmpDir, 'logo.svg');
		writeFileSync(sourcePath, LOGO);
		return {
			sourcePath,
			lightPath: join(tmpDir, 'favicon.svg'),
			darkPath: join(tmpDir, 'favicon-dark.svg')
		};
	}

	it('writes colorized, un-padded favicons without modifying the source', () => {
		const { sourcePath, lightPath, darkPath } = setupSource();
		const before = readFileSync(sourcePath, 'utf-8');

		writeThemeFavicons('#abcdef', '#012345', {
			sourcePath,
			lightOutPath: lightPath,
			darkOutPath: darkPath
		});

		const lightOut = readFileSync(lightPath, 'utf-8');
		const darkOut = readFileSync(darkPath, 'utf-8');

		// currentColor swapped to the requested palette in both files
		expect(lightOut).toContain('fill="#abcdef"');
		expect(lightOut).not.toContain('currentColor');
		expect(darkOut).toContain('fill="#012345"');
		expect(darkOut).not.toContain('currentColor');

		// default padding (0) keeps the wrapper off the output
		expect(lightOut).not.toContain('<g ');
		expect(darkOut).not.toContain('<g ');

		// source file is unchanged after the call
		expect(readFileSync(sourcePath, 'utf-8')).toBe(before);
	});

	it('writes colorized favicons wrapped in a padding <g transform>...</g>', () => {
		const { sourcePath, lightPath, darkPath } = setupSource();
		// mirror the production wiring: PWA_ASSET_GENERATOR.FAVICON_PADDING
		const padding = 0.04;
		// scale = 1 - 0.04 = 0.96; translate = (0.04 * 512) / 2 = 10.24
		const expectedTransform = 'transform="translate(10.24 10.24) scale(0.96)"';

		writeThemeFavicons('#111111', '#fafafa', {
			sourcePath,
			lightOutPath: lightPath,
			darkOutPath: darkPath,
			padding
		});

		const lightOut = readFileSync(lightPath, 'utf-8');
		const darkOut = readFileSync(darkPath, 'utf-8');

		// both files get the same padding wrapper derived from the viewBox
		expect(lightOut).toContain(`<g ${expectedTransform}>`);
		expect(lightOut.endsWith('</g></svg>')).toBe(true);
		expect(darkOut).toContain(`<g ${expectedTransform}>`);
		expect(darkOut.endsWith('</g></svg>')).toBe(true);

		// colorization still happens inside the wrapped group
		expect(lightOut).toContain('fill="#111111"');
		expect(lightOut).not.toContain('currentColor');
		expect(darkOut).toContain('fill="#fafafa"');
		expect(darkOut).not.toContain('currentColor');
		// light palette colour must not leak into dark output
		expect(darkOut).not.toContain('#111111');

		// source file is not modified by the padding step
		expect(readFileSync(sourcePath, 'utf-8')).toBe(LOGO);
	});
});
