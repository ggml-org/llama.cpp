import { describe, expect, it } from 'vitest';
import { decodeFileLinkPath, encodeFileLinkPath } from '$lib/utils';

describe('encodeFileLinkPath', () => {
	it('leaves a clean path unchanged', () => {
		expect(encodeFileLinkPath('/Users/foo/bar.txt')).toBe('/Users/foo/bar.txt');
	});

	it('encodes spaces per path segment', () => {
		expect(
			encodeFileLinkPath('/Users/allozaur/Desktop/Screenshot 2026-08-05 at 11.33.45.png')
		).toBe('/Users/allozaur/Desktop/Screenshot%202026-08-05%20at%2011.33.45.png');
	});

	it('preserves the leading and trailing slash (directory marker)', () => {
		expect(encodeFileLinkPath('/Users/foo/bar/')).toBe('/Users/foo/bar/');
	});

	it('encodes parentheses in macOS screenshot names', () => {
		expect(encodeFileLinkPath('/Users/foo/Pic (1).png')).toBe('/Users/foo/Pic%20(1).png');
	});
});

describe('decodeFileLinkPath', () => {
	it('decodes encoded segments back to the original path', () => {
		expect(
			decodeFileLinkPath('/Users/allozaur/Desktop/Screenshot%202026-08-05%20at%2011.33.45.png')
		).toBe('/Users/allozaur/Desktop/Screenshot 2026-08-05 at 11.33.45.png');
	});

	it('is the inverse of encodeFileLinkPath', () => {
		for (const path of [
			'/a/b.txt',
			'/Users/foo/Desktop/Screenshot 2026-08-05 at 11.33.45.png',
			'/Users/foo/bar (1)/dir/',
			'/sp ace/pa%th.txt'
		]) {
			expect(decodeFileLinkPath(encodeFileLinkPath(path))).toBe(path);
		}
	});

	it('falls back to the input on malformed percent sequences', () => {
		expect(decodeFileLinkPath('/a/%zz.txt')).toBe('/a/%zz.txt');
	});
});
