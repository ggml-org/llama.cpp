import { describe, expect, it } from 'vitest';
import {
	splitPathQuery,
	buildCaseInsensitiveGlob,
	rankEntries,
	joinPath,
	highlightMatch
} from '$lib/utils';

describe('splitPathQuery', () => {
	it('treats a plain query as a home-relative glob (not navigation)', () => {
		expect(splitPathQuery('docs')).toBeNull();
	});

	it('navigates the root for `/`', () => {
		expect(splitPathQuery('/')).toEqual({ parent: '/', last: '' });
	});

	it('navigates home for `~`', () => {
		expect(splitPathQuery('~')).toEqual({ parent: '~', last: '' });
	});

	it('splits an absolute path into parent and last segment', () => {
		expect(splitPathQuery('/Users/al/proj')).toEqual({ parent: '/Users/al', last: 'proj' });
	});

	it('splits a home-relative path into parent and last segment', () => {
		expect(splitPathQuery('~/Documents')).toEqual({ parent: '~', last: 'Documents' });
	});

	it('strips trailing slashes before splitting', () => {
		expect(splitPathQuery('/Users/al/')).toEqual({ parent: '/Users', last: 'al' });
	});

	it('handles a single-segment absolute path', () => {
		expect(splitPathQuery('/opt')).toEqual({ parent: '/', last: 'opt' });
	});
});

describe('buildCaseInsensitiveGlob', () => {
	it('wraps letters in case-insensitive character classes', () => {
		expect(buildCaseInsensitiveGlob('ab')).toBe('*[aA][bB]*');
	});

	it('passes through glob metacharacters', () => {
		expect(buildCaseInsensitiveGlob('a*b')).toContain('*');
	});
});

describe('rankEntries', () => {
	const entries = [
		{ path: '/h/README', type: 'dir' },
		{ path: '/h/read', type: 'dir' },
		{ path: '/h/readme.txt', type: 'dir' }
	];

	it('ranks exact basename match first', () => {
		const ranked = rankEntries(entries, 'read');
		expect(ranked[0].path).toBe('/h/read');
	});

	it('breaks ties by shorter path, then alphabetically', () => {
		const ranked = rankEntries(entries, 'read');
		expect(ranked[ranked.length - 1].path).toBe('/h/readme.txt');
	});

	it('does not mutate the input', () => {
		const snapshot = [...entries];
		rankEntries(entries, 'read');
		expect(entries).toEqual(snapshot);
	});
});

describe('joinPath', () => {
	it('joins base and relative avoiding a double slash', () => {
		expect(joinPath('/home/al/', 'docs')).toBe('/home/al/docs');
	});

	it('returns the relative path when base is empty', () => {
		expect(joinPath('', 'docs')).toBe('docs');
	});
});

describe('highlightMatch', () => {
	it('returns a single non-matching segment when query is empty', () => {
		expect(highlightMatch('abc', '')).toEqual([{ text: 'abc', match: false }]);
	});

	it('marks every case-insensitive occurrence of the query', () => {
		expect(highlightMatch('aXa', 'ax')).toEqual([
			{ text: 'aX', match: true },
			{ text: 'a', match: false }
		]);
	});

	it('returns non-matching text when the query is absent', () => {
		expect(highlightMatch('abc', 'z')).toEqual([{ text: 'abc', match: false }]);
	});
});
