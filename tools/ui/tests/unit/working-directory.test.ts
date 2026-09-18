import { GLOB, SEARCH } from '$lib/constants';
import {
	buildCaseInsensitiveGlob,
	buildGlobSearchArgs,
	highlightMatch,
	joinPath,
	rankEntries,
	splitPathQuery
} from '$lib/utils';
import { describe, expect, it } from 'vitest';

describe('splitPathQuery', () => {
	it('treats a plain query as a home-relative glob (not navigation)', () => {
		expect(splitPathQuery('docs')).toBeNull();
	});

	it('navigates the root for `/`', () => {
		expect(splitPathQuery('/')).toEqual({ last: '', parent: '/' });
	});

	it('navigates home for `~`', () => {
		expect(splitPathQuery('~')).toEqual({ last: '', parent: '~' });
	});

	it('splits an absolute path into parent and last segment', () => {
		expect(splitPathQuery('/Users/al/proj')).toEqual({ last: 'proj', parent: '/Users/al' });
	});

	it('navigates a Windows drive path written with backslashes', () => {
		expect(splitPathQuery('C:\\repos\\llama.cpp')).toEqual({
			last: 'llama.cpp',
			parent: 'C:/repos'
		});
	});

	it('navigates a Windows drive path written with forward slashes', () => {
		expect(splitPathQuery('D:/repos')).toEqual({ last: 'repos', parent: 'D:/' });
	});

	it('treats a bare drive as its root', () => {
		expect(splitPathQuery('D:')).toEqual({ last: '', parent: 'D:/' });
		expect(splitPathQuery('D:\\')).toEqual({ last: '', parent: 'D:/' });
	});

	it('navigates a UNC share', () => {
		expect(splitPathQuery('\\\\host\\share\\proj')).toEqual({
			last: 'proj',
			parent: '//host/share/'
		});
	});

	it('keeps a backslash as a POSIX filename character', () => {
		expect(splitPathQuery('/tmp/a\\b')).toEqual({ last: 'a\\b', parent: '/tmp' });
	});

	it('splits a home-relative path into parent and last segment', () => {
		expect(splitPathQuery('~/Documents')).toEqual({ last: 'Documents', parent: '~' });
	});

	it('strips trailing slashes before splitting', () => {
		expect(splitPathQuery('/Users/al/')).toEqual({ last: 'al', parent: '/Users' });
	});

	it('handles a single-segment absolute path', () => {
		expect(splitPathQuery('/opt')).toEqual({ last: 'opt', parent: '/' });
	});
});

describe('buildCaseInsensitiveGlob', () => {
	it('wraps letters in case-insensitive character classes', () => {
		expect(buildCaseInsensitiveGlob('ab')).toBe('*[aA][bB]*');
	});

	it('escapes glob metacharacters into literal fragments', () => {
		expect(buildCaseInsensitiveGlob('a*b')).toBe('*[aA][*][bB]*');
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
		expect(highlightMatch('abc', '')).toEqual([{ match: false, text: 'abc' }]);
	});

	it('marks every case-insensitive occurrence of the query', () => {
		expect(highlightMatch('aXa', 'ax')).toEqual([
			{ match: true, text: 'aX' },
			{ match: false, text: 'a' }
		]);
	});

	it('returns non-matching text when the query is absent', () => {
		expect(highlightMatch('abc', 'z')).toEqual([{ match: false, text: 'abc' }]);
	});
});

describe('buildGlobSearchArgs', () => {
	const DEPTH = 6;

	it('glob-matches home-relative within the scope path', () => {
		const args = buildGlobSearchArgs('docs', '/home', DEPTH);

		expect(args.path).toBe('/home');
		expect(args.include).toBe(buildCaseInsensitiveGlob('docs'));
		expect(args.maxDepth).toBe(DEPTH);
		expect(args.rankQuery).toBe('docs');
		expect(args.last).toBeUndefined();
	});

	it('navigates home for a `~` path query', () => {
		const args = buildGlobSearchArgs('~/proj', '/home', DEPTH);

		expect(args.path).toBe('~');
		expect(args.include).toBe(buildCaseInsensitiveGlob('proj'));
		expect(args.maxDepth).toBe(SEARCH.PATH_NAV_MAX_DEPTH);
		expect(args.rankQuery).toBe('proj');
		expect(args.last).toBe('proj');
	});

	it('lists the scope root when a path query has no last segment', () => {
		const args = buildGlobSearchArgs('~/', '/home', DEPTH);

		expect(args.path).toBe('~');
		expect(args.include).toBe(GLOB.WILDCARD);
		expect(args.maxDepth).toBe(SEARCH.PATH_NAV_MAX_DEPTH);
	});

	it('navigates an absolute path under its root', () => {
		const args = buildGlobSearchArgs('/usr/local/bin', '/home', DEPTH);

		expect(args.path).toBe('/usr/local');
		expect(args.include).toBe(buildCaseInsensitiveGlob('bin'));
		expect(args.maxDepth).toBe(SEARCH.PATH_NAV_MAX_DEPTH);
		expect(args.rankQuery).toBe('bin');
		expect(args.last).toBe('bin');
	});
});
