import { FileMentionEntryType } from '$lib/enums';
import {
	buildMentionInsertion,
	containsFileMentionLink,
	decodeFileLinkPath,
	encodeFileLinkPath,
	fileMentionLinkRe,
	getMentionBadgeIconPaths,
	getMentionBadgeLabel,
	MENTION_BADGE_FILE_ICON_PATHS,
	MENTION_BADGE_FOLDER_ICON_PATHS
} from '$lib/utils';
import { describe, expect, it } from 'vitest';

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

describe('fileMentionLinkRe', () => {
	it('matches a standard mention link', () => {
		expect(fileMentionLinkRe().test('[docs](file:///a/b)')).toBe(true);
		expect(containsFileMentionLink('[docs](file:///a/b)')).toBe(true);
	});

	it('does not match non-file links', () => {
		expect(fileMentionLinkRe().test('[foo](https://example.com)')).toBe(false);
		expect(fileMentionLinkRe().test('plain text')).toBe(false);
	});

	it('admits a close paren in a macOS-style file name', () => {
		const match = fileMentionLinkRe().exec(
			'[Screenshot (1).png](file:///Users/foo/Screenshot (1).png)'
		);

		expect(match).not.toBeNull();
		expect(match?.[1]).toBe('Screenshot (1).png');
		expect(match?.[2]).toBe('/Users/foo/Screenshot (1).png');
	});

	it('admits a parenthesized folder segment', () => {
		expect(
			fileMentionLinkRe().exec('[main.rs](file:///Users/foo/Project (Stuff)/main.rs)')?.[2]
		).toBe('/Users/foo/Project (Stuff)/main.rs');
	});

	it('stops at the closing paren of an adjacent badge', () => {
		expect(fileMentionLinkRe().exec('[a](file:///p)[b](file:///q)')?.[0]).toBe('[a](file:///p)');
	});
});

describe('getMentionBadgeIconPaths', () => {
	it('returns the folder glyphs for a trailing-separator path', () => {
		expect(getMentionBadgeIconPaths('/Users/foo/bar/')).toBe(MENTION_BADGE_FOLDER_ICON_PATHS);
	});

	it('returns the file glyphs otherwise', () => {
		expect(getMentionBadgeIconPaths('/Users/foo/bar.txt')).toBe(MENTION_BADGE_FILE_ICON_PATHS);
	});
});

describe('getMentionBadgeLabel', () => {
	it('returns the name by default', () => {
		expect(getMentionBadgeLabel('bar', '/Users/foo/bar/', false)).toBe('bar');
	});

	it('renders the decoded full path without the trailing separator', () => {
		expect(getMentionBadgeLabel('bar', '/Users/foo/bar/', true)).toBe('/Users/foo/bar');
		expect(getMentionBadgeLabel('shot', '/Users/foo/Screenshot%20(1).png', true)).toBe(
			'/Users/foo/Screenshot (1).png'
		);
	});

	it('abbreviates a known home prefix to a tilde', () => {
		expect(getMentionBadgeLabel('main.rs', '/home/user/src/main.rs', true, '/home/user')).toBe(
			'~/src/main.rs'
		);
	});

	it('falls back to the name when the decoded path is empty', () => {
		expect(getMentionBadgeLabel('root', '/', true)).toBe('root');
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

describe('buildMentionInsertion', () => {
	const file = (path: string, name: string) => ({
		name,
		path,
		type: FileMentionEntryType.FILE
	});
	const dir = (path: string, name: string) => ({
		name,
		path,
		type: FileMentionEntryType.DIRECTORY
	});

	it('splices a root-anchored file link in place of the token', () => {
		const value = 'hello @repo';
		const result = buildMentionInsertion(file('/Users/foo/myRepo', 'myRepo'), value, {
			end: 11,
			start: 6
		});

		expect(result).not.toBeNull();
		const { caretOffset, newValue } = result!;

		expect(newValue).toBe('hello [myRepo](file:///Users/foo/myRepo) ');
		expect(caretOffset).toBe(6 + '[myRepo](file:///Users/foo/myRepo) '.length);
	});

	it('keeps the trailing slash on the directory marker', () => {
		const value = 'see @src';
		const { newValue } = buildMentionInsertion(dir('/Users/foo/myRepo/src/', 'src'), value, {
			end: 8,
			start: 4
		})!;

		expect(newValue).toBe('see [src](file:///Users/foo/myRepo/src/) ');
	});

	it('escapes spaces and parens in the target', () => {
		const value = '@pic';
		const { newValue } = buildMentionInsertion(
			file('/Users/foo/Desktop/Pic (1).png', 'Pic (1).png'),
			value,
			{ end: 4, start: 0 }
		)!;

		expect(newValue).toBe('[Pic (1).png](file:///Users/foo/Desktop/Pic%20(1).png) ');
	});

	it('re-adds the directory marker when the cleaned path empties', () => {
		const { newValue } = buildMentionInsertion(dir('/', 'root'), '/', { end: 1, start: 0 })!;

		expect(newValue).toBe('[root](file:///) ');
	});

	it('returns null for an out-of-range token', () => {
		expect(buildMentionInsertion(file('/a/b.txt', 'b.txt'), 'x', { end: 5, start: 0 })).toBeNull();
		expect(buildMentionInsertion(file('/a/b.txt', 'b.txt'), 'x', { end: 1, start: 2 })).toBeNull();
	});
});
