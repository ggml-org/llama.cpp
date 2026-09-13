import { findMentionToken, takeMentionDismissSnapshot } from '$lib/utils';
import { describe, expect, it } from 'vitest';

describe('findMentionToken', () => {
	it('returns null for an empty/bare cursor', () => {
		expect(findMentionToken('', 0)).toBeNull();
		expect(findMentionToken('text', 0)).toBeNull();
	});

	it('recognizes a mention at the start of the value', () => {
		expect(findMentionToken('@pr', 3)).toEqual({ end: 3, query: 'pr', start: 0 });
	});

	it('recognizes a mention after a word boundary', () => {
		expect(findMentionToken('hello @pr', 9)).toEqual({ end: 9, query: 'pr', start: 6 });
	});

	it('returns null when the @ is mid-identifier', () => {
		expect(findMentionToken('em@', 3)).toBeNull();
		expect(findMentionToken('text@pr', 7)).toBeNull();
	});

	it('returns null when the cursor is past the whitespace break', () => {
		expect(findMentionToken('@pr hello', 9)).toBeNull();
	});

	it('treats boundary characters (parens, brackets, comma) as token starts', () => {
		expect(findMentionToken('(@pr', 4)).toEqual({ end: 4, query: 'pr', start: 1 });
		expect(findMentionToken('[@pr', 4)).toEqual({ end: 4, query: 'pr', start: 1 });
		expect(findMentionToken('a,@pr', 5)).toEqual({ end: 5, query: 'pr', start: 2 });
	});

	it('does not treat an identifier character as a boundary', () => {
		expect(findMentionToken('user@abc', 8)).toBeNull();
	});

	it('extracts the whole token up to the trailing boundary as the query', () => {
		expect(findMentionToken('@', 1)).toEqual({ end: 1, query: '', start: 0 });
		expect(findMentionToken('@hello', 6)).toEqual({ end: 6, query: 'hello', start: 0 });
	});

	it('keeps the whole token as the query when the caret is mid-token', () => {
		expect(findMentionToken('@hello', 4)).toEqual({ end: 6, query: 'hello', start: 0 });
		expect(findMentionToken('@hello world', 4)).toEqual({ end: 6, query: 'hello', start: 0 });
	});

	it('ignores a boundary @ and keeps the most recent token', () => {
		expect(findMentionToken('a @foo @bar', 11)).toEqual({ end: 11, query: 'bar', start: 7 });
	});
});

describe('takeMentionDismissSnapshot', () => {
	it('returns null when there is no valid mention at the cursor', () => {
		expect(takeMentionDismissSnapshot('plain text', 5)).toBeNull();
		expect(takeMentionDismissSnapshot('user@abc', 8)).toBeNull();
	});

	it('captures start and query of the current mention', () => {
		expect(takeMentionDismissSnapshot('hello @proj', 11)).toEqual({
			query: 'proj',
			start: 6
		});
	});
});
