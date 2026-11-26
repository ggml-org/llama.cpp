import {
	findMentionToken,
	isMentionDismissed,
	takeMentionDismissSnapshot
} from '#lib/utils/index.ts';
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

	it('keeps a single space before the caret in the token', () => {
		expect(findMentionToken('@My Documents', 13)).toEqual({
			end: 13,
			query: 'My Documents',
			start: 0
		});
		expect(findMentionToken('see @My Doc', 11)).toEqual({ end: 11, query: 'My Doc', start: 4 });
		expect(findMentionToken('@pr ', 4)).toEqual({ end: 4, query: 'pr ', start: 0 });
	});

	it('ends the token at two spaces, a newline or a space right after the @', () => {
		expect(findMentionToken('@pr  ', 5)).toBeNull();
		expect(findMentionToken('@pr  hello', 10)).toBeNull();
		expect(findMentionToken('@pr\nhello', 9)).toBeNull();
		expect(findMentionToken('@ noon', 6)).toBeNull();
	});

	it('leaves the text after the caret out of the token', () => {
		expect(findMentionToken('@My Doc and more', 7)).toEqual({ end: 7, query: 'My Doc', start: 0 });
		expect(findMentionToken('@My Docu and more', 7)).toEqual({
			end: 8,
			query: 'My Docu',
			start: 0
		});
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

describe('isMentionDismissed', () => {
	const snapshot = { query: 'pr', start: 6 };

	it('holds for the same token and for words typed after it', () => {
		expect(isMentionDismissed(snapshot, { query: 'pr', start: 6 })).toBe(true);
		expect(isMentionDismissed(snapshot, { query: 'pr is broken', start: 6 })).toBe(true);
	});

	it('ends when the token itself changes or another one starts', () => {
		expect(isMentionDismissed(snapshot, { query: 'pro', start: 6 })).toBe(false);
		expect(isMentionDismissed(snapshot, { query: 'pr', start: 9 })).toBe(false);
		expect(isMentionDismissed(null, { query: 'pr', start: 6 })).toBe(false);
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
