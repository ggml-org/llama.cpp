import { generateConversationTitle, stripMentionLinks } from '#lib/utils/index.ts';
import { describe, expect, it } from 'vitest';

describe('stripMentionLinks', () => {
	it('replaces a mention with its label', () => {
		expect(stripMentionLinks('refactor [src/lib](file:///repo/src/lib/) please')).toBe(
			'refactor src/lib please'
		);
	});

	it('replaces every mention in the value', () => {
		expect(stripMentionLinks('diff [a.m](file:///repo/a.m) with [b.m](file:///repo/b.m)')).toBe(
			'diff a.m with b.m'
		);
	});

	it('keeps a directory label and drops its whole path, trailing slash included', () => {
		expect(stripMentionLinks('[apps](file:///repo/ui/apps/chat/) has a bug')).toBe(
			'apps has a bug'
		);
	});

	it('never leaks a percent-encoded path into the label', () => {
		expect(stripMentionLinks('[Shot (1).png](file:///tmp/Shot%20(1).png) is blurry')).toBe(
			'Shot (1).png is blurry'
		);
	});

	it('leaves an ordinary markdown link alone', () => {
		expect(stripMentionLinks('see [docs](https://example.com/guide)')).toBe(
			'see [docs](https://example.com/guide)'
		);
	});

	it('leaves a bare at-sign alone', () => {
		expect(stripMentionLinks('mail me at foo@example.com')).toBe('mail me at foo@example.com');
	});
});

describe('generateConversationTitle', () => {
	it('trims plain content', () => {
		expect(generateConversationTitle('  why is my build slow \n')).toBe('why is my build slow');
	});

	it('drops the file url from a leading mention', () => {
		expect(
			generateConversationTitle(
				'[llama.cpp](file:///Users/me/git/llama.cpp/) how do I build for metal?'
			)
		).toBe('llama.cpp how do I build for metal?');
	});

	it('collapses mentions on the picked line when useFirstLine is set', () => {
		const content =
			'compare [main.c](file:///repo/src/main.c) with [main.cpp](file:///repo/src/main.cpp)\n\nmore detail here';

		expect(generateConversationTitle(content, true)).toBe('compare main.c with main.cpp');
	});

	it('collapses before picking the first line, so a mention-only line still yields text', () => {
		expect(generateConversationTitle('\n[notes.md](file:///tmp/notes.md)\nrest', true)).toBe(
			'notes.md'
		);
	});

	it('returns an empty title for content that is nothing but whitespace', () => {
		expect(generateConversationTitle('  \n\t \n', true)).toBe('');
	});
});
