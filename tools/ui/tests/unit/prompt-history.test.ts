import {
	canScrollInDirection,
	isCaretOnFirstLine,
	isCaretOnLastLine,
	appendImportedPrompts,
	getPromptHistoryEntries,
	parsePromptHistory,
	parsePromptHistoryBuckets,
	pushPromptHistory,
	setPromptHistoryEntries,
	userPromptTextsFromMessages,
	recallNext,
	recallPrevious,
	shouldLockPageScroll,
	swipeDirection,
	wheelHistoryDirection
} from '$lib/utils/prompt-history';
import { describe, expect, it } from 'vitest';

describe('caret line helpers', () => {
	it('treats an empty field as both first and last line', () => {
		expect(isCaretOnFirstLine('', 0)).toBe(true);
		expect(isCaretOnLastLine('', 0)).toBe(true);
	});

	it('detects the first line only before a newline', () => {
		expect(isCaretOnFirstLine('hello\nworld', 5)).toBe(true);
		expect(isCaretOnFirstLine('hello\nworld', 6)).toBe(false);
	});

	it('detects the last line at and after the final newline', () => {
		expect(isCaretOnLastLine('hello\nworld', 6)).toBe(true);
		expect(isCaretOnLastLine('hello\nworld', 3)).toBe(false);
	});
});

describe('pushPromptHistory', () => {
	it('skips blank and duplicate consecutive prompts', () => {
		expect(pushPromptHistory([], '  ')).toEqual([]);
		expect(pushPromptHistory(['a'], 'a')).toEqual(['a']);
		expect(pushPromptHistory(['a'], 'b')).toEqual(['a', 'b']);
	});

	it('trims and caps the stack', () => {
		expect(pushPromptHistory(['a'], '  b  ')).toEqual(['a', 'b']);
		expect(pushPromptHistory(['1', '2', '3'], '4', 3)).toEqual(['2', '3', '4']);
	});
});

describe('parsePromptHistory', () => {
	it('returns an empty list for invalid payloads', () => {
		expect(parsePromptHistory(null)).toEqual([]);
		expect(parsePromptHistory('not-json')).toEqual([]);
		expect(parsePromptHistory('{"x":1}')).toEqual([]);
	});

	it('keeps only non-empty strings', () => {
		expect(parsePromptHistory(JSON.stringify(['ok', '', 2, 'also']))).toEqual(['ok', 'also']);
	});
});

describe('prompt history buckets', () => {
	it('migrates a legacy array into combined history', () => {
		expect(parsePromptHistoryBuckets(JSON.stringify(['a', 'b']))).toEqual({
			combined: ['a', 'b'],
			sessions: {}
		});
	});

	it('keeps combined and per-session lists independent', () => {
		const store = parsePromptHistoryBuckets(
			JSON.stringify({
				combined: ['all'],
				sessions: { chat1: ['one'], chat2: ['two'] }
			})
		);

		expect(getPromptHistoryEntries(store, 'combine', 'chat1')).toEqual(['all']);
		expect(getPromptHistoryEntries(store, 'separate', 'chat1')).toEqual(['one']);
		expect(getPromptHistoryEntries(store, 'separate', 'missing')).toEqual([]);

		const after = setPromptHistoryEntries(store, 'separate', 'chat1', ['one', 'new']);

		expect(after.combined).toEqual(['all']);
		expect(after.sessions.chat2).toEqual(['two']);
		expect(after.sessions.chat1).toEqual(['one', 'new']);
	});

	it('appends imported prompts to combined and that session', () => {
		const store = parsePromptHistoryBuckets(
			JSON.stringify({ combined: ['old'], sessions: { chat1: ['old'] } })
		);
		const next = appendImportedPrompts(store, 'chat1', ['hello', 'hello', 'world']);

		expect(next.combined).toEqual(['old', 'hello', 'world']);
		expect(next.sessions.chat1).toEqual(['old', 'hello', 'world']);
	});

	it('extracts non-synthetic user prompts in time order', () => {
		expect(
			userPromptTextsFromMessages([
				{ content: 'second', role: 'user', timestamp: 2 },
				{ content: 'first', role: 'user', timestamp: 1 },
				{ content: 'skip', isSynthetic: true, role: 'user', timestamp: 3 },
				{ content: 'nope', role: 'assistant', timestamp: 4 },
				{ content: '  ', role: 'user', timestamp: 5 }
			])
		).toEqual(['first', 'second']);
	});
});

describe('recallPrevious / recallNext', () => {
	const entries = ['first', 'second'];
	const draftCursor = { draft: '', index: 2 };

	it('walks back from the live draft and restores it on the way forward', () => {
		const prev = recallPrevious(entries, draftCursor, 'typing');

		expect(prev).toEqual({
			cursor: { draft: 'typing', index: 1 },
			value: 'second'
		});

		const next = recallNext(entries, prev!.cursor, prev!.value);

		expect(next).toEqual({
			cursor: { draft: 'typing', index: 2 },
			value: 'typing'
		});
	});

	it('stops at the oldest entry and at the draft', () => {
		expect(recallPrevious(entries, { draft: '', index: 0 }, 'first')).toBeNull();
		expect(recallNext(entries, draftCursor, 'typing')).toBeNull();
		expect(recallPrevious([], draftCursor, 'x')).toBeNull();
	});
});

describe('swipeDirection', () => {
	it('requires a mostly-vertical travel past the minimum', () => {
		expect(swipeDirection(0, -10, 48)).toBeNull();
		expect(swipeDirection(80, -50, 48)).toBeNull();
		expect(swipeDirection(0, -48, 48)).toBe('up');
		expect(swipeDirection(10, 60, 48)).toBe('down');
	});
});

describe('wheelHistoryDirection', () => {
	it('maps vertical wheel to history swipe directions', () => {
		expect(wheelHistoryDirection(-40, 0)).toBe('up');
		expect(wheelHistoryDirection(40, 0)).toBe('down');
	});

	it('ignores tiny or horizontal-dominant wheels', () => {
		expect(wheelHistoryDirection(0, 0)).toBeNull();
		expect(wheelHistoryDirection(0.4, 0)).toBeNull();
		expect(wheelHistoryDirection(-10, 40)).toBeNull();
	});
});

describe('canScrollInDirection', () => {
	it('is false when the element does not overflow', () => {
		const el = { clientHeight: 40, scrollHeight: 40, scrollTop: 0 } as HTMLElement;

		expect(canScrollInDirection(el, 'up')).toBe(false);
		expect(canScrollInDirection(el, 'down')).toBe(false);
	});

	it('follows remaining scroll room', () => {
		const el = { clientHeight: 40, scrollHeight: 120, scrollTop: 0 } as HTMLElement;

		expect(canScrollInDirection(el, 'up')).toBe(false);
		expect(canScrollInDirection(el, 'down')).toBe(true);

		el.scrollTop = 80;
		expect(canScrollInDirection(el, 'up')).toBe(true);
		expect(canScrollInDirection(el, 'down')).toBe(false);
	});
});

describe('shouldLockPageScroll', () => {
	it('locks vertical page movement when the input cannot scroll', () => {
		const el = { clientHeight: 40, scrollHeight: 40, scrollTop: 0 } as HTMLElement;

		expect(shouldLockPageScroll(0, -20, el)).toBe(true);
		expect(shouldLockPageScroll(0, 20, el)).toBe(true);
	});

	it('does not lock a mostly-horizontal or tiny move', () => {
		expect(shouldLockPageScroll(40, -10, undefined)).toBe(false);
		expect(shouldLockPageScroll(0, -1, undefined)).toBe(false);
	});

	it('lets an overflowing input keep scrolling instead of locking the page', () => {
		const el = { clientHeight: 40, scrollHeight: 120, scrollTop: 0 } as HTMLElement;

		expect(shouldLockPageScroll(0, 30, el)).toBe(false);
		expect(shouldLockPageScroll(0, -30, el)).toBe(true);
	});
});
