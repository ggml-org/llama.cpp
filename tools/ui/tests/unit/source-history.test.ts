import { describe, expect, it } from 'vitest';
import { SourceHistory } from '$lib/utils';

describe('SourceHistory', () => {
	it('coalesces pushes inside the group window into one undo step', () => {
		const h = new SourceHistory(100, 800);
		h.push({ value: '', caret: 0 }, 1000);
		h.push({ value: 'a', caret: 1 }, 1200);
		h.push({ value: 'ab', caret: 2 }, 1500);

		expect(h.undo({ value: 'abc', caret: 3 })).toEqual({ value: '', caret: 0 });
		expect(h.undo({ value: '', caret: 0 })).toBeNull();
	});

	it('starts a new group once the window has passed', () => {
		const h = new SourceHistory(100, 800);
		h.push({ value: '', caret: 0 }, 1000);
		h.push({ value: 'abc', caret: 3 }, 2000);

		expect(h.undo({ value: 'abcdef', caret: 6 })).toEqual({ value: 'abc', caret: 3 });
		expect(h.undo({ value: 'abc', caret: 3 })).toEqual({ value: '', caret: 0 });
	});

	it('newGroup forces a separate entry even inside the window', () => {
		const h = new SourceHistory(100, 800);
		h.push({ value: '', caret: 0 }, 1000);
		h.push({ value: 'abc', caret: 3 }, 1100, true);

		expect(h.undo({ value: 'abc\n', caret: 4 })).toEqual({ value: 'abc', caret: 3 });
		expect(h.undo({ value: 'abc', caret: 3 })).toEqual({ value: '', caret: 0 });
	});

	it('redo round-trips and a fresh push clears the redo stack', () => {
		const h = new SourceHistory(100, 800);
		h.push({ value: '', caret: 0 }, 1000);

		const undone = h.undo({ value: 'abc', caret: 3 });
		expect(undone).toEqual({ value: '', caret: 0 });
		expect(h.redo({ value: '', caret: 0 })).toEqual({ value: 'abc', caret: 3 });

		h.undo({ value: 'abc', caret: 3 });
		h.push({ value: '', caret: 0 }, 5000);
		expect(h.redo({ value: 'x', caret: 1 })).toBeNull();
	});

	it('starts a new group on the first edit after an undo', () => {
		const h = new SourceHistory(100, 800);
		h.push({ value: '', caret: 0 }, 1000);
		h.undo({ value: 'abc', caret: 3 });

		h.push({ value: '', caret: 0 }, 1200);
		expect(h.undo({ value: 'x', caret: 1 })).toEqual({ value: '', caret: 0 });
	});

	it('evicts the oldest entry past the limit', () => {
		const h = new SourceHistory(2, 800);
		h.push({ value: 'one', caret: 0 }, 1000);
		h.push({ value: 'two', caret: 0 }, 2000);
		h.push({ value: 'three', caret: 0 }, 3000);

		expect(h.undo({ value: 'cur', caret: 0 })).toEqual({ value: 'three', caret: 0 });
		expect(h.undo({ value: 'three', caret: 0 })).toEqual({ value: 'two', caret: 0 });
		expect(h.undo({ value: 'two', caret: 0 })).toBeNull();
	});
});
