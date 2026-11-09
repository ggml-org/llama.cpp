import { SourceHistory } from '$lib/utils';
import { describe, expect, it } from 'vitest';

describe('SourceHistory', () => {
	it('coalesces pushes inside the group window into one undo step', () => {
		const h = new SourceHistory(100, 800);

		h.push({ caret: 0, value: '' }, 1000);
		h.push({ caret: 1, value: 'a' }, 1200);
		h.push({ caret: 2, value: 'ab' }, 1500);

		expect(h.undo({ caret: 3, value: 'abc' })).toEqual({ caret: 0, value: '' });
		expect(h.undo({ caret: 0, value: '' })).toBeNull();
	});

	it('starts a new group once the window has passed', () => {
		const h = new SourceHistory(100, 800);

		h.push({ caret: 0, value: '' }, 1000);
		h.push({ caret: 3, value: 'abc' }, 2000);

		expect(h.undo({ caret: 6, value: 'abcdef' })).toEqual({ caret: 3, value: 'abc' });
		expect(h.undo({ caret: 3, value: 'abc' })).toEqual({ caret: 0, value: '' });
	});

	it('newGroup forces a separate entry even inside the window', () => {
		const h = new SourceHistory(100, 800);

		h.push({ caret: 0, value: '' }, 1000);
		h.push({ caret: 3, value: 'abc' }, 1100, true);

		expect(h.undo({ caret: 4, value: 'abc\n' })).toEqual({ caret: 3, value: 'abc' });
		expect(h.undo({ caret: 3, value: 'abc' })).toEqual({ caret: 0, value: '' });
	});

	it('redo round-trips and a fresh push clears the redo stack', () => {
		const h = new SourceHistory(100, 800);

		h.push({ caret: 0, value: '' }, 1000);

		const undone = h.undo({ caret: 3, value: 'abc' });

		expect(undone).toEqual({ caret: 0, value: '' });
		expect(h.redo({ caret: 0, value: '' })).toEqual({ caret: 3, value: 'abc' });

		h.undo({ caret: 3, value: 'abc' });
		h.push({ caret: 0, value: '' }, 5000);
		expect(h.redo({ caret: 1, value: 'x' })).toBeNull();
	});

	it('starts a new group on the first edit after an undo', () => {
		const h = new SourceHistory(100, 800);

		h.push({ caret: 0, value: '' }, 1000);
		h.undo({ caret: 3, value: 'abc' });

		h.push({ caret: 0, value: '' }, 1200);
		expect(h.undo({ caret: 1, value: 'x' })).toEqual({ caret: 0, value: '' });
	});

	it('evicts the oldest entry past the limit', () => {
		const h = new SourceHistory(2, 800);

		h.push({ caret: 0, value: 'one' }, 1000);
		h.push({ caret: 0, value: 'two' }, 2000);
		h.push({ caret: 0, value: 'three' }, 3000);

		expect(h.undo({ caret: 0, value: 'cur' })).toEqual({ caret: 0, value: 'three' });
		expect(h.undo({ caret: 0, value: 'three' })).toEqual({ caret: 0, value: 'two' });
		expect(h.undo({ caret: 0, value: 'two' })).toBeNull();
	});
});
