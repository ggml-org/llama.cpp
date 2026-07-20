import { describe, expect, it } from 'vitest';
import { DiffLineKind } from '$lib/enums';
import { computeLineDiff, renderUnifiedDiff, type DiffLine } from '$lib/utils';

describe('computeLineDiff', () => {
	it('returns empty for two empty inputs', () => {
		expect(computeLineDiff('', '')).toEqual([]);
	});

	it('marks every line as removed for an empty new text', () => {
		expect(computeLineDiff('a\nb\nc', '')).toEqual([
			{ kind: 'remove', text: 'a', oldLine: 1 },
			{ kind: 'remove', text: 'b', oldLine: 2 },
			{ kind: 'remove', text: 'c', oldLine: 3 }
		]);
	});

	it('marks every line as added for an empty old text', () => {
		expect(computeLineDiff('', 'a\nb')).toEqual([
			{ kind: 'add', text: 'a', newLine: 1 },
			{ kind: 'add', text: 'b', newLine: 2 }
		]);
	});

	it('detects a single-line replace', () => {
		expect(computeLineDiff('old', 'new')).toEqual([
			{ kind: 'add', text: 'new', newLine: 1 },
			{ kind: 'remove', text: 'old', oldLine: 1 }
		]);
	});

	it('preserves interleaved context around additions', () => {
		const oldText = ['a', 'b', 'c'].join('\n');
		const newText = ['a', 'b', 'B', 'c'].join('\n');
		expect(computeLineDiff(oldText, newText)).toEqual([
			{ kind: 'context', text: 'a', oldLine: 1, newLine: 1 },
			{ kind: 'context', text: 'b', oldLine: 2, newLine: 2 },
			{ kind: 'add', text: 'B', newLine: 3 },
			{ kind: 'context', text: 'c', oldLine: 3, newLine: 4 }
		]);
	});

	it('preserves interleaved context around an isolated replace', () => {
		// Multi-line context around a one-line change -> the diff should
		// show context flanking the changed line at its natural position.
		const oldText = ['a', 'b', 'c', 'd'].join('\n');
		const newText = ['a', 'b', 'X', 'd'].join('\n');
		expect(computeLineDiff(oldText, newText)).toEqual([
			{ kind: 'context', text: 'a', oldLine: 1, newLine: 1 },
			{ kind: 'context', text: 'b', oldLine: 2, newLine: 2 },
			{ kind: 'add', text: 'X', newLine: 3 },
			{ kind: 'remove', text: 'c', oldLine: 3 },
			{ kind: 'context', text: 'd', oldLine: 4, newLine: 4 }
		]);
	});

	it('preserves interleaved context around removals', () => {
		const oldText = ['a', 'b', 'c', 'd'].join('\n');
		const newText = ['a', 'c', 'd'].join('\n');
		expect(computeLineDiff(oldText, newText)).toEqual([
			{ kind: 'context', text: 'a', oldLine: 1, newLine: 1 },
			{ kind: 'remove', text: 'b', oldLine: 2 },
			{ kind: 'context', text: 'c', oldLine: 3, newLine: 2 },
			{ kind: 'context', text: 'd', oldLine: 4, newLine: 3 }
		]);
	});

	it('handles purely identical inputs', () => {
		const text = 'x\ny\nz';
		const result = computeLineDiff(text, text);
		expect(result).toEqual([
			{ kind: 'context', text: 'x', oldLine: 1, newLine: 1 },
			{ kind: 'context', text: 'y', oldLine: 2, newLine: 2 },
			{ kind: 'context', text: 'z', oldLine: 3, newLine: 3 }
		]);
	});

	it('strips a trailing newline on the old/new inputs', () => {
		expect(computeLineDiff('a\n', 'a\nb\n')).toEqual([
			{ kind: 'context', text: 'a', oldLine: 1, newLine: 1 },
			{ kind: 'add', text: 'b', newLine: 2 }
		]);
	});

	it('normalizes trailing CR on each line', () => {
		expect(computeLineDiff('a\r\nb\r\n', 'a\nb')).toEqual([
			{ kind: 'context', text: 'a', oldLine: 1, newLine: 1 },
			{ kind: 'context', text: 'b', oldLine: 2, newLine: 2 }
		]);
	});

	it('keeps line numbers monotonic across mixed add/remove/context', () => {
		const oldText = ['l1', 'l2', 'l3', 'l4', 'l5'].join('\n');
		const newText = ['l1', 'l2-EDIT', 'l3', 'l4-NEW', 'l5'].join('\n');
		const diff = computeLineDiff(oldText, newText);

		// Walk the diff: every oldLine must increase strictly, and every
		// newLine must increase strictly. Lines missing one side (add or
		// remove) carry no number on that side.
		let lastOld = 0;
		let lastNew = 0;
		for (const line of diff) {
			if (line.oldLine !== undefined) {
				expect(line.oldLine).toBeGreaterThan(lastOld);
				lastOld = line.oldLine;
			}
			if (line.newLine !== undefined) {
				expect(line.newLine).toBeGreaterThan(lastNew);
				lastNew = line.newLine;
			}
		}
	});
});

describe('renderUnifiedDiff', () => {
	it('returns empty string for empty diff', () => {
		expect(renderUnifiedDiff([])).toBe('');
	});

	it('prefixes each line with `+`, `-`, or a single space', () => {
		const lines: DiffLine[] = [
			{ kind: DiffLineKind.CONTEXT, text: 'ctx' },
			{ kind: DiffLineKind.ADD, text: 'plus' },
			{ kind: DiffLineKind.REMOVE, text: 'minus' }
		];
		expect(renderUnifiedDiff(lines)).toBe(' ctx\n+plus\n-minus');
	});

	it('ignores oldLine/newLine metadata when emitting prefixes', () => {
		const lines: DiffLine[] = [
			{ kind: DiffLineKind.CONTEXT, text: 'a', oldLine: 1, newLine: 1 },
			{ kind: DiffLineKind.ADD, text: 'b', newLine: 2 }
		];
		expect(renderUnifiedDiff(lines)).toBe(' a\n+b');
	});
});
