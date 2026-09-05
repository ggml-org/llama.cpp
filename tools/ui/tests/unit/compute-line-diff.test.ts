import { DiffLineKind } from '$lib/enums';
import { computeLineDiff, type DiffLine, renderUnifiedDiff } from '$lib/utils';
import { describe, expect, it } from 'vitest';

describe('computeLineDiff', () => {
	it('returns empty for two empty inputs', () => {
		expect(computeLineDiff('', '')).toEqual([]);
	});

	it('marks every line as removed for an empty new text', () => {
		expect(computeLineDiff('a\nb\nc', '')).toEqual([
			{ kind: 'remove', oldLine: 1, text: 'a' },
			{ kind: 'remove', oldLine: 2, text: 'b' },
			{ kind: 'remove', oldLine: 3, text: 'c' }
		]);
	});

	it('marks every line as added for an empty old text', () => {
		expect(computeLineDiff('', 'a\nb')).toEqual([
			{ kind: 'add', newLine: 1, text: 'a' },
			{ kind: 'add', newLine: 2, text: 'b' }
		]);
	});

	it('detects a single-line replace', () => {
		expect(computeLineDiff('old', 'new')).toEqual([
			{ kind: 'add', newLine: 1, text: 'new' },
			{ kind: 'remove', oldLine: 1, text: 'old' }
		]);
	});

	it('preserves interleaved context around additions', () => {
		const oldText = ['a', 'b', 'c'].join('\n');
		const newText = ['a', 'b', 'B', 'c'].join('\n');

		expect(computeLineDiff(oldText, newText)).toEqual([
			{ kind: 'context', newLine: 1, oldLine: 1, text: 'a' },
			{ kind: 'context', newLine: 2, oldLine: 2, text: 'b' },
			{ kind: 'add', newLine: 3, text: 'B' },
			{ kind: 'context', newLine: 4, oldLine: 3, text: 'c' }
		]);
	});

	it('preserves interleaved context around an isolated replace', () => {
		// Multi-line context around a one-line change -> the diff should
		// show context flanking the changed line at its natural position.
		const oldText = ['a', 'b', 'c', 'd'].join('\n');
		const newText = ['a', 'b', 'X', 'd'].join('\n');

		expect(computeLineDiff(oldText, newText)).toEqual([
			{ kind: 'context', newLine: 1, oldLine: 1, text: 'a' },
			{ kind: 'context', newLine: 2, oldLine: 2, text: 'b' },
			{ kind: 'add', newLine: 3, text: 'X' },
			{ kind: 'remove', oldLine: 3, text: 'c' },
			{ kind: 'context', newLine: 4, oldLine: 4, text: 'd' }
		]);
	});

	it('preserves interleaved context around removals', () => {
		const oldText = ['a', 'b', 'c', 'd'].join('\n');
		const newText = ['a', 'c', 'd'].join('\n');

		expect(computeLineDiff(oldText, newText)).toEqual([
			{ kind: 'context', newLine: 1, oldLine: 1, text: 'a' },
			{ kind: 'remove', oldLine: 2, text: 'b' },
			{ kind: 'context', newLine: 2, oldLine: 3, text: 'c' },
			{ kind: 'context', newLine: 3, oldLine: 4, text: 'd' }
		]);
	});

	it('handles purely identical inputs', () => {
		const text = 'x\ny\nz';
		const result = computeLineDiff(text, text);

		expect(result).toEqual([
			{ kind: 'context', newLine: 1, oldLine: 1, text: 'x' },
			{ kind: 'context', newLine: 2, oldLine: 2, text: 'y' },
			{ kind: 'context', newLine: 3, oldLine: 3, text: 'z' }
		]);
	});

	it('strips a trailing newline on the old/new inputs', () => {
		expect(computeLineDiff('a\n', 'a\nb\n')).toEqual([
			{ kind: 'context', newLine: 1, oldLine: 1, text: 'a' },
			{ kind: 'add', newLine: 2, text: 'b' }
		]);
	});

	it('normalizes trailing CR on each line', () => {
		expect(computeLineDiff('a\r\nb\r\n', 'a\nb')).toEqual([
			{ kind: 'context', newLine: 1, oldLine: 1, text: 'a' },
			{ kind: 'context', newLine: 2, oldLine: 2, text: 'b' }
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
			{ kind: DiffLineKind.CONTEXT, newLine: 1, oldLine: 1, text: 'a' },
			{ kind: DiffLineKind.ADD, newLine: 2, text: 'b' }
		];

		expect(renderUnifiedDiff(lines)).toBe(' a\n+b');
	});
});
