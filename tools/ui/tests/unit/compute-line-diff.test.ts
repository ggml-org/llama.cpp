import { describe, expect, it } from 'vitest';
import {
	computeLineDiff,
	renderUnifiedDiff
} from '$lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/compute-line-diff';

describe('computeLineDiff', () => {
	it('returns empty for two empty inputs', () => {
		expect(computeLineDiff('', '')).toEqual([]);
	});

	it('marks every line as removed for an empty new text', () => {
		expect(computeLineDiff('a\nb\nc', '')).toEqual([
			{ kind: 'remove', text: 'a' },
			{ kind: 'remove', text: 'b' },
			{ kind: 'remove', text: 'c' }
		]);
	});

	it('marks every line as added for an empty old text', () => {
		expect(computeLineDiff('', 'a\nb')).toEqual([
			{ kind: 'add', text: 'a' },
			{ kind: 'add', text: 'b' }
		]);
	});

	it('detects a single-line replace', () => {
		expect(computeLineDiff('old', 'new')).toEqual([
			{ kind: 'add', text: 'new' },
			{ kind: 'remove', text: 'old' }
		]);
	});

	it('preserves interleaved context around additions', () => {
		const oldText = ['a', 'b', 'c'].join('\n');
		const newText = ['a', 'b', 'B', 'c'].join('\n');
		expect(computeLineDiff(oldText, newText)).toEqual([
			{ kind: 'context', text: 'a' },
			{ kind: 'context', text: 'b' },
			{ kind: 'add', text: 'B' },
			{ kind: 'context', text: 'c' }
		]);
	});

	it('preserves interleaved context around an isolated replace', () => {
		// Multi-line context around a one-line change -> the diff should
		// show context flanking the changed line at its natural position.
		const oldText = ['a', 'b', 'c', 'd'].join('\n');
		const newText = ['a', 'b', 'X', 'd'].join('\n');
		expect(computeLineDiff(oldText, newText)).toEqual([
			{ kind: 'context', text: 'a' },
			{ kind: 'context', text: 'b' },
			{ kind: 'add', text: 'X' },
			{ kind: 'remove', text: 'c' },
			{ kind: 'context', text: 'd' }
		]);
	});

	it('preserves interleaved context around removals', () => {
		const oldText = ['a', 'b', 'c', 'd'].join('\n');
		const newText = ['a', 'c', 'd'].join('\n');
		expect(computeLineDiff(oldText, newText)).toEqual([
			{ kind: 'context', text: 'a' },
			{ kind: 'remove', text: 'b' },
			{ kind: 'context', text: 'c' },
			{ kind: 'context', text: 'd' }
		]);
	});

	it('handles purely identical inputs', () => {
		const text = 'x\ny\nz';
		const result = computeLineDiff(text, text);
		expect(result).toEqual([
			{ kind: 'context', text: 'x' },
			{ kind: 'context', text: 'y' },
			{ kind: 'context', text: 'z' }
		]);
	});

	it('strips a trailing newline on the old/new inputs', () => {
		expect(computeLineDiff('a\n', 'a\nb\n')).toEqual([
			{ kind: 'context', text: 'a' },
			{ kind: 'add', text: 'b' }
		]);
	});

	it('normalizes trailing CR on each line', () => {
		expect(computeLineDiff('a\r\nb\r\n', 'a\nb')).toEqual([
			{ kind: 'context', text: 'a' },
			{ kind: 'context', text: 'b' }
		]);
	});
});

describe('renderUnifiedDiff', () => {
	it('returns empty string for empty diff', () => {
		expect(renderUnifiedDiff([])).toBe('');
	});

	it('prefixes each line with `+`, `-`, or a single space', () => {
		const lines = [
			{ kind: 'context' as const, text: 'ctx' },
			{ kind: 'add' as const, text: 'plus' },
			{ kind: 'remove' as const, text: 'minus' }
		];
		expect(renderUnifiedDiff(lines)).toBe(' ctx\n+plus\n-minus');
	});
});
