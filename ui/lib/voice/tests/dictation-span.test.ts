// Guards how dictation writes into the message: at the caret, replacing only its own text
// as the transcript grows, and keeping what the user typed around it.

import { DictationSpan } from '../src/dictation-span.js';
import { describe, expect, it } from 'vitest';

describe('DictationSpan', () => {
	it('writes into an empty message and replaces its own text as it grows', () => {
		const span = new DictationSpan('', 0);

		let value = span.apply('', 'Hello');

		expect(value).toBe('Hello');

		value = span.apply(value, 'Hello world');
		expect(value).toBe('Hello world');
		expect(span.end).toBe(value.length);
	});

	it('inserts at the caret with spaces, keeping the text before and after', () => {
		const start = 'Translate  into German';
		const span = new DictationSpan(start, 'Translate'.length);

		let value = span.apply(start, 'good');

		expect(value).toBe('Translate good  into German');

		value = span.apply(value, 'good morning');
		expect(value).toBe('Translate good morning  into German');
	});

	it('adds a space where the dictated text would run into a word', () => {
		const span = new DictationSpan('Say', 3);

		expect(span.apply('Say', 'hi')).toBe('Say hi');

		const middle = new DictationSpan('ab', 1);

		expect(middle.apply('ab', 'x')).toBe('a x b');
	});

	it('follows text typed before it while it streams', () => {
		const span = new DictationSpan('', 0);

		let value = span.apply('', 'one');

		const typed = `Note: ${value}`;

		span.edited(value, typed);
		value = span.apply(typed, 'one two');

		expect(value).toBe('Note: one two');
	});

	it('keeps text typed after it', () => {
		const span = new DictationSpan('', 0);

		let value = span.apply('', 'one');

		const typed = `${value}!`;

		span.edited(value, typed);
		value = span.apply(typed, 'one two');

		expect(value).toBe('one two !');
	});

	it('hands over text the user edits and dictates on after it', () => {
		const span = new DictationSpan('', 0);

		let value = span.apply('', 'helo');

		const fixed = 'hello';

		span.edited(value, fixed);
		value = span.apply(fixed, 'helo there');

		expect(value).toBe('hello there');
	});

	it('takes its text back out when the dictation is cancelled', () => {
		const start = 'Before after';
		const span = new DictationSpan(start, 6);
		const value = span.apply(start, 'middle');

		expect(value).toBe('Before middle after');
		expect(span.remove(value)).toBe(start);
		expect(span.isEmpty).toBe(true);
	});
});
