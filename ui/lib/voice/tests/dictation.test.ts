// Guards dictation into a plain text field, without the chat form: text streams in at the
// caret, the user's typing around it stays, the whole recording's text replaces the live
// text when it comes, and a cancel or a voice message takes the text back out.

import { createDictation } from '../src/dictation.js';
import { describe, expect, it, vi } from 'vitest';

/** A text field: its value and caret, and a dictation writing into it. */
function field(value = '', caret = value.length) {
	const state = { caret, changes: 0, value };
	const onError = vi.fn();
	const dictation = createDictation({
		getCaretOffset: () => state.caret,
		getValue: () => state.value,
		onChange: () => state.changes++,
		onError,
		setCaretOffset: (offset) => (state.caret = offset),
		setValue: (next) => (state.value = next)
	});

	/** The user types: the field changes, and dictation is told. */
	const type = (next: string) => {
		state.value = next;
		dictation.edited();
	};

	return { dictation, onError, state, type };
}

describe('createDictation', () => {
	it('writes at the caret, keeping the text around it', () => {
		const { dictation, state } = field('Dear Ana, thanks', 9);

		dictation.begin();
		dictation.write('the notes');
		dictation.write('the notes were great');

		expect(state.value).toBe('Dear Ana, the notes were great thanks');
		expect(dictation.active).toBe(true);
	});

	it('keeps what the user types before the dictated text', () => {
		const { dictation, state, type } = field('');

		dictation.begin();
		dictation.write('hello');
		type('Oh, hello');
		dictation.write('hello there');

		expect(state.value).toBe('Oh, hello there');
	});

	it("replaces the live text with the whole recording's, then puts the caret after it", async () => {
		const { dictation, state } = field('Note:');

		dictation.begin();
		dictation.write('buy milk an bred');

		const finished = dictation.finish({
			liveText: 'buy milk an bred',
			transcript: Promise.resolve('buy milk and bread')
		});

		expect(dictation.refining).toBe(true);

		await finished;

		expect(state.value).toBe('Note: buy milk and bread');
		expect(state.caret).toBe(state.value.length);
		expect(dictation.refining).toBe(false);
		expect(dictation.active).toBe(false);
	});

	it('keeps the live text when the whole recording comes back empty or fails', async () => {
		const empty = field();

		empty.dictation.begin();
		empty.dictation.write('kept');
		await empty.dictation.finish({ liveText: 'kept', transcript: Promise.resolve('  ') });
		expect(empty.state.value).toBe('kept');

		const failed = field();

		failed.dictation.begin();
		failed.dictation.write('kept');
		await failed.dictation.finish({
			liveText: 'kept',
			transcript: Promise.reject(new Error('busy'))
		});
		expect(failed.state.value).toBe('kept');
		expect(failed.onError).toHaveBeenCalledOnce();
	});

	it('settles once the text is final', async () => {
		const { dictation } = field();
		let resolve!: (text: string) => void;

		dictation.begin();
		dictation.write('draft');

		const finished = dictation.finish({
			liveText: 'draft',
			transcript: new Promise((done) => (resolve = done))
		});
		const settled = vi.fn();

		void dictation.settled().then(settled);
		await Promise.resolve();
		expect(settled).not.toHaveBeenCalled();

		resolve('final');
		await finished;
		expect(settled).toHaveBeenCalled();
	});

	it('takes the text back out on cancel, and drops text still on its way', async () => {
		const { dictation, state } = field('Hi');

		dictation.begin();
		dictation.write('there');

		let resolve!: (text: string) => void;
		const finished = dictation.finish({
			liveText: 'there',
			transcript: new Promise((done) => (resolve = done))
		});

		dictation.cancel();
		resolve('there you go');
		await finished;

		expect(state.value).toBe('Hi');
		expect(dictation.active).toBe(false);
	});

	it("hands a voice message's text over with whether the user edited it", () => {
		const { dictation, state, type } = field('');

		dictation.begin({ at: 0, keepsEdits: true });
		dictation.write('send the file');
		type('send that file');

		expect(dictation.take()).toEqual({ text: 'send that file', userEdited: true });
		expect(state.value).toBe('');
		expect(dictation.take()).toBeNull();
	});

	it('tells the caller when its state changes', async () => {
		const { dictation, state } = field();

		dictation.begin();
		dictation.write('a');
		await dictation.finish({ liveText: 'a', transcript: Promise.resolve('a') });

		// begin, refining on and off, and the end
		expect(state.changes).toBe(4);
	});
});
