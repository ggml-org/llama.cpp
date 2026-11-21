import { speechFraction, spokenLength, spokenWords } from '../../../src/lib/voice/spoken.js';
import { describe, expect, it } from 'vite-plus/test';

const TEXT = 'one two three four';

describe('spoken text', () => {
	it('lights nothing before playback and everything at its end', () => {
		expect(spokenLength(TEXT, [1, 1, 1, 1], 4000, 0)).toBe(0);
		expect(spokenLength(TEXT, [1, 1, 1, 1], 4000, 4000)).toBe(TEXT.length);
		expect(spokenLength(TEXT, [1, 1, 1, 1], 0, 100)).toBe(TEXT.length);
	});

	it('lights a word from its start, as a whole', () => {
		// a quarter of even speech reaches into "two"
		const length = spokenLength(TEXT, [1, 1, 1, 1], 4000, 1000);

		expect(TEXT.slice(0, length)).toBe('one two');
	});

	it('spreads the text over the speech, not over the pauses', () => {
		// the first half is silence: nothing is spoken yet halfway through
		expect(speechFraction([0, 0, 1, 1], 0.5)).toBe(0);
		expect(speechFraction([0, 0, 1, 1], 0.75)).toBe(0.5);
		expect(speechFraction([1, 0, 0, 1], 0.5)).toBe(0.5);
		// no speech to go by: the clip's own progress
		expect(speechFraction([], 0.3)).toBeCloseTo(0.3);
		expect(speechFraction([0, 0], 0.3)).toBeCloseTo(0.3);
	});

	it('cuts a text into words with their spaces and starts', () => {
		expect(spokenWords(' hi  there ')).toEqual([
			{ start: 0, word: ' hi  ' },
			{ start: 5, word: 'there ' }
		]);
		expect(spokenWords('')).toEqual([]);
	});
});
