// Guards the one place the engine tells speech from the room: against a noise floor that
// drops at once and creeps back up, above an absolute minimum, with a stricter threshold
// for a listener that asks for one.

import { VOICE_SPEECH } from '../src/constants.js';
import { levelToDb, rmsToDb } from '../src/levels.js';
import { isSpeechLevel, SpeechDetector } from '../src/speech-detector.js';
import { describe, expect, it } from 'vitest';

/** Pushes `ms` of one level in 50 ms steps; true if the last one was speech. */
function hold(detector: SpeechDetector, db: number, ms: number): boolean {
	let speech = false;

	for (let t = 0; t < ms; t += 50) speech = detector.push(db, 50);

	return speech;
}

describe('SpeechDetector', () => {
	it('hears speech above a quiet room, and not the room', () => {
		const detector = new SpeechDetector();

		expect(hold(detector, -60, 500)).toBe(false);
		expect(hold(detector, -35, 300)).toBe(true);
		expect(detector.isQuiet(-58)).toBe(true);
	});

	it('follows a loud room: its noise is not speech, but speech over it is', () => {
		const detector = new SpeechDetector();

		expect(hold(detector, -30, 1000)).toBe(false);
		expect(hold(detector, -18, 300)).toBe(true);
	});

	it('lets the floor creep up through a long sound, so a steady one stops counting', () => {
		const detector = new SpeechDetector();

		hold(detector, -60, 200);

		const riseMs = ((VOICE_SPEECH.ABOVE_FLOOR_DB + 10) / VOICE_SPEECH.FLOOR_RISE_DB_PER_S) * 1000;

		expect(hold(detector, -41, 300)).toBe(true);
		expect(hold(detector, -41, riseMs)).toBe(false);
	});

	it('starts from a known floor, so the first level can be speech', () => {
		expect(new SpeechDetector({ floorDb: -50 }).push(-35, 50)).toBe(true);
		expect(new SpeechDetector().push(-35, 50)).toBe(false);
	});

	it('takes a stricter threshold when asked', () => {
		const strict = new SpeechDetector({ floorDb: -50, speechDb: 15 });

		expect(strict.isSpeech(-40)).toBe(false);
		expect(strict.isSpeech(-30)).toBe(true);
	});

	it('never hears speech below the absolute minimum, however quiet the room', () => {
		expect(isSpeechLevel(VOICE_SPEECH.MIN_DB - 1, -120)).toBe(false);
		expect(isSpeechLevel(VOICE_SPEECH.MIN_DB + 1, -120)).toBe(true);
	});
});

describe('dB helpers', () => {
	it('turns RMS and 0..1 levels into dBFS', () => {
		expect(rmsToDb(1)).toBe(0);
		expect(rmsToDb(0.1)).toBeCloseTo(-20);
		expect(rmsToDb(0)).toBeLessThan(-100);
		expect(levelToDb(1)).toBeCloseTo(0);
		expect(levelToDb(0.5)).toBe(-30);
		expect(levelToDb(0)).toBe(-60);
	});
});
