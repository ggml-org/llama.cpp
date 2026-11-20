// Guards where a recording is cut into pieces to transcribe while it goes on: at pauses
// after speech, judged against the room's noise, and at the latest after a long stretch.

import { PauseDetector } from '../src/pauses.js';
import { describe, expect, it } from 'vitest';

const INTERVAL = 50;

/** Pushes `ms` worth of one level; true if any of them cut. */
function hold(detector: PauseDetector, level: number, ms: number): boolean {
	let cut = false;

	for (let t = 0; t < ms; t += INTERVAL) cut = detector.push(level) || cut;

	return cut;
}

describe('PauseDetector', () => {
	it('cuts at a pause after speech', () => {
		const detector = new PauseDetector({ intervalMs: INTERVAL });

		expect(hold(detector, 0.1, 300)).toBe(false);
		expect(hold(detector, 0.6, 600)).toBe(false);
		expect(hold(detector, 0.1, 300)).toBe(false);
		expect(hold(detector, 0.1, 300)).toBe(true);
	});

	it('does not cut silence alone, nor a short sound', () => {
		const detector = new PauseDetector({ intervalMs: INTERVAL });

		expect(hold(detector, 0.1, 2000)).toBe(false);
		expect(hold(detector, 0.6, 100)).toBe(false);
		expect(hold(detector, 0.1, 1000)).toBe(false);
	});

	it('follows the noise floor of the room', () => {
		const detector = new PauseDetector({ intervalMs: INTERVAL });

		// a loud room: its noise is not speech, and speech above it still counts
		expect(hold(detector, 0.45, 500)).toBe(false);
		expect(hold(detector, 0.8, 600)).toBe(false);
		expect(hold(detector, 0.45, 600)).toBe(true);
	});

	it('cuts a long stretch of speech at its next quiet moment', () => {
		const detector = new PauseDetector({ intervalMs: INTERVAL, maxSegmentMs: 2000 });

		expect(hold(detector, 0.1, 100)).toBe(false);
		expect(hold(detector, 0.6, 2500)).toBe(false);
		expect(detector.push(0.1)).toBe(true);
	});

	it('cuts anyway when no quiet moment comes, as with music playing in the room', () => {
		const detector = new PauseDetector({ intervalMs: INTERVAL, maxSegmentMs: 2000 });

		expect(hold(detector, 0.1, 100)).toBe(false);
		// speech over music: the level never comes back down
		expect(hold(detector, 0.6, 2850)).toBe(false);
		expect(detector.push(0.6)).toBe(true);
		// and again for the next piece
		expect(hold(detector, 0.6, 2950)).toBe(false);
		expect(detector.push(0.6)).toBe(true);
	});
});
