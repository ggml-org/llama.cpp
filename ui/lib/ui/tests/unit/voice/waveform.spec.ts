import { MIN_BAR_HEIGHT } from '../../../src/lib/constants/voice.constants.js';
import {
	barCapacity,
	barHeightPercent,
	formatClipPosition,
	formatClipTime,
	liveBars,
	pointerFraction,
	progressFraction,
	resamplePeaks,
	stepPosition
} from '../../../src/lib/voice/waveform.js';
import { describe, expect, it } from 'vite-plus/test';

describe('voice clip times', () => {
	it('formats m:ss, rounded to the second', () => {
		expect(formatClipTime(0)).toBe('0:00');
		expect(formatClipTime(7_000)).toBe('0:07');
		expect(formatClipTime(600)).toBe('0:01');
		expect(formatClipTime(65_400)).toBe('1:05');
		expect(formatClipTime(4_350_000)).toBe('72:30');
	});

	it('treats negative and non-finite times as zero', () => {
		expect(formatClipTime(-5)).toBe('0:00');
		expect(formatClipTime(Number.NaN)).toBe('0:00');
		expect(formatClipTime(Number.POSITIVE_INFINITY)).toBe('0:00');
	});

	it('says the position for screen readers', () => {
		expect(formatClipPosition(12_000, 65_000)).toBe('0:12 of 1:05');
	});
});

describe('waveform bars', () => {
	it('counts the bars that fit a width', () => {
		expect(barCapacity(100, 3, 2)).toBe(20);
		expect(barCapacity(98, 3, 2)).toBe(20);
		expect(barCapacity(97, 3, 2)).toBe(19);
		expect(barCapacity(0, 3, 2)).toBe(0);
		expect(barCapacity(Number.NaN, 3, 2)).toBe(0);
	});

	it('keeps peaks as they are when the count matches, clamped to 0..1', () => {
		expect(resamplePeaks([0.2, 1.4, -1, Number.NaN], 4)).toEqual([0.2, 1, 0, 0]);
	});

	it('keeps the loudest peak of each bucket when there are fewer bars', () => {
		expect(resamplePeaks([0.1, 0.9, 0.2, 0.3, 0.8, 0.1], 3)).toEqual([0.9, 0.3, 0.8]);
		expect(resamplePeaks([0.1, 0.2, 0.7, 0.4, 0.5], 2)).toEqual([0.2, 0.7]);
	});

	it('interpolates when there are more bars', () => {
		expect(resamplePeaks([0, 1], 5)).toEqual([0, 0.25, 0.5, 0.75, 1]);
		expect(resamplePeaks([0.4], 3)).toEqual([0.4, 0.4, 0.4]);
	});

	it('draws a flat line without peaks', () => {
		expect(resamplePeaks([], 3)).toEqual([0, 0, 0]);
		expect(resamplePeaks([0.5], 0)).toEqual([]);
	});

	it('fills the live width with the newest levels last', () => {
		// fewer levels than bars: stretched over the whole width
		expect(liveBars([0.1, 0.2, 0.3], 5).map((bar) => Number(bar?.toFixed(2)))).toEqual([
			0.1, 0.15, 0.2, 0.25, 0.3
		]);
		expect(liveBars([0.4], 3)).toEqual([0.4, 0.4, 0.4]);
		// more levels than bars: the newest ones scroll in from the right
		expect(liveBars([0.1, 0.2, 0.3, 2], 2)).toEqual([0.3, 1]);
		// nothing heard yet: the resting dots
		expect(liveBars([], 3)).toEqual([null, null, null]);
		expect(liveBars([0.5], 0)).toEqual([]);
	});

	it('never draws a bar under the resting height', () => {
		expect(barHeightPercent(null)).toBe(MIN_BAR_HEIGHT * 100);
		expect(barHeightPercent(0)).toBe(MIN_BAR_HEIGHT * 100);
		expect(barHeightPercent(0.5)).toBe(50);
		expect(barHeightPercent(3)).toBe(100);
	});
});

describe('seeking', () => {
	it('gives the played fraction', () => {
		expect(progressFraction(15_000, 60_000)).toBe(0.25);
		expect(progressFraction(90_000, 60_000)).toBe(1);
		expect(progressFraction(1_000, 0)).toBe(0);
	});

	it('maps a pointer to a fraction of the track', () => {
		expect(pointerFraction(150, 100, 200)).toBe(0.25);
		expect(pointerFraction(50, 100, 200)).toBe(0);
		expect(pointerFraction(400, 100, 200)).toBe(1);
		expect(pointerFraction(150, 100, 0)).toBe(0);
	});

	it('steps the position inside the clip', () => {
		expect(stepPosition(10_000, 5_000, 60_000)).toBe(15_000);
		expect(stepPosition(3_000, -5_000, 60_000)).toBe(0);
		expect(stepPosition(58_000, 5_000, 60_000)).toBe(60_000);
		expect(stepPosition(1_000, 5_000, 0)).toBe(0);
	});
});
