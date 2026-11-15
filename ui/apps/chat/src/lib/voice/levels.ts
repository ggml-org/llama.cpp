import { VOICE_LEVELS_FLOOR_DB } from './constants';

/** Root mean square of a block of samples. */
export function rms(samples: Float32Array): number {
	if (samples.length === 0) return 0;

	let sum = 0;

	for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];

	return Math.sqrt(sum / samples.length);
}

/**
 * RMS to a 0..1 level that follows loudness as heard: decibels mapped linearly from
 * the floor (0) to full scale (1), so quiet speech still moves the waveform.
 */
export function rmsToLevel(value: number, floorDb = VOICE_LEVELS_FLOOR_DB): number {
	if (value <= 0) return 0;

	const db = 20 * Math.log10(value);

	return Math.max(0, Math.min(1, 1 - db / floorDb));
}

/** Fixed-length level history, newest last. */
export class LevelHistory {
	private values: number[];

	constructor(readonly length: number) {
		this.values = new Array<number>(length).fill(0);
	}

	/** Adds a value, drops the oldest, and returns a fresh copy for listeners. */
	push(level: number): number[] {
		this.values.push(level);
		this.values.shift();

		return this.values.slice();
	}

	reset(): void {
		this.values.fill(0);
	}

	snapshot(): number[] {
		return this.values.slice();
	}
}
