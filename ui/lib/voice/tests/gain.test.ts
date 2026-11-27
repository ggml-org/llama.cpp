// Guards how quiet speech is raised for transcription: only speech is measured, the gain
// only raises and stops at its cap, loud peaks bend instead of clipping, and speech that
// stays quiet even at the cap is told.

import { VOICE_GAIN } from '../src/constants.js';
import {
	isTooQuiet,
	measureSpeech,
	normalizeSpeech,
	softLimit,
	SpeechGain,
	speechGainFor
} from '../src/gain.js';
import { floatToPcm16, pcm16ToFloat } from '../src/pcm.js';
import { describe, expect, it } from 'vitest';

const RATE = 16_000;

/** `ms` of a 220 Hz tone whose RMS is `level`, like a stretch of speech. */
function tone(level: number, ms: number): Float32Array {
	const out = new Float32Array(Math.round((RATE * ms) / 1000));
	const amplitude = level * Math.SQRT2;

	for (let i = 0; i < out.length; i++)
		out[i] = amplitude * Math.sin((2 * Math.PI * 220 * i) / RATE);

	return out;
}

/** `ms` of faint noise, a quiet room between words. */
function room(ms: number, level = 0.0005): Float32Array {
	const out = new Float32Array(Math.round((RATE * ms) / 1000));

	// deterministic: the same "noise" every run
	let seed = 7;

	for (let i = 0; i < out.length; i++) {
		seed = (seed * 16807) % 2147483647;
		out[i] = ((seed / 2147483647) * 2 - 1) * level * Math.sqrt(3);
	}

	return out;
}

function join(...parts: Float32Array[]): Float32Array {
	const out = new Float32Array(parts.reduce((n, part) => n + part.length, 0));

	let offset = 0;

	for (const part of parts) {
		out.set(part, offset);
		offset += part.length;
	}

	return out;
}

/** A recording: silence, speech at `level`, a pause, more speech, silence. */
function recording(level: number): Float32Array {
	return join(room(500), tone(level, 800), room(300), tone(level, 600), room(500));
}

function rmsOf(samples: Float32Array): number {
	let sum = 0;

	for (const sample of samples) sum += sample * sample;

	return Math.sqrt(sum / samples.length);
}

describe('speechGainFor', () => {
	it('raises quiet speech to the target and stops at the cap', () => {
		expect(speechGainFor(VOICE_GAIN.TARGET_RMS / 4)).toBeCloseTo(4);
		expect(speechGainFor(VOICE_GAIN.TARGET_RMS / 100)).toBe(VOICE_GAIN.MAX);
	});

	it('never lowers speech that is already loud enough', () => {
		expect(speechGainFor(VOICE_GAIN.TARGET_RMS * 3)).toBe(1);
		expect(speechGainFor(0)).toBe(1);
	});
});

describe('softLimit', () => {
	it('passes samples under the knee and bends louder ones below full scale', () => {
		expect(softLimit(0.5)).toBe(0.5);
		expect(softLimit(-0.5)).toBe(-0.5);
		expect(softLimit(3)).toBeLessThan(1);
		expect(softLimit(3)).toBeGreaterThan(VOICE_GAIN.KNEE);
		expect(softLimit(-3)).toBeCloseTo(-softLimit(3));
	});
});

describe('measureSpeech', () => {
	it('measures the speech, not the silence around it', () => {
		const speech = measureSpeech(recording(0.02), RATE);

		expect(speech).toBeGreaterThan(0.018);
		expect(speech).toBeLessThan(0.022);
	});

	it('finds no speech in a quiet room', () => {
		expect(measureSpeech(room(2000), RATE)).toBe(0);
	});
});

describe('normalizeSpeech', () => {
	it('raises a quiet recording so its speech reaches the target', () => {
		const result = normalizeSpeech(floatToPcm16(recording(0.02)), RATE);

		expect(result.gain).toBeCloseTo(VOICE_GAIN.TARGET_RMS / 0.02, 0);
		expect(result.tooQuiet).toBe(false);
		expect(measureSpeech(pcm16ToFloat(result.samples), RATE)).toBeGreaterThan(0.08);
	});

	it('gives the first words the same gain as the rest', () => {
		const result = normalizeSpeech(floatToPcm16(recording(0.02)), RATE);
		const raised = pcm16ToFloat(result.samples);
		// the first and the last 100 ms of speech
		const first = raised.subarray(RATE * 0.5, RATE * 0.6);
		const last = raised.subarray(RATE * 2.1, RATE * 2.2);

		expect(rmsOf(first)).toBeCloseTo(rmsOf(last), 2);
	});

	it('leaves loud speech as it is', () => {
		const samples = floatToPcm16(recording(0.2));
		const result = normalizeSpeech(samples, RATE);

		expect(result.gain).toBe(1);
		expect(result.samples).toBe(samples);
	});

	it('stops at the cap and tells that the speech is still too quiet', () => {
		const result = normalizeSpeech(floatToPcm16(recording(0.002)), RATE);

		expect(result.gain).toBe(VOICE_GAIN.MAX);
		expect(result.tooQuiet).toBe(true);
		expect(isTooQuiet(0.002)).toBe(true);
	});

	it('does not clip a loud peak in quiet speech', () => {
		const loudPeak = join(recording(0.02), tone(0.3, 50));
		const raised = pcm16ToFloat(normalizeSpeech(floatToPcm16(loudPeak), RATE).samples);

		expect(Math.max(...raised.map(Math.abs))).toBeLessThan(1);
	});

	it('leaves a recording with no speech alone', () => {
		const samples = floatToPcm16(room(2000));

		expect(normalizeSpeech(samples, RATE).samples).toBe(samples);
	});
});

describe('SpeechGain', () => {
	/** Feeds `samples` in 100 ms blocks; returns the gain after each. */
	function feed(gain: SpeechGain, samples: Float32Array, apply = true): number[] {
		const gains: number[] = [];

		for (let i = 0; i < samples.length; i += RATE / 10) {
			gain.process(samples.subarray(i, i + RATE / 10), apply);
			gains.push(gain.gain);
		}

		return gains;
	}

	it('rises over a few blocks of quiet speech, and holds through the silence after', () => {
		const gain = new SpeechGain();

		feed(gain, room(500));
		expect(gain.gain).toBe(1);

		const rising = feed(gain, tone(0.02, 800));

		expect(rising[0]).toBeGreaterThan(1);
		expect(rising.at(-1)).toBeGreaterThan(rising[0]);
		expect(rising.at(-1)).toBeGreaterThan(3.5);

		const held = gain.gain;

		feed(gain, room(500));
		expect(gain.gain).toBe(held);
	});

	it('falls at once when the speech gets loud, so it does not clip', () => {
		const gain = new SpeechGain();

		feed(gain, join(room(300), tone(0.02, 800)));
		expect(gain.gain).toBeGreaterThan(3);

		const out = gain.process(tone(0.3, 100));

		expect(gain.gain).toBeLessThan(3);
		expect(Math.max(...out.map(Math.abs))).toBeLessThan(1);
	});

	it('only measures when asked not to apply', () => {
		const gain = new SpeechGain();
		const block = tone(0.02, 100);

		feed(gain, room(300), false);

		expect(gain.process(block, false)).toBe(block);
		expect(gain.gain).toBeGreaterThan(1);
	});

	it('tells when speech stays too quiet even at the cap', () => {
		const gain = new SpeechGain();

		feed(gain, join(room(300), tone(0.025, 500)));
		expect(gain.tooQuiet).toBe(false);

		const faint = new SpeechGain();

		feed(faint, join(room(300, 0.0001), tone(0.002, 500)));
		expect(faint.tooQuiet).toBe(true);
	});
});
