// Guards the voice engine's audio math: PCM16 conversion, the streaming resampler,
// the WAV the server accepts (16-bit PCM mono header fields), peaks and levels.

import { LevelHistory, rms, rmsToLevel } from '../src/levels.js';
import {
	base64ToBytes,
	concatPcm16,
	createResampler,
	floatToPcm16,
	pcm16ToBase64,
	pcm16ToFloat
} from '../src/pcm.js';
import { computePeaks } from '../src/peaks.js';
import { encodeWav } from '../src/wav.js';
import { describe, expect, it } from 'vitest';

function sine(rate: number, seconds: number, frequency: number, amplitude = 0.5): Float32Array {
	const out = new Float32Array(Math.round(rate * seconds));

	for (let i = 0; i < out.length; i++) {
		out[i] = amplitude * Math.sin((2 * Math.PI * frequency * i) / rate);
	}

	return out;
}

describe('PCM16 conversion', () => {
	it('maps full scale to the int16 range and clips beyond it', () => {
		expect(Array.from(floatToPcm16(Float32Array.from([0, 1, -1, 2, -2, 0.5])))).toEqual([
			0, 32767, -32768, 32767, -32768, 16383
		]);
	});

	it('round-trips through floats within one step', () => {
		const input = Float32Array.from([0.25, -0.75, 0.1]);
		const back = pcm16ToFloat(floatToPcm16(input));

		for (let i = 0; i < input.length; i++) expect(back[i]).toBeCloseTo(input[i], 4);
	});

	it('joins chunks in order', () => {
		expect(
			Array.from(concatPcm16([Int16Array.from([1, 2]), Int16Array.from([]), Int16Array.from([3])]))
		).toEqual([1, 2, 3]);
	});

	it('encodes little-endian base64', () => {
		const bytes = base64ToBytes(pcm16ToBase64(Int16Array.from([1, -2, 0x1234])));

		expect(Array.from(bytes)).toEqual([0x01, 0x00, 0xfe, 0xff, 0x34, 0x12]);
	});

	it('encodes long inputs without overflowing the call stack', () => {
		const samples = new Int16Array(200_000).fill(7);

		expect(base64ToBytes(pcm16ToBase64(samples)).length).toBe(400_000);
	});
});

describe('createResampler', () => {
	it('passes the same rate through', () => {
		const input = Float32Array.from([0.1, 0.2]);

		expect(createResampler(16_000, 16_000).process(input)).toBe(input);
	});

	it('downsamples 48 kHz to 16 kHz in blocks, holding back at most the last sample', () => {
		const resampler = createResampler(48_000, 16_000);
		const input = sine(48_000, 1, 440);

		let total = 0;

		// odd block sizes, as the audio thread may deliver
		for (let offset = 0; offset < input.length; offset += 1000) {
			total += resampler.process(input.subarray(offset, offset + 1000)).length;
		}

		// the last output waits for the input after it
		expect(total).toBeGreaterThanOrEqual(15_999);
		expect(total).toBeLessThanOrEqual(16_000);
	});

	it('downsamples 44.1 kHz to within one sample of the expected length', () => {
		const resampler = createResampler(44_100, 16_000);
		const out = resampler.process(sine(44_100, 1, 300));

		expect(Math.abs(out.length - 16_000)).toBeLessThanOrEqual(2);
	});

	it('keeps a speech-band tone and its level', () => {
		const out = createResampler(48_000, 16_000).process(sine(48_000, 0.5, 440, 0.5));

		expect(rms(out)).toBeCloseTo(0.5 / Math.SQRT2, 2);
	});

	it('averages away a tone above the new Nyquist rate', () => {
		// 12 kHz aliases into speech at 16 kHz unless it is filtered out
		const out = createResampler(48_000, 16_000).process(sine(48_000, 0.5, 12_000, 0.5));

		expect(rms(out)).toBeLessThan(0.05);
	});

	it('upsamples by linear interpolation', () => {
		const out = createResampler(8_000, 16_000).process(Float32Array.from([0, 1, 0, -1, 0]));

		expect(Array.from(out)).toEqual([0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5]);
	});
});

describe('encodeWav', () => {
	it('writes a 16-bit PCM mono header the server accepts', async () => {
		const samples = Int16Array.from([0, 1000, -1000, 32767]);
		const blob = encodeWav(samples, 16_000);
		const view = new DataView(await blob.arrayBuffer());
		const ascii = (offset: number) =>
			String.fromCharCode(...new Uint8Array(view.buffer, offset, 4));

		expect(blob.type).toBe('audio/wav');
		expect(view.byteLength).toBe(44 + 8);
		expect(ascii(0)).toBe('RIFF');
		expect(view.getUint32(4, true)).toBe(36 + 8);
		expect(ascii(8)).toBe('WAVE');
		expect(ascii(12)).toBe('fmt ');
		expect(view.getUint32(16, true)).toBe(16);
		expect(view.getUint16(20, true)).toBe(1); // PCM
		expect(view.getUint16(22, true)).toBe(1); // mono
		expect(view.getUint32(24, true)).toBe(16_000);
		expect(view.getUint32(28, true)).toBe(32_000);
		expect(view.getUint16(32, true)).toBe(2);
		expect(view.getUint16(34, true)).toBe(16);
		expect(ascii(36)).toBe('data');
		expect(view.getUint32(40, true)).toBe(8);
		expect([0, 1, 2, 3].map((i) => view.getInt16(44 + i * 2, true))).toEqual([
			0, 1000, -1000, 32767
		]);
	});
});

describe('computePeaks', () => {
	it('gives the normalised loudest sample of each span', () => {
		const samples = Float32Array.from([0.1, -0.2, 0.4, -0.1, 0, 0.05, -0.8, 0.2]);

		expect(computePeaks(samples, 4)).toEqual([0.25, 0.5, 0.0625, 1]);
	});

	it('reads PCM16 the same as floats', () => {
		const floats = sine(16_000, 0.2, 200);
		const fromPcm = computePeaks(floatToPcm16(floats), 16);
		const fromFloats = computePeaks(floats, 16);

		fromPcm.forEach((value, i) => expect(value).toBeCloseTo(fromFloats[i], 3));
	});

	it('mixes the channels of a buffer', () => {
		const left = Float32Array.from([1, 0]);
		const right = Float32Array.from([0, 0.5]);
		const buffer = {
			getChannelData: (c: number) => (c === 0 ? left : right),
			length: 2,
			numberOfChannels: 2
		} as unknown as AudioBuffer;

		expect(computePeaks(buffer, 2)).toEqual([1, 0.5]);
	});

	it('keeps the count for silence, empty and short clips', () => {
		expect(computePeaks(new Float32Array(100), 8)).toEqual(new Array(8).fill(0));
		expect(computePeaks(new Float32Array(0), 8)).toEqual(new Array(8).fill(0));
		expect(computePeaks(Float32Array.from([0.5, -1]), 4)).toEqual([0.5, 0.5, 1, 1]);
	});
});

describe('levels', () => {
	it('maps loudness in decibels from the floor to full scale', () => {
		expect(rmsToLevel(0)).toBe(0);
		expect(rmsToLevel(1)).toBe(1);
		expect(rmsToLevel(2)).toBe(1);
		expect(rmsToLevel(0.001)).toBe(0); // -60 dBFS
		expect(rmsToLevel(0.0316)).toBeCloseTo(0.5, 2); // -30 dBFS
	});

	it('keeps a fixed-length history, newest last', () => {
		const history = new LevelHistory(3);

		history.push(0.1);
		history.push(0.2);

		const levels = history.push(0.3);

		expect(levels).toEqual([0.1, 0.2, 0.3]);
		expect(history.push(0.4)).toEqual([0.2, 0.3, 0.4]);
		// listeners get copies they can keep
		expect(levels).toEqual([0.1, 0.2, 0.3]);
	});
});
