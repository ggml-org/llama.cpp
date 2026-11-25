/**
 * Demo audio for the voice stories, made in the browser so no binary file sits in the repo: a WAV of a hummed tone
 * that rises and falls like syllables, with its peaks.
 */
export interface DemoClip {
	src: string;
	durationMs: number;
	peaks: number[];
}

const SAMPLE_RATE = 8000;
const PEAK_COUNT = 64;

/** Syllable-like loudness at time `t` (s): bursts of about 0.25 s with short gaps between words. */
function envelope(t: number, seed: number): number {
	const syllable = Math.max(Math.sin(t * Math.PI * 4 + seed), 0);
	const word = Math.sin(t * Math.PI * 0.9 + seed * 2) > -0.6 ? 1 : 0.08;
	const swell = 0.6 + 0.4 * Math.sin(t * 1.3 + seed);

	return syllable * word * swell;
}

function wavBlob(samples: Float32Array): Blob {
	const buffer = new ArrayBuffer(44 + samples.length * 2);
	const view = new DataView(buffer);
	const write = (offset: number, text: string) => {
		for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
	};

	write(0, 'RIFF');
	view.setUint32(4, 36 + samples.length * 2, true);
	write(8, 'WAVE');
	write(12, 'fmt ');
	view.setUint32(16, 16, true);
	view.setUint16(20, 1, true);
	view.setUint16(22, 1, true);
	view.setUint32(24, SAMPLE_RATE, true);
	view.setUint32(28, SAMPLE_RATE * 2, true);
	view.setUint16(32, 2, true);
	view.setUint16(34, 16, true);
	write(36, 'data');
	view.setUint32(40, samples.length * 2, true);

	for (let i = 0; i < samples.length; i++) {
		view.setInt16(44 + i * 2, Math.max(-1, Math.min(1, samples[i])) * 0x7fff, true);
	}

	return new Blob([buffer], { type: 'audio/wav' });
}

/** The loudest sample of each of `count` buckets, scaled so the loudest bucket is 1. */
export function peaksOf(samples: Float32Array, count = PEAK_COUNT): number[] {
	const size = Math.max(Math.floor(samples.length / count), 1);
	const peaks = Array.from({ length: count }, (_, bucket) => {
		let peak = 0;

		for (let i = bucket * size; i < Math.min((bucket + 1) * size, samples.length); i++) {
			peak = Math.max(peak, Math.abs(samples[i]));
		}

		return peak;
	});
	const loudest = Math.max(...peaks, 1e-6);

	return peaks.map((peak) => peak / loudest);
}

function demoSamples(seconds: number, seed: number): Float32Array {
	const samples = new Float32Array(Math.round(seconds * SAMPLE_RATE));

	for (let i = 0; i < samples.length; i++) {
		const t = i / SAMPLE_RATE;
		const pitch = 180 + 40 * Math.sin(t * 2 + seed);

		samples[i] = 0.5 * envelope(t, seed) * Math.sin(2 * Math.PI * pitch * t);
	}

	return samples;
}

/** The demo clip as a stored attachment holds it: base64 WAV, with its peaks and length. */
export async function demoStoredClip(
	seconds: number,
	seed = 1
): Promise<{ base64: string; durationMs: number; peaks: number[]; size: number }> {
	const samples = demoSamples(seconds, seed);
	const blob = wavBlob(samples);
	const bytes = new Uint8Array(await blob.arrayBuffer());

	let binary = '';

	for (let i = 0; i < bytes.length; i += 0x8000) {
		binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
	}

	return {
		base64: btoa(binary),
		durationMs: seconds * 1000,
		peaks: peaksOf(samples),
		size: blob.size
	};
}

const cache = new Map<string, DemoClip>();

/** A clip of `seconds`, the same for the same arguments. */
export function demoClip(seconds: number, seed = 1): DemoClip {
	const key = `${seconds}:${seed}`;
	const cached = cache.get(key);

	if (cached) return cached;

	const samples = demoSamples(seconds, seed);
	const clip = {
		durationMs: seconds * 1000,
		peaks: peaksOf(samples),
		src: URL.createObjectURL(wavBlob(samples))
	};

	cache.set(key, clip);

	return clip;
}

/** One live input level (0..1) at time `t` (s), as a microphone meter would report it while someone speaks. */
export function demoLevel(t: number, seed = 1): number {
	return Math.min(envelope(t, seed) * (0.75 + 0.25 * Math.random()), 1);
}
