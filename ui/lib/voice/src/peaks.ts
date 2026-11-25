import {
	VOICE_DATA_URL_HEADER_PATTERN,
	VOICE_DECODE_SAMPLE_RATE,
	VOICE_PEAKS_COUNT
} from './constants.js';
import { base64ToBytes } from './pcm.js';

/** Mono samples of a buffer: the mean of its channels. */
export function mixToMono(buffer: AudioBuffer): Float32Array {
	if (buffer.numberOfChannels === 1) return buffer.getChannelData(0);

	const out = new Float32Array(buffer.length);

	for (let c = 0; c < buffer.numberOfChannels; c++) {
		const channel = buffer.getChannelData(c);

		for (let i = 0; i < out.length; i++) out[i] += channel[i] / buffer.numberOfChannels;
	}

	return out;
}

/**
 * Peaks of a clip for a static waveform: the loudest sample of each of `count`
 * equal spans, scaled so the loudest span is 1. Silence gives all zeros.
 */
export function computePeaks(
	source: Float32Array | Int16Array | AudioBuffer,
	count = VOICE_PEAKS_COUNT
): number[] {
	const samples =
		source instanceof Float32Array || source instanceof Int16Array ? source : mixToMono(source);
	const scale = samples instanceof Int16Array ? 1 / 0x8000 : 1;
	const peaks = new Array<number>(count).fill(0);

	if (samples.length === 0 || count <= 0) return peaks;

	for (let p = 0; p < count; p++) {
		const start = Math.floor((p * samples.length) / count);
		// a clip shorter than `count` repeats samples rather than leaving gaps
		const end = Math.max(start + 1, Math.floor(((p + 1) * samples.length) / count));

		let max = 0;

		for (let i = start; i < end && i < samples.length; i++) {
			const value = Math.abs(samples[i] * scale);

			if (value > max) max = value;
		}

		peaks[p] = max;
	}

	const loudest = Math.max(...peaks);

	return loudest > 0 ? peaks.map((value) => value / loudest) : peaks;
}

/** Decodes an audio file (Blob, bytes, or base64 such as a stored attachment) to a buffer. */
export async function decodeAudio(source: Blob | ArrayBuffer | string): Promise<AudioBuffer> {
	const bytes =
		typeof source === 'string'
			? base64ToBytes(source.replace(VOICE_DATA_URL_HEADER_PATTERN, '')).buffer
			: source instanceof ArrayBuffer
				? source
				: await source.arrayBuffer();
	// an offline context decodes without a running audio device; the rate is a placeholder
	const context = new OfflineAudioContext(1, 1, VOICE_DECODE_SAMPLE_RATE);

	return context.decodeAudioData(bytes as ArrayBuffer);
}

/** Peaks of an encoded audio file, for stored clips that carry none. */
export async function computePeaksFromAudio(
	source: Blob | ArrayBuffer | string,
	count = VOICE_PEAKS_COUNT
): Promise<number[]> {
	return computePeaks(await decodeAudio(source), count);
}
