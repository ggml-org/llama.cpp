import { VOICE_BASE64_SLICE } from './constants.js';

/** Float samples (-1..1) to PCM16, clipped. */
export function floatToPcm16(samples: Float32Array): Int16Array {
	const out = new Int16Array(samples.length);

	for (let i = 0; i < samples.length; i++) {
		const s = Math.max(-1, Math.min(1, samples[i]));

		out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
	}

	return out;
}

/** PCM16 samples back to floats (-1..1). */
export function pcm16ToFloat(samples: Int16Array): Float32Array {
	const out = new Float32Array(samples.length);

	for (let i = 0; i < samples.length; i++) {
		out[i] = samples[i] / 0x8000;
	}

	return out;
}

/** Joins PCM16 chunks into one array. */
export function concatPcm16(chunks: readonly Int16Array[]): Int16Array {
	let length = 0;

	for (const chunk of chunks) length += chunk.length;

	const out = new Int16Array(length);

	let offset = 0;

	for (const chunk of chunks) {
		out.set(chunk, offset);
		offset += chunk.length;
	}

	return out;
}

/** Bytes to base64, in slices so long inputs do not overflow the call stack. */
export function bytesToBase64(bytes: Uint8Array): string {
	let binary = '';

	for (let i = 0; i < bytes.length; i += VOICE_BASE64_SLICE) {
		binary += String.fromCharCode(...bytes.subarray(i, i + VOICE_BASE64_SLICE));
	}

	return btoa(binary);
}

/** Base64 to bytes. */
export function base64ToBytes(base64: string): Uint8Array<ArrayBuffer> {
	const binary = atob(base64);
	const bytes = new Uint8Array(binary.length);

	for (let i = 0; i < binary.length; i++) {
		bytes[i] = binary.charCodeAt(i);
	}

	return bytes;
}

/** PCM16 samples as base64, little-endian as the realtime API expects. */
export function pcm16ToBase64(samples: Int16Array): string {
	// typed arrays use the platform byte order, which is little-endian on every browser target
	return bytesToBase64(new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength));
}

/** Streaming resampler: feed blocks of any size, get the samples ready so far. */
export interface Resampler {
	process(input: Float32Array): Float32Array;
}

/**
 * Streaming mono resampler. Going down, each output sample is a triangle-weighted mean
 * of the input around it, two output periods wide: a cheap low-pass that keeps most
 * aliasing out of speech. Going up, it interpolates linearly. Same rates pass through.
 */
export function createResampler(fromRate: number, toRate: number): Resampler {
	if (fromRate === toRate) {
		return { process: (input) => input };
	}

	const ratio = fromRate / toRate;
	// input reach on each side of an output sample
	const reach = Math.max(1, ratio);

	// input samples kept for the next block, and the position of the next output in them
	let pending = new Float32Array(0);
	let position = 0;

	return {
		process(input: Float32Array): Float32Array {
			const buffer = new Float32Array(pending.length + input.length);

			buffer.set(pending);
			buffer.set(input, pending.length);

			const out: number[] = [];

			if (ratio > 1) {
				while (position + reach < buffer.length) {
					let sum = 0;
					let weight = 0;

					for (let i = Math.max(0, Math.ceil(position - reach)); i <= position + reach; i++) {
						const w = 1 - Math.abs(i - position) / reach;

						sum += buffer[i] * w;
						weight += w;
					}

					out.push(weight > 0 ? sum / weight : 0);
					position += ratio;
				}
			} else {
				while (position + 1 < buffer.length) {
					const i = Math.floor(position);
					const t = position - i;

					out.push(buffer[i] * (1 - t) + buffer[i + 1] * t);
					position += ratio;
				}
			}

			// keep what the next output still reaches back to
			const keep = Math.max(0, Math.floor(position - (ratio > 1 ? reach : 0)));

			pending = buffer.slice(keep);
			position -= keep;

			return Float32Array.from(out);
		}
	};
}
