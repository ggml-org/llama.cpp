import { readGgufMetadata } from '$lib/utils/gguf-metadata';
import { describe, expect, it } from 'vitest';

type Entry =
	| [key: string, type: 'bool', value: boolean]
	| [key: string, type: 'u32', value: number]
	| [key: string, type: 'string', value: string]
	| [key: string, type: 'string[]', value: string[]]
	| [key: string, type: 'f32[]', value: number[]];

/** A GGUF header with the given metadata and no tensors. */
function gguf(entries: Entry[]): ArrayBuffer {
	const bytes: number[] = [];
	const u32 = (n: number) => bytes.push(...new Uint8Array(new Uint32Array([n]).buffer));
	const u64 = (n: number) => bytes.push(...new Uint8Array(new BigUint64Array([BigInt(n)]).buffer));
	const str = (s: string) => {
		const encoded = new TextEncoder().encode(s);

		u64(encoded.length);
		bytes.push(...encoded);
	};

	u32(0x46554747);
	u32(3);
	u64(0);
	u64(entries.length);

	for (const [key, type, value] of entries) {
		str(key);

		if (type === 'bool') {
			u32(7);
			bytes.push(value ? 1 : 0);
		} else if (type === 'u32') {
			u32(4);
			u32(value);
		} else if (type === 'string') {
			u32(8);
			str(value);
		} else if (type === 'string[]') {
			u32(9);
			u32(8);
			u64(value.length);
			value.forEach(str);
		} else {
			u32(9);
			u32(6);
			u64(value.length);
			value.forEach((n) => bytes.push(...new Uint8Array(new Float32Array([n]).buffer)));
		}
	}

	return new Uint8Array(bytes).buffer;
}

const MMPROJ: Entry[] = [
	['general.architecture', 'string', 'clip'],
	['clip.projector_type', 'string', 'gemma4'],
	['clip.vision.image_mean', 'f32[]', [0.5, 0.5, 0.5]],
	['tokenizer.ggml.tokens', 'string[]', ['<a>', '<b>']],
	['clip.has_vision_encoder', 'bool', true],
	['clip.has_audio_encoder', 'bool', true],
	['clip.audio.num_mel_bins', 'u32', 128]
];

describe('readGgufMetadata', () => {
	it('reads scalars and strings, skipping arrays', () => {
		const { complete, metadata } = readGgufMetadata(gguf(MMPROJ));

		expect(complete).toBe(true);

		expect(metadata.get('general.architecture')).toBe('clip');
		expect(metadata.get('clip.has_vision_encoder')).toBe(true);
		expect(metadata.get('clip.has_audio_encoder')).toBe(true);
		expect(metadata.get('clip.audio.num_mel_bins')).toBe(128);
		expect(metadata.has('clip.vision.image_mean')).toBe(false);
	});

	it('returns what it read before a buffer cut short', () => {
		const full = gguf(MMPROJ);
		const { complete, metadata } = readGgufMetadata(full.slice(0, full.byteLength - 20));

		expect(complete).toBe(false);

		expect(metadata.get('clip.has_vision_encoder')).toBe(true);
		expect(metadata.has('clip.audio.num_mel_bins')).toBe(false);
	});

	it('returns nothing for a file that is not GGUF', () => {
		expect(
			readGgufMetadata(new TextEncoder().encode('not a gguf file at all...').buffer)
		).toStrictEqual({ complete: false, metadata: new Map() });
		expect(readGgufMetadata(new ArrayBuffer(0))).toStrictEqual({
			complete: false,
			metadata: new Map()
		});
	});
});
