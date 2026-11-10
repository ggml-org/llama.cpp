import { HuggingFaceService } from '$lib/services/huggingface.service';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

/** A GGUF header with the given bool metadata and no tensors. */
function mmprojHeader(entries: Record<string, boolean>): ArrayBuffer {
	const bytes: number[] = [];
	const u32 = (n: number) => bytes.push(...new Uint8Array(new Uint32Array([n]).buffer));
	const u64 = (n: number) => bytes.push(...new Uint8Array(new BigUint64Array([BigInt(n)]).buffer));

	u32(0x46554747);
	u32(3);
	u64(0);
	u64(Object.keys(entries).length);

	for (const [key, value] of Object.entries(entries)) {
		const encoded = new TextEncoder().encode(key);

		u64(encoded.length);
		bytes.push(...encoded);
		u32(7);
		bytes.push(value ? 1 : 0);
	}

	return new Uint8Array(bytes).buffer;
}

function respond(body: ArrayBuffer, status: number) {
	const fetchMock = vi.fn(async () => new Response(body, { status }));

	vi.stubGlobal('fetch', fetchMock);

	return fetchMock;
}

describe('HuggingFaceService mmproj modalities', () => {
	beforeEach(() => {
		vi.spyOn(HuggingFaceService, 'isEnabled').mockReturnValue(true);
		vi.spyOn(console, 'error').mockImplementation(() => {});
	});

	afterEach(() => {
		vi.restoreAllMocks();
		vi.unstubAllGlobals();
	});

	it('takes audio from the pipeline tag and vision from the mmproj before its header is read', () => {
		expect(HuggingFaceService.getModalities('org/unread', 'text-to-speech', 'mmproj.gguf')).toEqual(
			{ audio: true, video: false, vision: true }
		);
		expect(HuggingFaceService.getModalities('org/plain', 'text-generation', undefined)).toEqual({
			audio: false,
			video: false,
			vision: false
		});
	});

	it('adds audio for an omni mmproj, with a ranged read of an encoded path', async () => {
		const fetchMock = respond(
			mmprojHeader({ 'clip.has_audio_encoder': true, 'clip.has_vision_encoder': true }),
			206
		);

		await HuggingFaceService.getMmprojEncoders('org/omni', 'sub dir/mmproj#1.gguf');

		expect(fetchMock).toHaveBeenCalledWith(
			'https://huggingface.co/org/omni/resolve/main/sub%20dir/mmproj%231.gguf',
			{ headers: { Range: 'bytes=0-65535' } }
		);
		expect(HuggingFaceService.getModalities('org/omni', 'any-to-any', 'mmproj.gguf')).toEqual({
			audio: true,
			video: false,
			vision: true
		});
	});

	it('drops vision for an audio-only mmproj', async () => {
		// the converter writes the vision key only when true
		respond(mmprojHeader({ 'clip.has_audio_encoder': true }), 206);

		await HuggingFaceService.getMmprojEncoders('org/audio-only', 'mmproj.gguf');

		expect(HuggingFaceService.getModalities('org/audio-only', undefined, 'mmproj.gguf')).toEqual({
			audio: true,
			video: false,
			vision: false
		});
	});

	it('keeps vision when a header cut short never said', async () => {
		const header = mmprojHeader({
			'clip.has_audio_encoder': true,
			'clip.has_vision_encoder': true
		});

		respond(header.slice(0, header.byteLength - 4), 206);

		expect(await HuggingFaceService.getMmprojEncoders('org/cut', 'mmproj.gguf')).toEqual({
			audio: true,
			vision: null
		});
		expect(HuggingFaceService.getModalities('org/cut', undefined, 'mmproj.gguf').vision).toBe(true);
	});

	it('keeps vision only for a vision mmproj', async () => {
		respond(mmprojHeader({ 'clip.has_vision_encoder': true }), 206);

		await HuggingFaceService.getMmprojEncoders('org/vision', 'mmproj.gguf');

		expect(HuggingFaceService.getModalities('org/vision', 'image-text-to-text', 'm.gguf')).toEqual({
			audio: false,
			video: false,
			vision: true
		});
	});

	it('keeps what the repo says for a gated mmproj, without reading it again', async () => {
		const fetchMock = respond(new ArrayBuffer(0), 401);

		expect(await HuggingFaceService.getMmprojEncoders('org/gated', 'mmproj.gguf')).toEqual({
			audio: false,
			vision: null
		});
		await HuggingFaceService.getMmprojEncoders('org/gated', 'mmproj.gguf');

		expect(fetchMock).toHaveBeenCalledTimes(1);
		expect(HuggingFaceService.getModalities('org/gated', undefined, 'mmproj.gguf')).toEqual({
			audio: false,
			video: false,
			vision: true
		});
	});

	it('rejects a full response that ignored the range, and retries on the next call', async () => {
		respond(mmprojHeader({ 'clip.has_audio_encoder': true }), 200);

		expect(await HuggingFaceService.getMmprojEncoders('org/no-range', 'mmproj.gguf')).toBeNull();
		expect(HuggingFaceService.getModalities('org/no-range', undefined, 'mmproj.gguf').audio).toBe(
			false
		);

		respond(mmprojHeader({ 'clip.has_audio_encoder': true }), 206);

		expect(await HuggingFaceService.getMmprojEncoders('org/no-range', 'mmproj.gguf')).toEqual({
			audio: true,
			vision: false
		});
	});
});
