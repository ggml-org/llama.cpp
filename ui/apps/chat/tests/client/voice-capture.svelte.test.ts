// Guards VoiceCapture in a real browser, fed by a synthesized microphone (an oscillator
// into a MediaStream, since the fake capture device flags are not available here): the
// levels move, the PCM worklet loads from its Blob URL and streams chunks, stop() returns
// a WAV clip with a plausible duration and peaks, and every path stops the tracks.

import {
	VOICE_MICROPHONE_MESSAGE,
	VoiceCapture,
	VoiceCaptureError,
	VoiceCaptureState
} from '@ggml-org/llama-voice';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

const sources: AudioContext[] = [];

/** A 440 Hz tone as a microphone stream. */
function toneStream(): MediaStream {
	const context = new AudioContext();
	const oscillator = context.createOscillator();
	const destination = context.createMediaStreamDestination();

	oscillator.frequency.value = 440;
	oscillator.connect(destination);
	oscillator.start();
	sources.push(context);

	return destination.stream;
}

// audio contexts only run after a user gesture, as the mic button gives in the app
beforeAll(async () => {
	const button = document.createElement('button');

	button.textContent = 'Record';
	document.body.append(button);
	await userEvent.click(button);
	button.remove();
});

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/** Waits for `done`, polling: a loaded test run can start the audio graph late. */
async function waitFor(done: () => boolean, timeoutMs = 5000) {
	const deadline = Date.now() + timeoutMs;

	while (!done()) {
		if (Date.now() > deadline) throw new Error('timed out waiting for audio');

		await wait(20);
	}
}

afterEach(async () => {
	vi.restoreAllMocks();
	vi.unstubAllGlobals();

	for (const context of sources.splice(0)) await context.close();
});

describe('VoiceCapture', () => {
	it('streams levels and PCM chunks, and stops with a WAV clip', async () => {
		const warn = vi.spyOn(console, 'warn');
		const stream = toneStream();
		const capture = new VoiceCapture({ levelsIntervalMs: 20, levelsLength: 16, stream });
		const levels: (readonly number[])[] = [];
		const chunks: Int16Array[] = [];

		capture.onLevels((value) => levels.push(value));
		capture.onPcm((chunk) => chunks.push(chunk));

		await capture.start();
		expect(capture.state).toBe(VoiceCaptureState.RECORDING);

		// at least 600 ms of tone recorded, however long the audio graph took to start
		await waitFor(() => chunks.length >= 6 && levels.length > 5);

		const clip = await capture.stop();

		// the worklet loaded from its Blob URL under COOP/COEP, as llama-server serves the UI, no fallback
		expect(crossOriginIsolated).toBe(true);
		expect(warn).not.toHaveBeenCalledWith(
			expect.stringContaining('AudioWorklet unavailable'),
			expect.anything()
		);

		expect(levels.length).toBeGreaterThan(5);
		expect(levels.at(-1)).toHaveLength(16);
		expect(Math.max(...levels.at(-1)!)).toBeGreaterThan(0.5);

		// 100 ms chunks at 16 kHz, the last one may be shorter
		expect(chunks.length).toBeGreaterThan(3);
		expect(chunks[0]).toHaveLength(1600);

		expect(clip.sampleRate).toBe(16_000);
		expect(clip.durationMs).toBeGreaterThan(500);
		expect(clip.durationMs).toBeLessThan(5000);
		expect(clip.peaks).toHaveLength(64);
		expect(Math.max(...clip.peaks)).toBe(1);
		// a steady tone: once it reaches the capture, no span of the clip is silent. The capture
		// starts with its first audio block, which can come before the tone stream's first sound
		const toneFrom = clip.peaks.findIndex((peak) => peak > 0.5);

		expect(toneFrom).toBeGreaterThanOrEqual(0);
		expect(clip.peaks.slice(toneFrom).every((peak) => peak > 0.5)).toBe(true);
		expect(clip.peaks.length - toneFrom).toBeGreaterThan(16);

		const header = new DataView(await clip.blob.slice(0, 44).arrayBuffer());
		const pcmBytes = chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0);

		expect(clip.blob.type).toBe('audio/wav');
		expect(header.getUint16(22, true)).toBe(1);
		expect(header.getUint32(24, true)).toBe(16_000);
		expect(header.getUint16(34, true)).toBe(16);
		expect(header.getUint32(40, true)).toBe(pcmBytes);

		// the clip decodes as audio
		const decoded = await new OfflineAudioContext(1, 1, 16_000).decodeAudioData(
			await clip.blob.arrayBuffer()
		);

		expect(decoded.duration * 1000).toBeCloseTo(clip.durationMs, 0);
		expect(stream.getTracks().every((track) => track.readyState === 'ended')).toBe(true);
		expect(capture.state).toBe(VoiceCaptureState.STOPPED);
	});

	it('cancel stops the tracks and drops the recording', async () => {
		const stream = toneStream();
		const capture = new VoiceCapture({ stream });

		await capture.start();
		await wait(200);
		capture.cancel();

		expect(stream.getTracks().every((track) => track.readyState === 'ended')).toBe(true);
		expect(capture.state).toBe(VoiceCaptureState.IDLE);
		await expect(capture.stop()).rejects.toThrow();
	});

	it('cancel during start releases the microphone once it arrives', async () => {
		const stream = toneStream();

		let grant!: (value: MediaStream) => void;

		vi.spyOn(navigator.mediaDevices, 'getUserMedia').mockImplementation(
			() => new Promise((resolve) => (grant = resolve))
		);

		const capture = new VoiceCapture();
		const started = capture.start().catch((e) => e);

		capture.cancel();
		grant(stream);

		expect(await started).toBeInstanceOf(DOMException);
		expect(stream.getTracks().every((track) => track.readyState === 'ended')).toBe(true);
	});

	it('a page hide tears the capture down', async () => {
		const stream = toneStream();
		const capture = new VoiceCapture({ stream });

		await capture.start();
		dispatchEvent(new PageTransitionEvent('pagehide'));

		expect(capture.state).toBe(VoiceCaptureState.IDLE);
		expect(stream.getTracks().every((track) => track.readyState === 'ended')).toBe(true);
	});

	it('asks for the raw microphone signal and names a refusal', async () => {
		const getUserMedia = vi
			.spyOn(navigator.mediaDevices, 'getUserMedia')
			.mockRejectedValue(new DOMException('denied', 'NotAllowedError'));
		const capture = new VoiceCapture();
		const error = await capture.start().catch((e) => e);

		expect(getUserMedia).toHaveBeenCalledWith({
			audio: { autoGainControl: false, echoCancellation: false, noiseSuppression: false }
		});
		expect(error).toBeInstanceOf(VoiceCaptureError);
		expect(error.message).toBe(VOICE_MICROPHONE_MESSAGE.denied);
		expect(capture.state).toBe(VoiceCaptureState.IDLE);
	});
});
