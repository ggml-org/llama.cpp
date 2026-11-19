// Guards opening the microphone ahead of a recording: only when the page may already use
// it, handed over to a recording with the same options, and let go otherwise.

import { VOICE_MICROPHONE_CONSTRAINTS } from '../src/constants.js';
import { microphoneRequest } from '../src/microphone.js';
import { MicrophoneWarmup } from '../src/warm-microphone.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

interface FakeTrack {
	readyState: 'live' | 'ended';
	stop: () => void;
}

function fakeStream(): MediaStream & { track: FakeTrack } {
	const track: FakeTrack = {
		readyState: 'live',
		stop() {
			track.readyState = 'ended';
		}
	};

	return {
		getAudioTracks: () => [track],
		getTracks: () => [track],
		track
	} as unknown as MediaStream & { track: FakeTrack };
}

let permission: PermissionState;
let getUserMedia: ReturnType<typeof vi.fn>;

beforeEach(() => {
	vi.useFakeTimers();
	permission = 'granted';
	getUserMedia = vi.fn(async () => fakeStream());
	vi.stubGlobal('navigator', {
		mediaDevices: { getUserMedia },
		permissions: { query: async () => ({ state: permission }) }
	});
});

afterEach(() => {
	vi.useRealTimers();
	vi.unstubAllGlobals();
});

describe('microphoneRequest', () => {
	it('asks for the raw signal, the caller’s processing on top, from the picked device', () => {
		expect(microphoneRequest({})).toEqual({ audio: VOICE_MICROPHONE_CONSTRAINTS });
		expect(microphoneRequest({ constraints: { echoCancellation: true }, deviceId: 'usb' })).toEqual(
			{
				audio: { ...VOICE_MICROPHONE_CONSTRAINTS, deviceId: 'usb', echoCancellation: true }
			}
		);
	});
});

describe('MicrophoneWarmup', () => {
	it('hands the open microphone to a recording with the same options', async () => {
		const warmup = new MicrophoneWarmup();

		warmup.open({ deviceId: 'usb' });

		const stream = (await warmup.take({ deviceId: 'usb' })) as MediaStream & { track: FakeTrack };

		expect(getUserMedia).toHaveBeenCalledOnce();
		expect(stream.track.readyState).toBe('live');
		// handed over: nothing is left to take
		expect(await warmup.take({ deviceId: 'usb' })).toBeNull();
	});

	it('opens it once for the same options, however often the pointer comes back', async () => {
		const warmup = new MicrophoneWarmup();

		warmup.open();
		warmup.release();
		warmup.open();

		expect(await warmup.take()).not.toBeNull();
		expect(getUserMedia).toHaveBeenCalledOnce();
	});

	it('never asks for access: without it, nothing opens', async () => {
		permission = 'prompt';

		const warmup = new MicrophoneWarmup();

		warmup.open();

		expect(await warmup.take()).toBeNull();
		expect(getUserMedia).not.toHaveBeenCalled();
	});

	it('lets go of a microphone opened for other options', async () => {
		const warmup = new MicrophoneWarmup();
		const opened = fakeStream();

		getUserMedia.mockResolvedValueOnce(opened);
		warmup.open({ deviceId: 'usb' });

		expect(await warmup.take({ deviceId: 'built-in' })).toBeNull();
		expect(opened.track.readyState).toBe('ended');
	});

	it('lets go a moment after release, unless a recording takes it first', async () => {
		const warmup = new MicrophoneWarmup();
		const opened = fakeStream();

		getUserMedia.mockResolvedValueOnce(opened);
		warmup.open();
		await Promise.resolve();
		warmup.release(1000);

		await vi.advanceTimersByTimeAsync(999);
		expect(opened.track.readyState).toBe('live');

		await vi.advanceTimersByTimeAsync(1);
		expect(opened.track.readyState).toBe('ended');
		expect(await warmup.take()).toBeNull();
	});
});
