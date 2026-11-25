// Guards the shared capture context: it idles only once the last capture using it lets
// go, so one still letting go cannot silence the next recording.

import { captureContext, idleCaptureContext } from '../src/audio-context.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

class FakeAudioContext {
	state: AudioContextState = 'running';
	suspend = vi.fn(async () => {
		this.state = 'suspended';
	});
	resume = vi.fn(async () => {});
	close = vi.fn(async () => {});
	audioWorklet = { addModule: async () => {} };
}

beforeEach(() => {
	vi.stubGlobal('AudioContext', FakeAudioContext);
});

afterEach(() => {
	vi.unstubAllGlobals();
});

describe('captureContext', () => {
	it('idles once the last capture lets go', () => {
		const first = captureContext().context;
		const second = captureContext().context;

		expect(second).toBe(first);

		idleCaptureContext(first);
		expect(first.suspend).not.toHaveBeenCalled();

		idleCaptureContext(second);
		expect(first.suspend).toHaveBeenCalledOnce();
	});
});
