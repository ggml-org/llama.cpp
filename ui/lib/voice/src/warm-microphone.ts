import { VOICE_TRACK_LIVE, VOICE_WARMUP_RELEASE_MS } from './constants.js';
import { microphoneAllowed, microphoneRequest, stopStream } from './microphone.js';
import type { VoiceCaptureOptions } from './types.js';

type MicrophoneOptions = Pick<VoiceCaptureOptions, 'constraints' | 'deviceId'>;

interface Warm {
	key: string;
	stream: Promise<MediaStream | null>;
}

/**
 * Opens the microphone a moment before a recording starts, e.g. while the pointer is on
 * the record button, so the device has settled (its level, the system's input gain) by
 * the time the first words come, and the recording starts at once. A recording then
 * takes the open stream (take) instead of asking for one.
 *
 * It only opens a microphone the page may already use: it never makes the browser ask,
 * as nothing was pressed yet. The browser shows its recording indicator while it is open,
 * so it is let go shortly after the pointer leaves (release).
 */
export class MicrophoneWarmup {
	private warm: Warm | null = null;
	private timer: ReturnType<typeof setTimeout> | undefined;

	/** Opens the microphone for these options, keeping one that is open for the same already. */
	open(options: MicrophoneOptions = {}): void {
		clearTimeout(this.timer);

		const request = microphoneRequest(options);
		const key = JSON.stringify(request);

		if (this.warm?.key === key) return;

		this.drop();

		const stream = (async () => {
			if (!navigator.mediaDevices?.getUserMedia || !(await microphoneAllowed())) return null;

			try {
				return await navigator.mediaDevices.getUserMedia(request);
			} catch {
				// the recording asks again, and reports what is wrong
				return null;
			}
		})();

		this.warm = { key, stream };
	}

	/**
	 * Hands the open stream over for a recording with these options, which then owns it;
	 * null when none is open for them (it is let go), so the recording asks for its own.
	 */
	async take(options: MicrophoneOptions = {}): Promise<MediaStream | null> {
		clearTimeout(this.timer);

		const warm = this.warm;

		this.warm = null;

		if (!warm) return null;

		const stream = await warm.stream;
		const live =
			stream?.getAudioTracks().some((track) => track.readyState === VOICE_TRACK_LIVE) ?? false;

		if (warm.key !== JSON.stringify(microphoneRequest(options)) || !live) {
			stopStream(stream);

			return null;
		}

		return stream;
	}

	/** Lets the microphone go after `delayMs`, unless a recording takes it or it is opened again. */
	release(delayMs: number = VOICE_WARMUP_RELEASE_MS): void {
		clearTimeout(this.timer);
		this.timer = setTimeout(() => this.drop(), delayMs);
	}

	/** Lets the microphone go now. */
	drop(): void {
		clearTimeout(this.timer);

		const warm = this.warm;

		this.warm = null;
		void warm?.stream.then(stopStream);
	}
}
