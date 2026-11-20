import type { RealtimeUnsupportedReason } from './enums.js';

/** A realtime session failed; the caller can transcribe the clip over HTTP instead. */
export class RealtimeError extends Error {
	constructor(message: string, options?: { cause?: unknown }) {
		super(message, options);
		this.name = 'RealtimeError';
	}
}

/**
 * Realtime transcription is not available: the model has no streaming audio encoder
 * (`model`), or the socket could not be opened (`connection`), as with a server that
 * checks an API key, which a browser websocket cannot send.
 */
export class RealtimeUnsupportedError extends RealtimeError {
	readonly reason: RealtimeUnsupportedReason;

	constructor(message: string, reason: RealtimeUnsupportedReason, options?: { cause?: unknown }) {
		super(message, options);
		this.name = 'RealtimeUnsupportedError';
		this.reason = reason;
	}
}

/** The transcription endpoint answered with an error; `message` is the server's own. */
export class VoiceTranscriptionError extends Error {
	readonly status: number;

	constructor(message: string, status: number) {
		super(message);
		this.name = 'VoiceTranscriptionError';
		this.status = status;
	}
}

/** The microphone did not start; `message` says why in words for the user. */
export class VoiceCaptureError extends Error {
	constructor(message: string, options?: { cause?: unknown }) {
		super(message, options);
		this.name = 'VoiceCaptureError';
	}
}
