import { VOICE_DOM_ERROR, VOICE_MICROPHONE_MESSAGE } from './constants.js';

/**
 * What the browser offers for voice capture, and what to tell the user when it does not.
 * Plain checks and messages, so the engine can say why a recording cannot start without
 * the app's constants.
 */

/** Whether this browser has what voice capture uses: getUserMedia and an AudioContext. */
export function isVoiceCaptureSupported(): boolean {
	return !!(
		typeof navigator !== 'undefined' &&
		navigator.mediaDevices &&
		typeof navigator.mediaDevices.getUserMedia === 'function' &&
		typeof window !== 'undefined' &&
		typeof window.AudioContext === 'function'
	);
}

/**
 * Why this page can't record at all, or null when it can: a reason no tap will fix. Browsers
 * only expose the microphone to a secure context, HTTPS or localhost, so a llama-server
 * opened over plain HTTP on the local network or over Tailscale can't record.
 */
export function microphoneUnavailableReason(): string | null {
	if (typeof window === 'undefined') return null;

	if (!window.isSecureContext) return VOICE_MICROPHONE_MESSAGE.insecure;

	if (!isVoiceCaptureSupported()) return VOICE_MICROPHONE_MESSAGE.unsupported;

	return null;
}

/** Why the microphone would not start, from the error the browser raised. */
export function describeMicrophoneError(error: unknown): string {
	switch (error instanceof DOMException ? error.name : '') {
		case VOICE_DOM_ERROR.NOT_ALLOWED:
		case VOICE_DOM_ERROR.SECURITY:
			return VOICE_MICROPHONE_MESSAGE.denied;
		case VOICE_DOM_ERROR.NOT_FOUND:
		case VOICE_DOM_ERROR.OVERCONSTRAINED:
			return VOICE_MICROPHONE_MESSAGE.notFound;
		case VOICE_DOM_ERROR.ABORT:
		case VOICE_DOM_ERROR.NOT_READABLE:
			return VOICE_MICROPHONE_MESSAGE.busy;
		default:
			return error instanceof Error && error.message
				? `${VOICE_MICROPHONE_MESSAGE.failed} ${error.message}`
				: VOICE_MICROPHONE_MESSAGE.failed;
	}
}
