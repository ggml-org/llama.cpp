import { VOICE_AUDIO_CONTEXT_STATE, VOICE_AUDIO_SINK_NONE } from './constants.js';
import { loadPcmTapModule } from './pcm-tap.js';

/**
 * The page's one AudioContext for voice capture. Making a context and loading the
 * capture worklet take a noticeable moment, so this is done once (ideally ahead of
 * the first recording, see warmUpVoiceCapture) and the context is suspended between
 * recordings instead of closed.
 *
 * Where the browser allows it the context has no output device (`sinkId: none`):
 * capture never opens the speakers, so it cannot click or change their format while
 * something else plays.
 */

let shared: AudioContext | null = null;
let modulesReady: Promise<void> | null = null;
// captures using the shared context: it idles once the last one lets go, not before, as a
// capture still letting go may overlap the next one's start
let holders = 0;

function createContext(): AudioContext {
	try {
		// Chrome renders without an output device; other browsers reject the option
		return new AudioContext({
			sinkId: VOICE_AUDIO_SINK_NONE
		} as AudioContextOptions);
	} catch {
		return new AudioContext();
	}
}

function sharedContext(): { context: AudioContext; ready: Promise<void> } {
	if (!shared || shared.state === VOICE_AUDIO_CONTEXT_STATE.CLOSED) {
		shared = createContext();
		holders = 0;
		modulesReady = loadPcmTapModule(shared).catch((error) => {
			// the tap falls back to a ScriptProcessor
			console.warn('AudioWorklet unavailable, using ScriptProcessor:', error);
		});
	}

	return { context: shared, ready: modulesReady ?? Promise.resolve() };
}

/**
 * The shared context for a capture, made on first use, with the PCM tap's worklet loading.
 * The capture hands it back with idleCaptureContext.
 */
export function captureContext(): {
	context: AudioContext;
	ready: Promise<void>;
} {
	const handle = sharedContext();

	holders++;

	return handle;
}

/**
 * Makes the context and loads the worklet ahead of a recording, e.g. when the chat
 * form gets focus. It asks for no microphone, so the browser shows no recording
 * indicator; outside a user gesture the context waits suspended.
 */
export function warmUpVoiceCapture(): void {
	if (typeof AudioContext === 'undefined') return;

	try {
		void sharedContext().ready;
	} catch {
		// no audio here: the recording reports it when it starts
	}
}

/** Hands back a capture's context: it idles between recordings, so it renders nothing. */
export function idleCaptureContext(context: AudioContext): void {
	if (context !== shared) {
		if (context.state !== VOICE_AUDIO_CONTEXT_STATE.CLOSED) context.close().catch(() => {});

		return;
	}

	holders = Math.max(0, holders - 1);

	if (holders === 0 && context.state === VOICE_AUDIO_CONTEXT_STATE.RUNNING)
		context.suspend().catch(() => {});
}
