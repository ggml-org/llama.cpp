import { loadPcmTapModule } from './pcm-tap';

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

function createContext(): AudioContext {
	try {
		// Chrome renders without an output device; other browsers reject the option
		return new AudioContext({ sinkId: { type: 'none' } } as AudioContextOptions);
	} catch {
		return new AudioContext();
	}
}

/** The shared context, made on first use, with the PCM tap's worklet loading. */
export function captureContext(): { context: AudioContext; ready: Promise<void> } {
	if (!shared || shared.state === 'closed') {
		shared = createContext();
		modulesReady = loadPcmTapModule(shared).catch((error) => {
			// the tap falls back to a ScriptProcessor
			console.warn('AudioWorklet unavailable, using ScriptProcessor:', error);
		});
	}

	return { context: shared, ready: modulesReady ?? Promise.resolve() };
}

/**
 * Makes the context and loads the worklet ahead of a recording, e.g. when the chat
 * form gets focus. It asks for no microphone, so the browser shows no recording
 * indicator; outside a user gesture the context waits suspended.
 */
export function warmUpVoiceCapture(): void {
	if (typeof AudioContext === 'undefined') return;

	try {
		void captureContext().ready;
	} catch {
		// no audio here: the recording reports it when it starts
	}
}

/** Lets the context idle between recordings, so it renders nothing. */
export function idleCaptureContext(context: AudioContext): void {
	if (context !== shared) {
		if (context.state !== 'closed') context.close().catch(() => {});

		return;
	}

	if (context.state === 'running') context.suspend().catch(() => {});
}
