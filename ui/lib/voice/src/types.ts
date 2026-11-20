import type { VoiceCapture } from './capture.js';

/** Voice engine types. */

/** A finished recording, ready to send and to draw. */
export interface VoiceClip {
	/** WAV, 16-bit PCM, mono */
	blob: Blob;
	durationMs: number;
	/** Normalised 0..1, fixed count, for the static waveform */
	peaks: number[];
	sampleRate: number;
}

/** Called with the live level history: 0..1, fixed length, newest last. */
export type VoiceLevelsListener = (levels: readonly number[]) => void;

/** Called with each PCM16 mono chunk at the capture rate. */
export type VoicePcmListener = (chunk: Int16Array) => void;

/** Called when the speech starts or stops being too quiet even at the highest gain. */
export type VoiceQuietListener = (quiet: boolean) => void;

export interface VoiceCaptureOptions {
	/** Rate of the PCM chunks and of the clip (default 16000) */
	sampleRate?: number;
	/** Length of a PCM chunk (default 100 ms) */
	chunkMs?: number;
	/** Values in the level history (default 48) */
	levelsLength?: number;
	/** Time between two level values (default 50 ms) */
	levelsIntervalMs?: number;
	/** Peaks of the clip (default 64) */
	peaksCount?: number;
	/** Extra constraints merged over the raw-signal ones (no echo cancellation, noise suppression or auto gain) */
	constraints?: MediaTrackConstraints;
	/** Microphone to record from, see resolveMicrophone(); left out, the system default */
	deviceId?: string;
	/** Use this stream instead of asking for the microphone; the capture stops its tracks too */
	stream?: MediaStream;
	/** Raise quiet speech (gain.ts): the streamed chunks as they go, the clip as a whole */
	boost?: boolean;
}

/**
 * How the engine reaches a server. The app wires its own URL resolution and
 * auth in, so the engine never imports app stores.
 */
export interface VoiceServerContext {
	/** Request target for an API path, e.g. `(path) => apiUrl(path, serverId)` */
	resolveUrl: (path: string) => string;
	/** HTTP headers such as the API key; a browser websocket cannot carry them */
	getHeaders?: () => Record<string, string>;
	/** Fetch to use, defaults to the global one */
	fetch?: typeof fetch;
	/** Base for relative URLs, defaults to `location.href` */
	baseHref?: string;
}

export interface TranscribeClipOptions {
	/** Model id, required by a router server */
	model?: string;
	/** ISO-639-1 language hint */
	language?: string;
	/** Replaces the model's own transcription prompt */
	prompt?: string;
	/** File name sent with the audio (default `audio.wav`) */
	filename?: string;
	signal?: AbortSignal;
}

export interface RealtimeTranscriberOptions {
	/** Model id, sent as `?model=` (the router routes on it) */
	model?: string;
	/** Rate of the PCM16 chunks given to `append` */
	sampleRate: number;
	/** ISO-639-1 language hint, used by a conversation model switched to transcription */
	language?: string;
	/** Sent as `?autoload=`; false keeps a router from loading the model */
	autoload?: boolean;
	connectTimeoutMs?: number;
	finishTimeoutMs?: number;
	/** New text of the current item, and the whole live text */
	onDelta?: (delta: string, text: string) => void;
	/** The whole transcript so far, once an item completes */
	onCompleted?: (text: string) => void;
	/** Errors after the session started; the session goes on unless finish() fails */
	onError?: (error: Error) => void;
}

export interface VoiceSessionOptions {
	/** Capture options, or a capture made by the caller */
	capture?: VoiceCaptureOptions | VoiceCapture;
	/** Try the realtime socket first (default true); off for servers behind a tunnel */
	realtime?: boolean;
	/** Where and with which model to transcribe; leave out to record only */
	transcribe?: {
		context: VoiceServerContext;
		model?: string;
		language?: string;
		autoload?: boolean;
	};
	onLevels?: VoiceLevelsListener;
	/** Whether the speech is too quiet even at the highest gain, when that changes */
	onQuiet?: VoiceQuietListener;
	/** Live text while speaking, only when the realtime socket streams it */
	onText?: (text: string) => void;
	/** Once known: true when text streams while speaking, false when it comes after recording */
	onRealtime?: (live: boolean) => void;
}

export interface VoiceSessionResult {
	clip: VoiceClip;
	/** Text the realtime socket gave while speaking, null when it streamed none */
	liveText: string | null;
	/**
	 * Resolves with the final transcript (realtime, else HTTP; a recording transcribed in pieces
	 * once more as a whole); null when not asked to transcribe
	 */
	transcript: Promise<string> | null;
}

export interface VoiceSession {
	readonly capture: VoiceCapture;
	/** Ends the recording and resolves with the clip and the transcript (or its promise) */
	stop(): Promise<VoiceSessionResult>;
	/** Drops the recording and closes everything */
	cancel(): void;
}
