/**
 * Voice engine defaults. Kept next to the engine (not in `#lib/constants/index.ts`) so the
 * folder can move into its own package without pulling the app's constants along.
 */

/** Rate of the PCM chunks and of the recorded WAV, what speech encoders expect. */
export const VOICE_SAMPLE_RATE = 16_000;

/** Length of the PCM chunks handed to the stream. */
export const VOICE_CHUNK_MS = 100;

/** Number of values in the live level history, newest last. */
export const VOICE_LEVELS_LENGTH = 48;

/** Time between two values of the live level history. */
export const VOICE_LEVELS_INTERVAL_MS = 50;

/** Level floor in dBFS: quieter input reads as 0. */
export const VOICE_LEVELS_FLOOR_DB = -60;

/** Analyser window, about 20 ms at 48 kHz. */
export const VOICE_ANALYSER_FFT_SIZE = 1024;

/** Number of peaks of a clip's static waveform. */
export const VOICE_PEAKS_COUNT = 64;

/**
 * Microphone constraints: the raw signal. Echo cancellation makes the system treat the
 * recording as a call, which lowers or interrupts what else is playing (Apple's voice
 * processing, Windows' communications ducking); nothing plays while we record, so it
 * has nothing to cancel. Auto gain fades the input in, losing the first words, and
 * speech models take unprocessed audio well.
 */
export const VOICE_MICROPHONE_CONSTRAINTS: MediaTrackConstraints = {
	autoGainControl: false,
	echoCancellation: false,
	noiseSuppression: false
};

/** How long a started capture waits for its first audio before it counts as recording anyway. */
export const VOICE_FIRST_AUDIO_TIMEOUT_MS = 1500;

/** Name of the PCM tap worklet processor. */
export const VOICE_WORKLET_PROCESSOR = 'llama-voice-pcm-tap';

/** Frames the worklet gathers before it posts them, about 43 ms at 48 kHz. */
export const VOICE_WORKLET_BATCH_FRAMES = 2048;

/** Buffer size of the ScriptProcessor fallback. */
export const VOICE_SCRIPT_PROCESSOR_FRAMES = 2048;

/** Bytes per String.fromCharCode call when encoding base64, below the argument limit. */
export const VOICE_BASE64_SLICE = 0x8000;

/** Server routes, relative so a server under a subpath still works. */
export const VOICE_API_PATHS = {
	REALTIME: './v1/realtime',
	TRANSCRIPTIONS: './v1/audio/transcriptions'
} as const;

/** Query parameters of the realtime socket. */
export const VOICE_REALTIME_QUERY = {
	AUTOLOAD: 'autoload',
	INTENT: 'intent',
	INTENT_TRANSCRIPTION: 'transcription',
	MODEL: 'model'
} as const;

/** Realtime events this client sends and reads, see tools/server/README.md. */
export const VOICE_REALTIME_EVENTS = {
	APPEND: 'input_audio_buffer.append',
	COMMIT: 'input_audio_buffer.commit',
	COMPLETED: 'conversation.item.input_audio_transcription.completed',
	DELTA: 'conversation.item.input_audio_transcription.delta',
	ERROR: 'error',
	SESSION_CREATED: 'session.created',
	SESSION_UPDATE: 'session.update'
} as const;

/** The session type that transcribes; a conversation session is switched to it. */
export const VOICE_REALTIME_SESSION_TRANSCRIPTION = 'transcription';

/** What the server says when the model has no streaming audio encoder. */
export const VOICE_REALTIME_UNSUPPORTED_PATTERN = /does not support realtime transcription/i;

/** Deadlines of a realtime session. */
export const VOICE_REALTIME_TIMEOUTS = {
	/** Socket open until session.created; a busy slot can hold it back */
	CONNECT_MS: 10_000,
	/** Final commit until the last completed event */
	FINISH_MS: 20_000
} as const;
