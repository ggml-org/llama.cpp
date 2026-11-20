/** Voice engine defaults. */

/** Type of the recorded clips: 16-bit PCM WAV, what speech encoders and `input_audio` take. */
export const VOICE_WAV_MIME = 'audio/wav';

/** Why the microphone can't record, as the app's microphone button and toasts tell the user. */
export const VOICE_MICROPHONE_MESSAGE = {
	busy: 'The microphone is in use by another app.',
	denied: "Microphone access was denied. Allow it in the browser's site settings to record.",
	failed: "The microphone couldn't start.",
	insecure:
		'The microphone only works over HTTPS or on localhost. Open this page over HTTPS to record.',
	notFound: 'No microphone was found.',
	unsupported: "This browser can't record audio."
} as const;

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

/**
 * How live text is cut into pieces for a model that transcribes per commit: at a pause
 * after speech, or at the next quiet moment once a piece is this long. Short pieces keep
 * the text close behind the speaker; the cut always falls between words.
 */
export const VOICE_PIECE = {
	MAX_MS: 3000,
	MIN_SPEECH_MS: 250,
	PAUSE_MS: 350
} as const;

/**
 * Longest recording that is transcribed once more as a whole when it ends. Pieces cut at
 * the speaker's pauses come back as sentences of their own ("I went. To the store."); one
 * pass over the whole clip punctuates by meaning. Past this, the pieces' text is kept.
 */
export const VOICE_RETRANSCRIBE_MAX_MS = 120_000;

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

/**
 * How Qwen3-ASR opens its answer: the language it heard, then the tag its text follows,
 * e.g. `language French<asr_text>`.
 */
export const VOICE_ASR_TEXT_PREFIX_PATTERN = /^\s*(?:language\s+[^<\n]{0,40})?<asr_text>/i;

/** The tag around Qwen3-ASR's text, wherever it stands. */
export const VOICE_ASR_TEXT_TAG_PATTERN = /<\/?asr_text>/gi;

/**
 * How VibeVoice-ASR opens its answer: a JSON array of timed segments, or the start of
 * one while it streams.
 */
export const VOICE_SEGMENTS_START_PATTERN = /^\s*\[\s*(?:\{\s*(?:"|$)|$)/;

/** The text of each timed segment, its closing quote missing while the last one streams. */
export const VOICE_SEGMENTS_CONTENT_PATTERN = /"content"\s*:\s*"((?:[^"\\]|\\.)*)/gi;

/** An escape cut short at the end of a JSON string that is still streaming. */
export const VOICE_UNFINISHED_ESCAPE_PATTERN = /\\(?:u[0-9a-f]{0,3})?$/i;

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

/**
 * Speech detection (speech-detector.ts): what tells speech from the room, for the pause
 * detector, the speech gain and any feature listening for speech. Levels in dBFS.
 */
export const VOICE_SPEECH = {
	/** How far above the noise floor a level is speech */
	ABOVE_FLOOR_DB: 9,
	/** How fast the floor creeps back up through louder levels */
	FLOOR_RISE_DB_PER_S: 2,
	/** Quieter than this is never speech, whatever the room */
	MIN_DB: -54,
	/** Up to this far above the floor a level is quiet: the room, or a breath between words */
	QUIET_ABOVE_FLOOR_DB: 5
} as const;

/**
 * Speech gain (gain.ts): how a quiet microphone is brought up for transcription. Levels
 * are RMS of samples in -1..1.
 */
export const VOICE_GAIN = {
	/** Frames a finished recording is measured in */
	FRAME_MS: 20,
	/** Where in a finished recording's frames, quietest first, its noise floor is read */
	FLOOR_QUANTILE: 0.1,
	/** The soft limiter's knee: louder samples bend towards full scale */
	KNEE: 0.8,
	/** How long speech must stay too quiet before the recording says so, as the microphone may settle */
	QUIET_HOLD_MS: 2000,
	/** The most it is raised: 8x, +18 dB, so a far-off voice does not bring the room up with it */
	MAX: 8,
	/** How much of the way to its target the streaming gain rises per speech block */
	RISE: 0.3,
	/** How much of each new speech block the streaming speech level takes in */
	SPEECH_FOLLOW: 0.2,
	/** Speech is raised towards this level, about -20 dBFS */
	TARGET_RMS: 0.1,
	/** Speech below this even at the highest gain (about -32 dBFS) reads as too quiet */
	TOO_QUIET_RMS: 0.025
} as const;

/**
 * How long a microphone opened ahead of a recording (MicrophoneWarmup) stays open once
 * the pointer leaves the button, in case it comes back to press it.
 */
export const VOICE_WARMUP_RELEASE_MS = 1500;

/** Messages of the errors a recording, a transcription or a realtime session ends with. */
export const VOICE_ERROR_MESSAGE = {
	captureAlreadyStarted: 'Voice capture already started',
	captureCancelled: 'Voice capture cancelled',
	captureNotStarted: 'No voice capture to stop',
	noTranscript: 'The server returned no transcript.',
	realtimeClosed: 'The realtime session was closed.',
	realtimeClosedEarly: 'The realtime socket closed before the transcript completed.',
	realtimeConnectTimeout: 'The realtime session did not start in time.',
	realtimeFailed: 'Realtime transcription failed.',
	realtimeFinishTimeout: 'The transcript did not complete in time.',
	realtimeNotOpen: 'The realtime session is not open.',
	realtimeRefused: (code: number) =>
		`The realtime socket closed before the session started (code ${code}).`,
	realtimeSocketFailed: 'The realtime socket could not be opened.',
	recorderBusy: 'Voice recorder already running',
	recorderIdle: 'No voice recording to stop',
	transcriptionFailed: 'Transcription failed',
	workletUnavailable: 'AudioWorklet unavailable'
} as const;

/** The DOMException names the engine raises or tells apart. */
export const VOICE_DOM_ERROR = {
	ABORT: 'AbortError',
	NOT_ALLOWED: 'NotAllowedError',
	NOT_FOUND: 'NotFoundError',
	NOT_READABLE: 'NotReadableError',
	OVERCONSTRAINED: 'OverconstrainedError',
	SECURITY: 'SecurityError'
} as const;

/** AudioContext states the engine checks. */
export const VOICE_AUDIO_CONTEXT_STATE = {
	CLOSED: 'closed',
	RUNNING: 'running',
	SUSPENDED: 'suspended'
} as const satisfies Record<string, AudioContextState>;

/** Output of the capture context where the browser allows none: it never opens the speakers. */
export const VOICE_AUDIO_SINK_NONE = { type: 'none' } as const;

/** What the browser lists microphones as. */
export const VOICE_AUDIO_INPUT_KIND = 'audioinput' satisfies MediaDeviceKind;

/** Entries Chrome lists beside the microphones: the system default and the communications device. */
export const VOICE_DEVICE_ALIAS = {
	COMMUNICATIONS: 'communications',
	DEFAULT: 'default'
} as const;

/** Microphone labels that name a Bluetooth headset, whose microphone switches it to call mode. */
export const VOICE_BLUETOOTH_LABEL_PATTERN = /bluetooth|airpods|beats|hands-?free|headset|buds/i;

/** Microphone labels that name the computer's own microphone. */
export const VOICE_BUILT_IN_LABEL_PATTERN = /built-?in|internal|macbook|imac|laptop/i;

/** The Permissions API entry for the microphone, and the state that needs no prompt. */
export const VOICE_MICROPHONE_PERMISSION = {
	GRANTED: 'granted' satisfies PermissionState,
	NAME: 'microphone'
} as const;

/** State of a track that still records. */
export const VOICE_TRACK_LIVE = 'live' satisfies MediaStreamTrackState;

/** Messages between the main thread and the PCM tap worklet, and how long a flush may take. */
export const VOICE_WORKLET_MESSAGE = {
	FLUSH: 'flush',
	FLUSHED: 'flushed'
} as const;

export const VOICE_WORKLET_FLUSH_TIMEOUT_MS = 250;

/** Type of the worklet's inline source. */
export const VOICE_WORKLET_SOURCE_MIME = 'text/javascript';

/** Sample rate of the offline context that decodes stored clips: a placeholder, decoding keeps the file's own. */
export const VOICE_DECODE_SAMPLE_RATE = 44_100;

/** A data URL's header, before the base64 payload. */
export const VOICE_DATA_URL_HEADER_PATTERN = /^data:[^,]*,/;

/** Quieter than anything a microphone gives, for a block of digital silence (dBFS). */
export const VOICE_SILENCE_DB = -120;

/** Defaults of a PauseDetector, see PauseDetectorOptions. */
export const VOICE_PAUSE_DEFAULTS = {
	MAX_SEGMENT_MS: 10_000,
	MIN_SPEECH_MS: 300,
	/** How far past its longest length a piece waits for a quiet moment before it is cut anyway */
	OVERTIME: 1.5,
	PAUSE_MS: 500
} as const;

/** Fields of the `/v1/audio/transcriptions` form, and the response format asked for. */
export const VOICE_TRANSCRIPTION_FORM = {
	FILE: 'file',
	LANGUAGE: 'language',
	MODEL: 'model',
	PROMPT: 'prompt',
	RESPONSE_FORMAT: 'response_format'
} as const;

export const VOICE_TRANSCRIPTION_RESPONSE_FORMAT = 'json';

/** File names the audio is sent under: a whole clip, or a piece of a recording still going. */
export const VOICE_CLIP_FILENAME = {
	CLIP: 'audio.wav',
	PIECE: 'piece.wav'
} as const;

/** The header the browser sets itself for a multipart body. */
export const VOICE_CONTENT_TYPE_HEADER = 'content-type';

/** An error body that is an HTML page, which says nothing useful in a toast. */
export const VOICE_HTML_BODY_PATTERN = /^</;

/** Socket protocols for the page's HTTP protocol. */
export const VOICE_SOCKET_PROTOCOL = {
	HTTPS: 'https:',
	WS: 'ws:',
	WSS: 'wss:'
} as const;

/** Format of the PCM16 chunks sent to the realtime socket. */
export const VOICE_REALTIME_AUDIO_FORMAT = 'audio/pcm';

/** Whitespace at either end of a text, where two pieces join without another space. */
export const VOICE_TRAILING_SPACE_PATTERN = /\s$/;
export const VOICE_LEADING_SPACE_PATTERN = /^\s/;

/** Punctuation that closes onto the word before it, so dictated text needs no space before it. */
export const VOICE_CLOSING_PUNCTUATION_PATTERN = /^[,.;:!?)\]}]/;
