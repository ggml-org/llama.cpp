import { STORAGE_APP_NAME } from './storage.constants';
import { VoiceRecordingBarStatus } from '@ggml-org/llama-ui/voice';
import { ChatFormVoiceState } from '#lib/enums/voice.enums.ts';

/** What the chat form's microphone says it will do, or why it can't, on its button. */
export const VOICE_MODE_MESSAGE = {
	dictate: 'Dictate',
	dictating: 'Dictating',
	realtimeOnlyBehindKey:
		'This model only transcribes live, over a connection a browser cannot open on a server with an API key.',
	recording: 'Recording a voice message',
	refining: 'Refining transcript...',
	sendAfterRefining: 'Sending once the transcript is refined',
	separateModelHint:
		'The transcription model is loaded on the server alongside the chat model, which a server that keeps one model at a time unloads.',
	starting: 'Starting...',
	tooQuiet:
		'Your microphone sounds quiet. Move closer, or turn its input level up in your system settings.',
	transcribing: 'Transcribing...',
	transcriptionFailed: 'Transcription failed',
	unavailable:
		'No model with audio input is available. In router mode, load one or pick a transcription model in Settings > Voice.',
	voiceMessage: 'Send a voice message'
} as const;

/** The status of a server that has no such route for the model, e.g. a live-only model asked over HTTP. */
export const VOICE_NOT_SUPPORTED_STATUS = 501;

/** Where the chat form remembers that it showed the hint about a separate transcription model. */
export const VOICE_SEPARATE_MODEL_HINT_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME}.voiceSeparateModelHintSeen`;

/** Name prefix of a voice message's WAV file; the rest makes it unique per recording. */
export const VOICE_CLIP_NAME_PREFIX = 'voice-';

/** File extension of a voice message. */
export const VOICE_CLIP_EXTENSION = '.wav';

/** Characters of a random id in a voice message's file name, after its time. */
export const VOICE_CLIP_ID_LENGTH = 8;

/** How often the recording's elapsed time is updated. */
export const VOICE_ELAPSED_TICK_MS = 200;

/**
 * A mouse starts recording on press, so the click that ends that press comes within this
 * long and is not another tap.
 */
export const VOICE_PRESS_CLICK_MS = 1000;

/**
 * How long the pointer or the focus rests on the microphone button before the microphone
 * opens ahead of a press, so a pointer only passing over it opens nothing (and the browser
 * shows no recording indicator for it).
 */
export const VOICE_WARMUP_REST_MS = 300;

/** How long a transcript that came back keeps looking for the message it belongs to. */
export const VOICE_TRANSCRIPT_APPLY_RETRY_DELAYS_MS = [500, 1500, 4000] as const;

/** A string that is already a URL (data, blob, http...), which an <audio> plays as it is. */
export const VOICE_URL_PATTERN = /^[a-z][a-z0-9+.-]*:/i;

/** A URL whose audio has to be fetched before its peaks can be drawn. */
export const VOICE_FETCHABLE_URL_PATTERN = /^(blob|https?):/i;

/** Number of characters at a stored clip's end that, with its name and length, tell clips apart. */
export const VOICE_PEAKS_CACHE_TAIL_LENGTH = 32;

/** The engine's server routes are relative (`./v1/...`); the app's base takes them from the root. */
export const VOICE_RELATIVE_PATH_PATTERN = /^\./;

/** A model file's extension, left out where a single model's name is shown. */
export const VOICE_GGUF_EXTENSION_PATTERN = /\.gguf$/i;

/** Element of the message box's editor: a tap there edits the live text of a voice message. */
export const VOICE_EDITOR_SELECTOR = '.cm-editor';

/** What Settings > Voice says beside its pickers. */
export const VOICE_SETTINGS_LABEL = {
	automatic: 'Automatic',
	chatModels: 'Chat models with audio input',
	disconnectedMicrophone: 'A microphone that is not connected',
	microphone: (index: number) => `Microphone ${index}`,
	speechToText: 'Speech to text',
	systemDefault: 'System default'
} as const;

/** What a voice message attachment is called, in place of its file name. */
export const VOICE_MESSAGE_LABEL = 'Voice message';

/**
 * The Transcription model setting's Automatic choice: the chat model when it hears, else a
 * model with audio input that is already loaded, so dictating loads nothing.
 */
export const VOICE_TRANSCRIPTION_MODEL_AUTO = '';

/**
 * Names of speech-to-text models, which write down what is said and are not meant to chat
 * (Qwen3-ASR, Whisper, Parakeet...). The server tells them apart by their chat template and
 * does not say so, so the app goes by the name.
 */
export const VOICE_TRANSCRIPTION_ONLY_PATTERN =
	/(^|[^a-z0-9])(asr|whisper|parakeet|canary|moonshine|sensevoice|stt)([^a-z0-9]|$)/i;

/** How the model picker and the Models manager name a kind of model with audio input. */
export const VOICE_MODEL_ROLE_LABEL = {
	transcription: 'Speech to text',
	voice: 'Voice'
} as const;

/**
 * What the microphone on a model's row says: on the model that transcribes dictation for
 * chat models that cannot hear, whether it was chosen or picked automatically (a loaded
 * model with audio input), and on any other model that hears, that it can be the one.
 */
export const VOICE_DICTATION_MODEL_LABEL = {
	automatic: 'Transcribes dictation, picked automatically. Click to keep it',
	chosen: 'Transcribes dictation. Click to pick automatically',
	current: 'transcribes dictation',
	/** The phone menu's keep on the model Automatic picked. */
	keep: 'Keep for dictation',
	/** The phone menu's way back from a chosen model to Automatic. */
	pickAutomatically: 'Pick dictation model automatically',
	use: 'Use for dictation'
} as const;

/** What each kind means, in the badge's tooltip. */
export const VOICE_MODEL_ROLE_DESCRIPTION = {
	transcription:
		'A speech-to-text model: it writes down what is said, for dictation, and is not meant to chat.',
	voice:
		'Hears audio: as the chat model it gets your voice messages, and it can also transcribe dictation.'
} as const;

/** What the recording bar shows for each state of the chat form's recording. */
export const VOICE_RECORDING_BAR_STATUS: Record<ChatFormVoiceState, VoiceRecordingBarStatus> = {
	[ChatFormVoiceState.IDLE]: VoiceRecordingBarStatus.RECORDING,
	[ChatFormVoiceState.RECORDING]: VoiceRecordingBarStatus.RECORDING,
	[ChatFormVoiceState.STARTING]: VoiceRecordingBarStatus.STARTING,
	[ChatFormVoiceState.TRANSCRIBING]: VoiceRecordingBarStatus.WAITING
};
