import { STORAGE_APP_NAME } from './storage.constants';

/** What the chat form's microphone says it will do, or why it can't, on its button. */
export const VOICE_MODE_MESSAGE = {
	dictate: 'Dictate',
	dictating: 'Dictating',
	recording: 'Recording a voice message',
	refining: 'Refining transcript...',
	sendAfterRefining: 'Sending once the transcript is refined',
	separateModelHint:
		'The transcription model is loaded on the server alongside the chat model, which a server that keeps one model at a time unloads.',
	starting: 'Starting...',
	tooQuiet:
		'Your microphone sounds quiet. Move closer, or turn its input level up in your system settings.',
	transcribing: 'Transcribing...',
	unavailable:
		'No model with audio input is available. In router mode, load one or pick a transcription model in Settings > Voice.',
	voiceMessage: 'Send a voice message'
} as const;

/** Where the chat form remembers that it showed the hint about a separate transcription model. */
export const VOICE_SEPARATE_MODEL_HINT_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME}.voiceSeparateModelHintSeen`;

/** Name prefix of a voice message's WAV file; the rest makes it unique per recording. */
export const VOICE_CLIP_NAME_PREFIX = 'voice-';

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

/** How the model picker and the Models manager name a model's part in voice input. */
export const VOICE_MODEL_ROLE_LABEL = {
	dictation: 'Dictation',
	transcription: 'Speech to text',
	voice: 'Voice'
} as const;

/** What each part means, in the badge's tooltip. */
export const VOICE_MODEL_ROLE_DESCRIPTION = {
	dictationAuto:
		'Transcribes your dictation with chat models that cannot hear. Picked automatically, as a loaded model with audio input; choose one in Settings > Voice.',
	dictationChosen:
		'Transcribes your dictation with chat models that cannot hear, as chosen in Settings > Voice.',
	transcription:
		'A speech-to-text model: it writes down what is said, for dictation, and is not meant to chat.',
	voice:
		'Hears audio: as the chat model it gets your voice messages, and it can also transcribe dictation.'
} as const;
