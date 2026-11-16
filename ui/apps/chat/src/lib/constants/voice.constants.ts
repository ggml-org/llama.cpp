import { STORAGE_APP_NAME } from './storage.constants';

/** What the chat form's microphone says it will do, or why it can't, on its button. */
export const VOICE_MODE_MESSAGE = {
	dictate: 'Dictate',
	dictating: 'Dictating',
	recording: 'Recording a voice message',
	separateModelHint:
		'The transcription model is loaded on the server alongside the chat model, which a server that keeps one model at a time unloads.',
	transcribing: 'Transcribing...',
	unavailable:
		'No model with audio input is available. In router mode, load one or pick a transcription model in Settings > Voice.',
	voiceMessage: 'Send a voice message'
} as const;

/** Where the chat form remembers that it showed the hint about a separate transcription model. */
export const VOICE_SEPARATE_MODEL_HINT_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME}.voiceSeparateModelHintSeen`;

/** Name prefix of a voice message's WAV file; the rest makes it unique per recording. */
export const VOICE_CLIP_NAME_PREFIX = 'voice-';

/**
 * The Transcription model setting's Automatic choice: the chat model when it hears, else a
 * model with audio input that is already loaded, so dictating loads nothing.
 */
export const VOICE_TRANSCRIPTION_MODEL_AUTO = '';
