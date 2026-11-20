import {
	VOICE_MODE_MESSAGE,
	VOICE_TRANSCRIPTION_MODEL_AUTO,
	VOICE_TRANSCRIPTION_ONLY_PATTERN
} from '#lib/constants/voice.constants.ts';
import { VoiceInputMode, VoiceMode, VoiceModelRole } from '#lib/enums/index.ts';

export interface VoiceModeInput extends Omit<
	DictationModelInput,
	'singleModel' | 'singleModelHasAudio'
> {
	/** The Voice input setting. */
	setting: VoiceInputMode | string;
	/** The conversation's effective model, null when none is picked. */
	chatModel: string | null;
	chatModelHasAudio: boolean;
}

export type VoiceModeResolution =
	| {
			mode: VoiceMode.VOICE_MESSAGE;
			/** The chat model: it writes the transcript too, so a router swaps no model. */
			transcriptionModel: string | null;
			separateModel: false;
	  }
	| {
			mode: VoiceMode.DICTATION;
			transcriptionModel: string | null;
			/** A router model other than the chat model, which transcribing may load. */
			separateModel: boolean;
	  }
	| { mode: VoiceMode.UNAVAILABLE; reason: string };

/**
 * What a recording becomes for the conversation's model. Automatic sends the voice to a
 * chat model with audio input and dictates otherwise, also with a speech-to-text model
 * picked for the conversation: it writes down what is said and has nothing to answer a
 * voice message with. Dictation transcribes with the chat model when it can hear, else
 * (router mode) with the chosen transcription model or, on Automatic, a model with audio
 * input that is already loaded.
 */
export function resolveVoiceMode(input: VoiceModeInput): VoiceModeResolution {
	const { chatModel, chatModelHasAudio, isRouter, setting } = input;
	const sendsVoice =
		setting !== VoiceInputMode.DICTATION && !(chatModel && isTranscriptionOnlyModel(chatModel));

	if (sendsVoice && chatModelHasAudio) {
		return { mode: VoiceMode.VOICE_MESSAGE, separateModel: false, transcriptionModel: chatModel };
	}

	if (chatModelHasAudio) {
		return { mode: VoiceMode.DICTATION, separateModel: false, transcriptionModel: chatModel };
	}

	// a single-model server has no other model to transcribe with
	if (!isRouter) return { mode: VoiceMode.UNAVAILABLE, reason: VOICE_MODE_MESSAGE.unavailable };

	const transcriptionModel = dictationModelFor(input);

	if (transcriptionModel) {
		return {
			mode: VoiceMode.DICTATION,
			// on Automatic it is already loaded, so transcribing swaps no model in or out
			separateModel:
				input.transcriptionModel.trim() !== VOICE_TRANSCRIPTION_MODEL_AUTO &&
				transcriptionModel !== chatModel,
			transcriptionModel
		};
	}

	return { mode: VoiceMode.UNAVAILABLE, reason: VOICE_MODE_MESSAGE.unavailable };
}

/** What the microphone button says it will do, or why it can't. */
export function voiceModeLabel(resolution: VoiceModeResolution): string {
	switch (resolution.mode) {
		case VoiceMode.VOICE_MESSAGE:
			return VOICE_MODE_MESSAGE.voiceMessage;
		case VoiceMode.DICTATION:
			return VOICE_MODE_MESSAGE.dictate;
		default:
			return resolution.reason;
	}
}

/** Whether a model is a speech-to-text one, by its name: for dictation, not for chat. */
export function isTranscriptionOnlyModel(modelId: string): boolean {
	return VOICE_TRANSCRIPTION_ONLY_PATTERN.test(modelId);
}

export interface DictationModelInput {
	isRouter: boolean;
	/** The Transcription model setting; VOICE_TRANSCRIPTION_MODEL_AUTO picks one. */
	transcriptionModel: string;
	transcriptionModelHasAudio: boolean;
	/** Router mode: a loaded model with audio input, which Automatic falls back to. */
	loadedAudioModel?: string | null;
	/** Single-model mode: the server's model, and whether it hears. */
	singleModel?: string | null;
	singleModelHasAudio?: boolean;
}

/**
 * The model that transcribes dictation for chat models that cannot hear, as
 * resolveVoiceMode() picks it, or null when none can: the chosen transcription model, or on
 * Automatic a loaded model with audio input; a single-model server's model when it hears.
 */
export function dictationModelFor(input: DictationModelInput): string | null {
	if (!input.isRouter) return input.singleModelHasAudio ? (input.singleModel ?? null) : null;

	const chosen = input.transcriptionModel.trim();

	if (chosen !== VOICE_TRANSCRIPTION_MODEL_AUTO) {
		return input.transcriptionModelHasAudio ? chosen : null;
	}

	return input.loadedAudioModel ?? null;
}

/**
 * What kind of model with audio input a model is: a speech-to-text model, a chat model
 * that hears, or none (it cannot hear).
 */
export function voiceModelRole(modelId: string, hasAudio: boolean): VoiceModelRole | null {
	if (!hasAudio) return null;

	return isTranscriptionOnlyModel(modelId) ? VoiceModelRole.TRANSCRIPTION : VoiceModelRole.VOICE;
}
