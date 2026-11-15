import { VOICE_MODE_MESSAGE } from '#lib/constants/voice.constants.ts';
import { VoiceInputMode, VoiceMode } from '#lib/enums/index.ts';

export interface VoiceModeInput {
	/** The Voice input setting. */
	setting: VoiceInputMode | string;
	/** The conversation's effective model, null when none is picked. */
	chatModel: string | null;
	chatModelHasAudio: boolean;
	isRouter: boolean;
	/** The Transcription model setting; empty means the chat model. */
	transcriptionModel: string;
	transcriptionModelHasAudio: boolean;
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
 * model with audio input and dictates otherwise; dictation transcribes with the chat model
 * when it can hear, else (router mode) with the chosen transcription model.
 */
export function resolveVoiceMode(input: VoiceModeInput): VoiceModeResolution {
	const { chatModel, chatModelHasAudio, isRouter, setting } = input;

	if (setting !== VoiceInputMode.DICTATION && chatModelHasAudio) {
		return { mode: VoiceMode.VOICE_MESSAGE, separateModel: false, transcriptionModel: chatModel };
	}

	if (chatModelHasAudio) {
		return { mode: VoiceMode.DICTATION, separateModel: false, transcriptionModel: chatModel };
	}

	// a single-model server has no other model to transcribe with
	const chosen = isRouter ? input.transcriptionModel.trim() : '';

	if (chosen && input.transcriptionModelHasAudio) {
		return {
			mode: VoiceMode.DICTATION,
			separateModel: chosen !== chatModel,
			transcriptionModel: chosen
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
