// Guards what the microphone does for a conversation's model: a voice message for a model
// that hears, dictation otherwise, and why it can't when no model can transcribe; and each
// model's part in voice input, as the model lists show it.

import {
	SETTING_CONFIG_DEFAULT,
	SETTINGS_CHAT_SECTIONS,
	SETTINGS_KEYS,
	SETTINGS_SECTION_SLUGS,
	VOICE_MODE_MESSAGE,
	VOICE_TRANSCRIPTION_MODEL_AUTO
} from '#lib/constants/index.ts';
import { SettingsFieldType, VoiceInputMode, VoiceMode, VoiceModelRole } from '#lib/enums/index.ts';
import {
	dictationModelFor,
	isTranscriptionOnlyModel,
	resolveVoiceMode,
	type VoiceModeInput,
	voiceModeLabel,
	voiceModelRole
} from '#lib/utils/voice-mode.ts';
import { describe, expect, it } from 'vitest';

const base: VoiceModeInput = {
	chatModel: 'gemma-3n',
	chatModelHasAudio: true,
	isRouter: true,
	setting: VoiceInputMode.AUTO,
	transcriptionModel: '',
	transcriptionModelHasAudio: false
};

describe('resolveVoiceMode', () => {
	it('sends a voice message to a chat model with audio, transcribed by that model', () => {
		expect(resolveVoiceMode(base)).toEqual({
			mode: VoiceMode.VOICE_MESSAGE,
			separateModel: false,
			transcriptionModel: 'gemma-3n'
		});
	});

	it('dictates with a speech-to-text model picked for the conversation', () => {
		expect(resolveVoiceMode({ ...base, chatModel: 'org/VibeVoice-ASR:Q8_0' })).toEqual({
			mode: VoiceMode.DICTATION,
			separateModel: false,
			transcriptionModel: 'org/VibeVoice-ASR:Q8_0'
		});
	});

	it('dictates with the chat model when the user always dictates', () => {
		expect(resolveVoiceMode({ ...base, setting: VoiceInputMode.DICTATION })).toEqual({
			mode: VoiceMode.DICTATION,
			separateModel: false,
			transcriptionModel: 'gemma-3n'
		});
	});

	it('prefers the chat model over a chosen transcription model when it can hear', () => {
		const result = resolveVoiceMode({
			...base,
			setting: VoiceInputMode.DICTATION,
			transcriptionModel: 'voxtral',
			transcriptionModelHasAudio: true
		});

		expect(result).toMatchObject({ separateModel: false, transcriptionModel: 'gemma-3n' });
	});

	it('dictates with the chosen model for a chat model without audio, in router mode', () => {
		expect(
			resolveVoiceMode({
				...base,
				chatModel: 'qwen3',
				chatModelHasAudio: false,
				transcriptionModel: 'voxtral',
				transcriptionModelHasAudio: true
			})
		).toEqual({ mode: VoiceMode.DICTATION, separateModel: true, transcriptionModel: 'voxtral' });
	});

	it('is unavailable when no model can transcribe', () => {
		const noAudio = { ...base, chatModel: 'qwen3', chatModelHasAudio: false };

		expect(resolveVoiceMode(noAudio)).toEqual({
			mode: VoiceMode.UNAVAILABLE,
			reason: VOICE_MODE_MESSAGE.unavailable
		});
		// a chosen model without audio input does not count
		expect(
			resolveVoiceMode({
				...noAudio,
				transcriptionModel: 'llama',
				transcriptionModelHasAudio: false
			}).mode
		).toBe(VoiceMode.UNAVAILABLE);
	});

	it('falls back to a loaded model with audio input on Automatic, loading nothing', () => {
		const noAudio = { ...base, chatModel: 'qwen3', chatModelHasAudio: false };

		expect(resolveVoiceMode({ ...noAudio, loadedAudioModel: 'qwen3-asr' })).toEqual({
			mode: VoiceMode.DICTATION,
			separateModel: false,
			transcriptionModel: 'qwen3-asr'
		});
		// a chosen model wins over the loaded one
		expect(
			resolveVoiceMode({
				...noAudio,
				loadedAudioModel: 'qwen3-asr',
				transcriptionModel: 'voxtral',
				transcriptionModelHasAudio: true
			})
		).toEqual({ mode: VoiceMode.DICTATION, separateModel: true, transcriptionModel: 'voxtral' });
		// so does the chat model, when it hears
		expect(resolveVoiceMode({ ...base, loadedAudioModel: 'qwen3-asr' }).mode).toBe(
			VoiceMode.VOICE_MESSAGE
		);
		// and nothing is loaded behind the user's back
		expect(resolveVoiceMode({ ...noAudio, loadedAudioModel: null }).mode).toBe(
			VoiceMode.UNAVAILABLE
		);
		expect(
			resolveVoiceMode({ ...noAudio, isRouter: false, loadedAudioModel: 'qwen3-asr' }).mode
		).toBe(VoiceMode.UNAVAILABLE);
	});

	it('ignores the transcription model on a single-model server', () => {
		expect(
			resolveVoiceMode({
				...base,
				chatModelHasAudio: false,
				isRouter: false,
				transcriptionModel: 'voxtral',
				transcriptionModelHasAudio: true
			}).mode
		).toBe(VoiceMode.UNAVAILABLE);
	});

	it('labels the microphone with what it will do', () => {
		expect(voiceModeLabel(resolveVoiceMode(base))).toBe('Send a voice message');
		expect(voiceModeLabel(resolveVoiceMode({ ...base, setting: VoiceInputMode.DICTATION }))).toBe(
			'Dictate'
		);
		expect(voiceModeLabel(resolveVoiceMode({ ...base, chatModelHasAudio: false }))).toBe(
			VOICE_MODE_MESSAGE.unavailable
		);
	});
});

describe('voice settings', () => {
	const section = SETTINGS_CHAT_SECTIONS.find((s) => s.slug === SETTINGS_SECTION_SLUGS.VOICE);

	it('default to automatic voice input and transcription model', () => {
		expect(SETTING_CONFIG_DEFAULT[SETTINGS_KEYS.VOICE_INPUT_MODE]).toBe(VoiceInputMode.AUTO);
		expect(SETTING_CONFIG_DEFAULT[SETTINGS_KEYS.VOICE_TRANSCRIBE_MESSAGES]).toBe(true);
		expect(SETTING_CONFIG_DEFAULT[SETTINGS_KEYS.VOICE_COLLAPSE_TRANSCRIPTS]).toBe(false);
		expect(SETTING_CONFIG_DEFAULT[SETTINGS_KEYS.VOICE_BOOST]).toBe(true);
		expect(SETTING_CONFIG_DEFAULT[SETTINGS_KEYS.VOICE_NOISE_SUPPRESSION]).toBe(false);
		expect(SETTING_CONFIG_DEFAULT[SETTINGS_KEYS.VOICE_ECHO_CANCELLATION]).toBe(false);
		expect(SETTING_CONFIG_DEFAULT[SETTINGS_KEYS.VOICE_AUTO_GAIN]).toBe(false);
		expect(SETTING_CONFIG_DEFAULT[SETTINGS_KEYS.VOICE_TRANSCRIPTION_MODEL]).toBe(
			VOICE_TRANSCRIPTION_MODEL_AUTO
		);
		expect(SETTING_CONFIG_DEFAULT[SETTINGS_KEYS.VOICE_TRANSCRIPTION_LANGUAGE]).toBe('');
	});

	it('have a section of their own, the model picker drawn by it', () => {
		const keys = section?.fields?.map((field) => field.key);

		expect(keys).toEqual([
			SETTINGS_KEYS.VOICE_INPUT_MODE,
			SETTINGS_KEYS.VOICE_TRANSCRIBE_MESSAGES,
			SETTINGS_KEYS.VOICE_COLLAPSE_TRANSCRIPTS,
			SETTINGS_KEYS.VOICE_BOOST,
			SETTINGS_KEYS.VOICE_NOISE_SUPPRESSION,
			SETTINGS_KEYS.VOICE_ECHO_CANCELLATION,
			SETTINGS_KEYS.VOICE_AUTO_GAIN,
			SETTINGS_KEYS.VOICE_TRANSCRIPTION_LANGUAGE
		]);
		expect(section?.fields?.[0].type).toBe(SettingsFieldType.SELECT);
	});
});

describe('isTranscriptionOnlyModel', () => {
	it('knows speech-to-text models by name', () => {
		expect(isTranscriptionOnlyModel('ggml-org/Qwen3-ASR-0.6B-GGUF:Q8_0')).toBe(true);
		expect(isTranscriptionOnlyModel('ggerganov/whisper-large-v3')).toBe(true);
		expect(isTranscriptionOnlyModel('nvidia/parakeet-tdt-0.6b')).toBe(true);
	});

	it('leaves chat models with audio input alone, even with the letters inside a word', () => {
		expect(isTranscriptionOnlyModel('ggml-org/LFM3-Omni-2B-GGUF:Q8_0')).toBe(false);
		expect(isTranscriptionOnlyModel('google/gemma-3n-E4B')).toBe(false);
		expect(isTranscriptionOnlyModel('org/Lasr-chat')).toBe(false);
	});
});

describe('dictationModelFor', () => {
	const router = { isRouter: true, transcriptionModelHasAudio: true };

	it('is the chosen transcription model when it hears', () => {
		expect(dictationModelFor({ ...router, transcriptionModel: 'asr' })).toBe('asr');
		expect(
			dictationModelFor({ ...router, transcriptionModel: 'asr', transcriptionModelHasAudio: false })
		).toBeNull();
	});

	it('is a loaded model with audio input on Automatic, none when nothing hears', () => {
		expect(
			dictationModelFor({
				...router,
				loadedAudioModel: 'omni',
				transcriptionModel: VOICE_TRANSCRIPTION_MODEL_AUTO
			})
		).toBe('omni');
		expect(
			dictationModelFor({ ...router, transcriptionModel: VOICE_TRANSCRIPTION_MODEL_AUTO })
		).toBeNull();
	});

	it("is a single-model server's model when it hears", () => {
		const single = { isRouter: false, transcriptionModel: '', transcriptionModelHasAudio: false };

		expect(dictationModelFor({ ...single, singleModel: 'omni', singleModelHasAudio: true })).toBe(
			'omni'
		);
		expect(
			dictationModelFor({ ...single, singleModel: 'text', singleModelHasAudio: false })
		).toBeNull();
	});
});

describe('voiceModelRole', () => {
	it('tells speech-to-text models from chat models that hear, and leaves the rest out', () => {
		expect(voiceModelRole('org/Qwen3-ASR', true)).toBe(VoiceModelRole.TRANSCRIPTION);
		expect(voiceModelRole('omni', true)).toBe(VoiceModelRole.VOICE);
		expect(voiceModelRole('text-only', false)).toBeNull();
	});
});
