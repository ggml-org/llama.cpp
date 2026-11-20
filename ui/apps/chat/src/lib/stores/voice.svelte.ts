/**
 * voiceStore - Which model does what in voice input
 *
 * One place for what the microphone does with the conversation's model (a voice message,
 * dictation, or nothing, see resolveVoiceMode) and for the model that transcribes
 * dictation, so the chat form, the model picker and the Models manager agree on it: where
 * models are listed, a badge says what kind of hearing model each is (roleOf) and a
 * microphone which one transcribes dictation (isDictationModel), and picks it.
 */

import { SETTINGS_KEYS, VOICE_TRANSCRIPTION_MODEL_AUTO } from '#lib/constants/index.ts';
import { VoiceInputMode, VoiceModelRole } from '#lib/enums/index.ts';
import { serverStore } from '#lib/stores/llama-server.svelte.ts';
// direct imports between stores, not via the barrel, to avoid circular deps
import { modelsStore } from '#lib/stores/models/index.svelte.ts';
import { settingsStore } from '#lib/stores/settings/index.svelte.ts';
import { dictationModelFor, resolveVoiceMode, voiceModelRole } from '#lib/utils/voice-mode.ts';

class VoiceStore {
	/** A router with a model that hears: the microphone stays, so it can say where to pick one. */
	readonly anyAudioModel = $derived.by(() => {
		void modelsStore.props.cacheVersion;

		return (
			serverStore.isRouterMode && modelsStore.models.some((option) => this.hasAudio(option.model))
		);
	});
	private readonly config = $derived(settingsStore.config);

	private readonly transcriptionSetting = $derived(
		String(this.config[SETTINGS_KEYS.VOICE_TRANSCRIPTION_MODEL] ?? VOICE_TRANSCRIPTION_MODEL_AUTO)
	);

	/** The dictation model is picked automatically, not chosen in Settings > Voice. */
	readonly dictationIsAutomatic = $derived(
		this.transcriptionSetting.trim() === VOICE_TRANSCRIPTION_MODEL_AUTO
	);

	/** Router mode: what Automatic transcribes with when the chat model can't hear, a loaded model with audio input. */
	readonly loadedAudioModel = $derived.by(() => {
		void modelsStore.props.cacheVersion;

		if (!serverStore.isRouterMode) return null;

		return (
			modelsStore.models.find(
				(option) => this.hasAudio(option.model) && modelsStore.isModelLoaded(option.model)
			)?.model ?? null
		);
	});

	/** The model that transcribes dictation for chat models that can't hear, null when none can. */
	readonly dictationModel = $derived.by(() => {
		const singleModel = serverStore.isRouterMode ? null : modelsStore.activeModelId;

		return dictationModelFor({
			isRouter: serverStore.isRouterMode,
			loadedAudioModel: this.loadedAudioModel,
			singleModel,
			singleModelHasAudio: this.hasAudio(singleModel),
			transcriptionModel: this.transcriptionSetting,
			transcriptionModelHasAudio: this.hasAudio(this.transcriptionSetting || null)
		});
	});

	/** What the microphone does for the conversation's model. */
	readonly resolution = $derived.by(() => {
		const chatModel = modelsStore.activeModelId;

		return resolveVoiceMode({
			chatModel,
			chatModelHasAudio: this.hasAudio(chatModel),
			isRouter: serverStore.isRouterMode,
			loadedAudioModel: this.loadedAudioModel,
			setting: String(this.config[SETTINGS_KEYS.VOICE_INPUT_MODE] ?? VoiceInputMode.AUTO),
			transcriptionModel: this.transcriptionSetting,
			transcriptionModelHasAudio: this.hasAudio(this.transcriptionSetting || null)
		});
	});

	/** Whether a model has audio input, as far as its props say. */
	hasAudio(model: string | null): boolean {
		void modelsStore.props.cacheVersion;

		return model ? modelsStore.props.modelSupportsAudio(model) : false;
	}

	/** Whether a model is the one that transcribes dictation for chat models that can't hear. */
	isDictationModel(model: string): boolean {
		return this.dictationModel === model;
	}

	/** What kind of model with audio input a model is, null when it cannot hear. */
	roleOf(model: string): VoiceModelRole | null {
		return voiceModelRole(model, this.hasAudio(model));
	}

	/**
	 * What a click on a model's microphone does: makes it the model that transcribes
	 * dictation, or, on the one already chosen, hands the choice back to Automatic.
	 */
	toggleDictation(model: string): void {
		const chosen = !this.dictationIsAutomatic && this.transcriptionSetting.trim() === model;

		settingsStore.updateConfig(
			SETTINGS_KEYS.VOICE_TRANSCRIPTION_MODEL,
			chosen ? VOICE_TRANSCRIPTION_MODEL_AUTO : model
		);
	}

	/** Makes a model the one that transcribes dictation (Settings > Voice > Transcription model). */
	useForDictation(model: string): void {
		settingsStore.updateConfig(SETTINGS_KEYS.VOICE_TRANSCRIPTION_MODEL, model);
	}
}

export const voiceStore = new VoiceStore();
