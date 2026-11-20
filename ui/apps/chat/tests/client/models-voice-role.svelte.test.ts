// Guards how the model picker shows each model's part in voice input: a badge for a
// speech-to-text model and for a chat model that hears, nothing for one that can't, and a
// microphone among the row's actions that is lit on the model that transcribes dictation
// and picks it on any other model that hears; and that a loaded model kept only for
// dictation sits apart from the chat models, under its own heading.

import ModelsSelectorListWrapper from './components/ModelsSelectorListWrapper.svelte';
import { SETTINGS_KEYS, VOICE_TRANSCRIPTION_MODEL_AUTO } from '#lib/constants/index.ts';
import { ServerModelStatus, ServerRole } from '#lib/enums/index.ts';
import { serverStore } from '#lib/stores/llama-server.svelte.ts';
import { modelsStore } from '#lib/stores/models/index.svelte.ts';
import { settingsStore } from '#lib/stores/settings/index.svelte.ts';
import type { ApiModelDataEntry, ModelOption } from '#lib/types/index.ts';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';

function option(model: string, audio: boolean): ModelOption {
	return {
		capabilities: [],
		id: model,
		modalities: { audio, video: false, vision: false },
		model,
		name: model
	};
}

function loadedEntry(id: string): ApiModelDataEntry {
	return {
		created: 0,
		id,
		in_cache: true,
		object: 'model',
		owned_by: 'llamacpp',
		path: `/models/${id}`,
		status: { value: ServerModelStatus.LOADED }
	};
}

const CHAT = option('org/chat-text:Q4_K_M', false);
const OMNI = option('org/omni-2B:Q8_0', true);
const ASR = option('org/Qwen3-ASR-0.6B:Q8_0', true);

let previousConfig: typeof settingsStore.config;

beforeEach(() => {
	previousConfig = settingsStore.config;
	modelsStore.models = [CHAT, OMNI, ASR];
	modelsStore.routerModels = [loadedEntry(CHAT.model), loadedEntry(ASR.model)];
	serverStore.role = ServerRole.ROUTER;
});

afterEach(() => {
	settingsStore.config = previousConfig;
});

function useTranscriptionModel(model: string) {
	settingsStore.config = {
		...settingsStore.config,
		[SETTINGS_KEYS.VOICE_TRANSCRIPTION_MODEL]: model
	};
}

function renderList(currentModel: string) {
	const items = [CHAT, OMNI, ASR].map((entry) => ({ option: entry }));

	return render(ModelsSelectorListWrapper, {
		currentModel,
		groups: { available: [{ items, orgName: null }] },
		loaded: [{ option: CHAT }, { option: ASR }]
	});
}

/** The role badge of each listed row, by model, as the rows show them. */
function roles(container: HTMLElement): Record<string, string | null> {
	const out: Record<string, string | null> = {};

	for (const row of container.querySelectorAll<HTMLElement>('[role="option"]')) {
		const model = [CHAT, OMNI, ASR].find((entry) => row.querySelector(`[title="${entry.model}"]`));

		if (model)
			out[model.model] =
				row.querySelector<HTMLElement>('[data-slot="model-voice-role"]')?.dataset.role ?? null;
	}

	return out;
}

/** The microphone action of a model's row, null when the row has none. */
function microphone(container: HTMLElement, model: string): HTMLElement | null {
	for (const row of container.querySelectorAll<HTMLElement>('[role="option"]')) {
		if (row.querySelector(`[title="${model}"]`)) {
			return row.querySelector<HTMLElement>('[data-slot="model-dictation-action"]');
		}
	}

	return null;
}

/** The models whose microphone is lit, as the ones that transcribe dictation. */
function dictating(container: HTMLElement): string[] {
	return [CHAT, OMNI, ASR]
		.map((entry) => entry.model)
		.filter((model) => microphone(container, model)?.dataset.dictates !== undefined);
}

describe('voice roles in the model picker', () => {
	it('badges a speech-to-text model and a chat model that hears', () => {
		useTranscriptionModel(OMNI.model);

		const screen = renderList(CHAT.model);

		expect(roles(screen.container)).toEqual({
			[ASR.model]: 'transcription',
			[CHAT.model]: null,
			[OMNI.model]: 'voice'
		});
	});

	it('lights the microphone of the dictation model, and offers it on the others that hear', () => {
		useTranscriptionModel(OMNI.model);

		const screen = renderList(CHAT.model);

		expect(dictating(screen.container)).toEqual([OMNI.model]);
		expect(microphone(screen.container, ASR.model)).not.toBeNull();
		// a model that cannot hear has no microphone
		expect(microphone(screen.container, CHAT.model)).toBeNull();
	});

	it('picks the dictation model with a click, and hands it back to Automatic with another', async () => {
		useTranscriptionModel(OMNI.model);

		const screen = renderList(CHAT.model);
		const click = (model: string) =>
			microphone(screen.container, model)!.querySelector('button')!.click();

		click(ASR.model);
		await expect.poll(() => dictating(screen.container)).toEqual([ASR.model]);
		expect(settingsStore.config[SETTINGS_KEYS.VOICE_TRANSCRIPTION_MODEL]).toBe(ASR.model);

		click(ASR.model);
		await expect
			.poll(() => settingsStore.config[SETTINGS_KEYS.VOICE_TRANSCRIPTION_MODEL])
			.toBe(VOICE_TRANSCRIPTION_MODEL_AUTO);
	});

	it('lists a loaded model kept for dictation apart from the chat models', () => {
		useTranscriptionModel(ASR.model);

		const screen = renderList(CHAT.model);
		const heading = screen.container.querySelector('[data-slot="models-loaded-dictation-heading"]');

		expect(heading?.textContent?.trim()).toBe('For dictation');
		// the dictation model comes after the heading, the chat model before it
		const rows = [...screen.container.querySelectorAll('[role="option"]')].slice(0, 2);

		expect(rows[0].textContent).toContain('chat-text');
		expect(rows[1].textContent).toContain('Qwen3-ASR');
		expect(heading?.compareDocumentPosition(rows[1]) ?? 0).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
	});

	it('keeps the conversation model with the chat models, even when it dictates', () => {
		useTranscriptionModel(ASR.model);

		const screen = renderList(ASR.model);

		expect(
			screen.container.querySelector('[data-slot="models-loaded-dictation-heading"]')
		).toBeNull();
	});

	it('on Automatic, gives dictation to the loaded model that hears', () => {
		useTranscriptionModel(VOICE_TRANSCRIPTION_MODEL_AUTO);

		const screen = renderList(CHAT.model);

		expect(dictating(screen.container)).toEqual([ASR.model]);
		expect(roles(screen.container)[OMNI.model]).toBe('voice');
	});
});
