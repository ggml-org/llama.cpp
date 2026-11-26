// Guards the phone model sheet's held-row menu: a model that hears offers to transcribe
// dictation, as the desktop row's microphone does, and a model that cannot hear does not.

import ModelsSelectorSheetWrapper from './components/ModelsSelectorSheetWrapper.svelte';
import {
	MODELS_SHEET_TEXT,
	SETTINGS_KEYS,
	VOICE_DICTATION_MODEL_LABEL,
	VOICE_TRANSCRIPTION_MODEL_AUTO
} from '#lib/constants/index.ts';
import { ServerRole, VoiceModelRole } from '#lib/enums/index.ts';
import { serverStore, settingsStore, voiceStore } from '#lib/stores/index.ts';
import type { ModelItem } from '#lib/utils/index.ts';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

const HEARS = 'org/Voxtral-Mini-3B';
const DEAF = 'org/Qwen3-8B';
const item = (model: string): ModelItem => ({
	option: { capabilities: [], id: model, model, name: model }
});

let previousConfig: typeof settingsStore.config;
let previousRole: typeof serverStore.role;

beforeEach(async () => {
	previousConfig = settingsStore.config;
	previousRole = serverStore.role;
	settingsStore.config = {
		...settingsStore.config,
		[SETTINGS_KEYS.VOICE_TRANSCRIPTION_MODEL]: VOICE_TRANSCRIPTION_MODEL_AUTO
	};
	serverStore.role = ServerRole.ROUTER;
	vi.spyOn(voiceStore, 'roleOf').mockImplementation((model) =>
		model === HEARS ? VoiceModelRole.VOICE : null
	);
	// the drawer captures the pointer of a press on its content, and a dispatched touch has none
	vi.spyOn(Element.prototype, 'setPointerCapture').mockImplementation(() => {});
	await page.viewport(390, 844);
});

afterEach(() => {
	vi.restoreAllMocks();
	settingsStore.config = previousConfig;
	serverStore.role = previousRole;
});

/** Holds a row with a finger: a touch press, then the context menu a long touch raises. */
async function hold(row: HTMLElement) {
	row.dispatchEvent(
		new PointerEvent('pointerdown', { bubbles: true, isPrimary: true, pointerType: 'touch' })
	);
	row.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
}

function renderSheet() {
	return render(ModelsSelectorSheetWrapper, {
		activeId: DEAF,
		open: true,
		quickItems: [item(DEAF), item(HEARS)]
	});
}

it('offers a model that hears for dictation, and picks it', async () => {
	const screen = renderSheet();
	const row = screen.getByRole('button', { name: /Voxtral/ });

	await expect.element(row).toBeVisible();
	await hold(row.element() as HTMLElement);

	const use = page.getByRole('menuitem', { name: VOICE_DICTATION_MODEL_LABEL.use });

	await expect.element(use).toBeVisible();
	await use.click();

	expect(settingsStore.config[SETTINGS_KEYS.VOICE_TRANSCRIPTION_MODEL]).toBe(HEARS);
});

it('does not offer a model that cannot hear', async () => {
	const screen = renderSheet();
	const row = screen.getByRole('button', { name: /Qwen3/ });

	await expect.element(row).toBeVisible();
	await hold(row.element() as HTMLElement);

	await expect
		.element(page.getByRole('menuitem', { name: MODELS_SHEET_TEXT.MORE_DETAILS }))
		.toBeVisible();
	expect(page.getByRole('menuitem', { name: VOICE_DICTATION_MODEL_LABEL.use }).query()).toBeNull();
});

it('hands the chosen model back to Automatic', async () => {
	settingsStore.config = {
		...settingsStore.config,
		[SETTINGS_KEYS.VOICE_TRANSCRIPTION_MODEL]: HEARS
	};
	vi.spyOn(voiceStore, 'isDictationModel').mockImplementation((model) => model === HEARS);

	const screen = renderSheet();
	const row = screen.getByRole('button', { name: /Voxtral/ });

	await expect.element(row).toBeVisible();
	await hold(row.element() as HTMLElement);
	await page.getByRole('menuitem', { name: VOICE_DICTATION_MODEL_LABEL.pickAutomatically }).click();

	expect(settingsStore.config[SETTINGS_KEYS.VOICE_TRANSCRIPTION_MODEL]).toBe(
		VOICE_TRANSCRIPTION_MODEL_AUTO
	);
});

it('keeps the model Automatic picked', async () => {
	vi.spyOn(voiceStore, 'isDictationModel').mockImplementation((model) => model === HEARS);

	const screen = renderSheet();
	const row = screen.getByRole('button', { name: /Voxtral/ });

	await expect.element(row).toBeVisible();
	await hold(row.element() as HTMLElement);
	await page.getByRole('menuitem', { name: VOICE_DICTATION_MODEL_LABEL.keep }).click();

	expect(settingsStore.config[SETTINGS_KEYS.VOICE_TRANSCRIPTION_MODEL]).toBe(HEARS);
});
