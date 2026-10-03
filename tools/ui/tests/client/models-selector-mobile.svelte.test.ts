// Guards the phone model trigger of the chat form: the manager dialog is the picker
// there, so the trigger opens it on the model the conversation runs.

import ModelsSelectorMobileTrigger from '$lib/components/app/models/ModelsSelector/ModelsSelectorMobileTrigger.svelte';
import { ServerRole } from '$lib/enums';
import { modelsStore } from '$lib/stores/models/index.svelte';
import { serverStore } from '$lib/stores/server.svelte';
import { uiStore } from '$lib/stores/ui.svelte';
import type { ModelOption } from '$lib/types/models';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { page } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

const PHONE = { height: 844, width: 390 };
const DESKTOP = { height: 900, width: 1280 };
const option: ModelOption = {
	capabilities: [],
	id: 'org/Qwen3-8B:Q4_K_M',
	model: 'org/Qwen3-8B:Q4_K_M',
	name: 'Qwen3-8B'
};
const other: ModelOption = {
	capabilities: [],
	id: 'org/Qwen3-8B:Q8_0',
	model: 'org/Qwen3-8B:Q8_0',
	name: 'Qwen3-8B-Q8_0'
};

describe('model trigger on a phone', () => {
	beforeEach(async () => {
		serverStore.role = ServerRole.ROUTER;
		modelsStore.models = [option, other];
		uiStore.manageModelFocus = null;
		uiStore.manageModelsOpen = false;
		await page.viewport(PHONE.width, PHONE.height);
	});

	afterEach(async () => {
		modelsStore.models = [];
		await page.viewport(DESKTOP.width, DESKTOP.height);
	});

	it('opens the manager', async () => {
		const screen = render(ModelsSelectorMobileTrigger, { currentModel: option.model });

		await screen.getByRole('button').click();

		expect(uiStore.manageModelsOpen).toBe(true);
		expect(uiStore.manageModelFocus).toBeNull();
	});
});
