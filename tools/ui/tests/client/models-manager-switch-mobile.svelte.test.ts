// Guards the phone path into a model switch: the composer trigger opens the manager, so
// the pane has to be able to point the open chat at the model it shows.

import ModelsManagerWrapper from './components/ModelsManagerWrapper.svelte';
import { ServerRole } from '$lib/enums';
import { modelsStore } from '$lib/stores/models/index.svelte';
import { serverStore } from '$lib/stores/server.svelte';
import { uiStore } from '$lib/stores/ui.svelte';
import type { ModelOption } from '$lib/types/models';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
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

describe('models manager on a phone', () => {
	beforeEach(async () => {
		serverStore.role = ServerRole.ROUTER;
		modelsStore.models = [option];
		modelsStore.favoriteModelIds = new Set();
		modelsStore.hiddenModelIds = new Set();
		modelsStore.selectedModelId = null;
		modelsStore.selectedModelName = null;
		// a loaded model keeps the switch from reaching for the server
		modelsStore.routerModels = [
			{
				created: 0,
				id: option.model,
				in_cache: true,
				object: 'model',
				owned_by: 'llamacpp',
				path: `/models/${option.model}`,
				status: { value: 'loaded' }
			}
		];
		uiStore.manageModelsOpen = true;
		uiStore.manageModelFocus = null;
		await page.viewport(PHONE.width, PHONE.height);
	});

	afterEach(async () => {
		vi.restoreAllMocks();
		uiStore.manageModelsOpen = false;
		uiStore.manageModelFocus = null;
		await page.viewport(DESKTOP.width, DESKTOP.height);
	});

	it('points the open chat at the model the pane shows', async () => {
		// the pick itself belongs to the store, so the test only pins the selection and
		// the hand back to the chat; a cold model would reach for the server
		const selectModelById = vi.spyOn(modelsStore, 'selectModelById').mockResolvedValue(undefined);
		const load = vi.spyOn(modelsStore.status, 'load').mockResolvedValue(undefined);
		const screen = render(ModelsManagerWrapper);

		uiStore.manageModelFocus = option.id;

		await screen.getByRole('button', { name: 'Use in this chat' }).click();

		await expect.poll(() => selectModelById.mock.calls.length).toBe(1);
		expect(selectModelById).toHaveBeenCalledWith(option.id, { recordRecent: true });
		await expect.poll(() => uiStore.manageModelsOpen).toBe(false);
		expect(load).toHaveBeenCalledTimes(0);
	});
});
