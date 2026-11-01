// Guards the hidden-model rule of the selector: a hidden model leaves every
// section, while the model a conversation already uses still resolves, since it
// can be hidden and selected at the same time.

import ModelsSelectorHookHarness from './components/ModelsSelectorHookHarness.svelte';
import { ServerModelStatus, ServerRole } from '$lib/enums';
import { modelsStore } from '$lib/stores/models/index.svelte';
import { serverStore } from '$lib/stores/server.svelte';
import type { ApiModelDataEntry, ModelOption } from '$lib/types';
import { beforeEach, describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';

function option(model: string): ModelOption {
	return { capabilities: [], id: model, model, name: model };
}

function routerEntry(id: string, status: ServerModelStatus): ApiModelDataEntry {
	return {
		created: 0,
		id,
		in_cache: true,
		object: 'model',
		owned_by: 'llamacpp',
		path: `/models/${id}`,
		status: { value: status }
	};
}

const VISIBLE = option('org/visible:Q4_K_M');
const HIDDEN = option('org/hidden:Q4_K_M');
const HIDDEN_LOADED = option('org/hidden-loaded:Q4_K_M');
const HIDDEN_FAVORITE = option('org/hidden-favorite:Q4_K_M');

beforeEach(() => {
	modelsStore.models = [VISIBLE, HIDDEN, HIDDEN_LOADED, HIDDEN_FAVORITE];
	modelsStore.favoriteModelIds = new Set([HIDDEN_FAVORITE.model]);
	modelsStore.hiddenModelIds = new Set([HIDDEN.id, HIDDEN_LOADED.id, HIDDEN_FAVORITE.id]);
	modelsStore.routerModels = [routerEntry(HIDDEN_LOADED.model, ServerModelStatus.LOADED)];
	serverStore.role = ServerRole.ROUTER;
});

describe('hidden models in the selector', () => {
	it('lists what is not hidden', () => {
		const screen = render(ModelsSelectorHookHarness);

		expect(screen.component.listedIds()).toEqual([VISIBLE.id]);
	});

	it('keeps a hidden favorite and a hidden loaded model out of the sections', () => {
		const screen = render(ModelsSelectorHookHarness);
		const listed = screen.component.listedIds();

		expect(listed).not.toContain(HIDDEN_FAVORITE.id);
		expect(listed).not.toContain(HIDDEN_LOADED.id);
	});

	it('reports an empty list when everything is hidden', () => {
		modelsStore.hiddenModelIds = new Set([
			VISIBLE.id,
			HIDDEN.id,
			HIDDEN_LOADED.id,
			HIDDEN_FAVORITE.id
		]);

		const screen = render(ModelsSelectorHookHarness);

		expect(screen.component.isEmpty()).toBe(true);
	});

	it('still resolves the model the conversation uses when it is hidden', () => {
		const screen = render(ModelsSelectorHookHarness, {
			currentModel: () => HIDDEN.model
		});

		expect(screen.component.shownModelId()).toBe(HIDDEN.id);
	});
});
