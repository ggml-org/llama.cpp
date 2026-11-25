// Guards the model selector's sections: the server's models are listed directly, and sit in an
// Other models section only beside favorites or loaded models, which are sections of their own.

import ModelsSelectorListWrapper from './components/ModelsSelectorListWrapper.svelte';
import { ServerRole } from '#lib/enums/index.ts';
import { serverStore } from '#lib/stores/llama-server.svelte.ts';
import { modelsStore } from '#lib/stores/models/index.svelte.ts';
import type { ModelOption } from '#lib/types/index.ts';
import { beforeEach, describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';

const option = (name: string): ModelOption => ({
	capabilities: [],
	id: `org/${name}:Q4_K_M`,
	model: `org/${name}:Q4_K_M`,
	name
});
const ALPHA = option('Alpha-8B');
const BETA = option('Beta-8B');

beforeEach(() => {
	serverStore.role = ServerRole.ROUTER;
	modelsStore.models = [ALPHA, BETA];
});

function renderList(extra: { favorites?: ModelOption[]; loaded?: ModelOption[] } = {}) {
	return render(ModelsSelectorListWrapper, {
		currentModel: null,
		favorites: (extra.favorites ?? []).map((entry) => ({ option: entry })),
		groups: { available: [{ items: [{ option: BETA }], orgName: null }] },
		loaded: (extra.loaded ?? []).map((entry) => ({ option: entry }))
	});
}

describe('model selector sections', () => {
	it('lists the models without a section when none is a favorite or loaded', async () => {
		const screen = renderList();

		await expect.element(screen.getByRole('option')).toBeVisible();
		await expect.element(screen.getByText('Other models')).not.toBeInTheDocument();
		await expect.element(screen.getByText('Local models')).not.toBeInTheDocument();
	});

	it('puts the other models in a section beside the favorites', async () => {
		const screen = renderList({ favorites: [ALPHA] });

		await expect.element(screen.getByText('Favorites')).toBeVisible();
		await expect.element(screen.getByText('Other models')).toBeVisible();
	});

	it('puts the other models in a section beside the loaded ones', async () => {
		const screen = renderList({ loaded: [ALPHA] });

		await expect.element(screen.getByText('Loaded models')).toBeVisible();
		await expect.element(screen.getByText('Other models')).toBeVisible();
	});
});
