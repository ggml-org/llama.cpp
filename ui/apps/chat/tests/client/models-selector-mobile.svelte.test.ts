// Guards the phone model picker: the selector serves a phone a bottom sheet with the
// desktop menu's search and sections, opening on the models in reach when there are any.

import ModelsSelector from '#lib/components/app/models/ModelsSelector/ModelsSelector.svelte';
import { ServerRole } from '#lib/enums/index.ts';
import { ModelsService } from '#lib/services/models.service.ts';
import { serverStore } from '#lib/stores/llama-server.svelte.ts';
import { modelsStore } from '#lib/stores/models/index.svelte.ts';
import type { ModelOption } from '#lib/types/models.js';
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
const other: ModelOption = {
	capabilities: [],
	id: 'org/Qwen3-8B:Q8_0',
	model: 'org/Qwen3-8B:Q8_0',
	name: 'Qwen3-8B-Q8_0'
};

describe('model picker drawer on a phone', () => {
	beforeEach(async () => {
		serverStore.role = ServerRole.ROUTER;
		// the picker mounts a fetch of its own: serve it the same two models, so
		// the list it rebuilds does not clear the seeded one mid-test
		vi.spyOn(ModelsService, 'list').mockResolvedValue({
			data: [{ id: option.model }, { id: other.model }]
		} as never);
		modelsStore.models = [option, other];
		await modelsStore.selectModelById(option.id);
		await page.viewport(PHONE.width, PHONE.height);
	});

	afterEach(async () => {
		modelsStore.models = [];
		modelsStore.recentModelIds = [];
		vi.restoreAllMocks();
		await page.viewport(DESKTOP.width, DESKTOP.height);
	});

	it("opens on the short view with the chat's model, even with no other in reach", async () => {
		const screen = render(ModelsSelector);

		await screen.getByRole('button').click();

		await expect.element(screen.getByRole('button', { name: 'All models' })).toBeVisible();
		await expect.element(screen.getByText('Q4_K_M')).toBeVisible();
		await expect.element(screen.getByText('Q8_0')).not.toBeInTheDocument();
		await expect.element(screen.getByPlaceholder('Search models')).toBeVisible();
	});

	it("lists the chat's model and a recent one, and all of them on their page", async () => {
		modelsStore.recentModelIds = [other.id];

		const screen = render(ModelsSelector);

		await screen.getByRole('button').click();

		await expect.element(screen.getByText('Q4_K_M')).toBeVisible();
		await expect.element(screen.getByText('Q8_0')).toBeVisible();
		await expect.element(screen.getByPlaceholder('Search models')).toBeVisible();

		await screen.getByRole('button', { name: 'All models' }).click();

		await expect.element(screen.getByRole('heading', { name: 'All models' })).toBeVisible();
		await expect.element(screen.getByText('Q4_K_M')).toBeVisible();
	});
});
