// Guards the phone layout of the manager: a row offers its actions in a drawer there,
// since a dropdown of small rows is not a touch target.

import ModelsManagerRowWrapper from './components/ModelsManagerRowWrapper.svelte';
import { ServerRole } from '$lib/enums';
import { modelsStore } from '$lib/stores/models/index.svelte';
import { serverStore } from '$lib/stores/server.svelte';
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

describe('manager model row on a phone', () => {
	beforeEach(async () => {
		// the row only offers a load control on a router server
		serverStore.role = ServerRole.ROUTER;
		modelsStore.favoriteModelIds = new Set();
		await page.viewport(PHONE.width, PHONE.height);
	});

	afterEach(async () => {
		await page.viewport(DESKTOP.width, DESKTOP.height);
	});

	it('offers the row actions in a drawer', async () => {
		const screen = render(ModelsManagerRowWrapper, {
			onSelect: () => {},
			option,
			selected: false
		});

		await screen.getByRole('button', { exact: true, name: 'Model actions' }).click();

		await expect.element(screen.getByRole('button', { name: 'Add to favorites' })).toBeVisible();
	});

	it('runs an action once the drawer has closed', async () => {
		const screen = render(ModelsManagerRowWrapper, {
			onSelect: () => {},
			option,
			selected: false
		});

		await screen.getByRole('button', { exact: true, name: 'Model actions' }).click();
		await screen.getByRole('button', { name: 'Add to favorites' }).click();

		await expect.poll(() => modelsStore.favoriteModelIds.has(option.model)).toBe(true);
		await expect
			.poll(() => screen.getByRole('button', { name: 'Add to favorites' }).query())
			.toBeNull();
	});
});
