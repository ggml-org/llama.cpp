// Guards the manager table ordering and filtering: the context column sorts and
// filters once a model's context is known, from the option or the Hub record.

import ModelsManagerWrapper from './components/ModelsManagerWrapper.svelte';
import { SETTINGS_KEYS } from '$lib/constants';
import { HuggingFaceService } from '$lib/services';
import { modelsStore, settingsStore } from '$lib/stores';
import type { ModelOption } from '$lib/types/models';
import { SvelteMap } from 'svelte/reactivity';
import { beforeEach, expect, it, vi } from 'vitest';
import { render } from 'vitest-browser-svelte';

function option(model: string, contextLength?: number): ModelOption {
	return {
		capabilities: [],
		contextLength,
		id: model,
		model,
		name: model
	};
}

const models = [
	option('org/alpha-8b:Q4_K_M', 8192),
	option('org/beta-8b:Q4_K_M', 131072),
	option('org/gamma-8b:Q4_K_M', 32768)
];
// the router listing carries no context, so a row starts without one
const modelsWithoutContext = models.map((model) => ({ ...model, contextLength: undefined }));

beforeEach(() => {
	modelsStore.routerModels = [];
	modelsStore.models = modelsWithoutContext;
	settingsStore.config[SETTINGS_KEYS.GROUP_MODELS_BY_FAMILY] = false;
});

/** Renders the manager and waits for the test models to show. */
async function renderWithModels(rows: ModelOption[] = modelsWithoutContext) {
	const screen = render(ModelsManagerWrapper);

	modelsStore.models = rows;

	await expect.element(screen.getByText(/gamma\s+8B/)).toBeVisible();

	return screen;
}

function rowNames(container: HTMLElement): string[] {
	return [...container.querySelectorAll('button[aria-pressed]')].map(
		(row) => row.textContent ?? ''
	);
}

/** The Hub details cache the rows read their context from. */
function detailsCache() {
	return (
		HuggingFaceService as unknown as {
			detailsCache: SvelteMap<string, { gguf?: { context_length?: number } } | null>;
		}
	).detailsCache;
}

function warmCache() {
	const cache = detailsCache();

	cache.set('org/alpha-8b', { gguf: { context_length: 8192 } });
	cache.set('org/beta-8b', { gguf: { context_length: 131072 } });
	cache.set('org/gamma-8b', { gguf: { context_length: 32768 } });
}

it('sorts by context', async () => {
	const screen = await renderWithModels(models);

	// lowest first
	await screen.getByTitle('Sort by context, lowest first').click();

	const ascending = rowNames(screen.container);

	await screen.getByTitle('Sort by context, highest first').click();

	const descending = rowNames(screen.container);

	expect(ascending).not.toEqual(descending);
});

it('sorts by context with family grouping on', async () => {
	settingsStore.config[SETTINGS_KEYS.GROUP_MODELS_BY_FAMILY] = true;

	const screen = await renderWithModels(models);

	// lowest first
	await screen.getByTitle('Sort by context, lowest first').click();

	const ascending = rowNames(screen.container);

	await screen.getByTitle('Sort by context, highest first').click();

	const descending = rowNames(screen.container);

	expect(ascending).not.toEqual(descending);
});

it('re-sorts when the Hub details arrive after the sort was clicked', async () => {
	const screen = await renderWithModels();

	// the user sorts while the contexts are still unknown
	await screen.getByTitle('Sort by context, lowest first').click();

	// then the rows fetch their Hub records
	warmCache();

	// the table re-sorts once the cache answers
	await vi.waitFor(() => {
		const names = rowNames(screen.container).join(' | ');

		expect(names.indexOf('alpha')).toBeLessThan(names.indexOf('gamma'));
		expect(names.indexOf('gamma')).toBeLessThan(names.indexOf('beta'));

		return names;
	});
});

it('filters by search', async () => {
	const screen = await renderWithModels();

	await screen.getByPlaceholder('Search your models').fill('beta');

	await expect.element(screen.getByText(/alpha\s+8B/)).not.toBeVisible();
	await expect.element(screen.getByText(/beta\s+8B/)).toBeVisible();
});

it('filters by context and sorts from the Hub details cache', async () => {
	warmCache();

	const screen = await renderWithModels();

	// open the context filter and ask for 32K or more
	await screen.getByText('Context:').click();
	await screen.getByText('32K or more').click();

	await expect.element(screen.getByText(/alpha\s+8B/)).not.toBeVisible();
	await expect.element(screen.getByText(/gamma\s+8B/)).toBeVisible();

	// sorting re-runs once the cache answers
	await screen.getByTitle('Sort by context, lowest first').click();

	const names = rowNames(screen.container).join(' | ');

	expect(names.indexOf('gamma')).toBeLessThan(names.indexOf('beta'));
});
