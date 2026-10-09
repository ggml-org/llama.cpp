// Guards the manager table ordering and filtering: the context column sorts and
// filters once a model's context is known, from the option or the Hub record.

import ModelsManagerWrapper from './components/ModelsManagerWrapper.svelte';
import { SETTINGS_KEYS } from '$lib/constants';
import { ServerModelStatus } from '$lib/enums';
import { HuggingFaceService } from '$lib/services';
import { modelsStore, settingsStore } from '$lib/stores';
import type { ApiModelDataEntry } from '$lib/types';
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
	modelsStore.favoriteModelIds = new Set();
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

it('lists the favorited quants of a repo as flat rows', async () => {
	// one quant of the beta repo is a favorite, the other stays with the repo
	modelsStore.favoriteModelIds = new Set(['org/alpha-8b:Q4_K_M', 'org/beta-8b:Q4_K_M']);

	const screen = await renderWithModels([
		modelsWithoutContext[0],
		modelsWithoutContext[1],
		option('org/beta-8b:Q8_0'),
		modelsWithoutContext[2]
	]);

	// the favorite quant is a model row of its own, not a repo with subitems
	expect(screen.container.textContent).not.toContain('2 quants available');

	// the repo appears once in favorites for its favorited quant and once in the
	// local block for the quant left behind
	expect(screen.getByText(/beta\s+8B/).elements().length).toBe(2);
});

/** A router listing entry carrying only the meta context. */
function entry(model: string, nCtxTrain: number): ApiModelDataEntry {
	return {
		created: 0,
		id: model,
		in_cache: false,
		meta: { n_ctx_train: nCtxTrain },
		object: 'model',
		owned_by: 'llamacpp',
		path: `/models/${model}`,
		status: { value: ServerModelStatus.UNLOADED }
	};
}

it('sorts by the meta context of a listing without the router field', async () => {
	// a listing that skips the router's GGUF read reports the trained context
	// only as meta.n_ctx_train, so the option mapping falls back to it
	vi.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL) => {
		const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;

		if (url.includes('/props')) {
			return new Response(
				JSON.stringify({
					default_generation_settings: { n_ctx: 0, params: {} },
					model_alias: 'llama-server',
					model_path: 'none',
					role: 'router'
				}),
				{ headers: { 'Content-Type': 'application/json' }, status: 200 }
			);
		}

		if (url.includes('/server')) {
			return new Response(
				JSON.stringify({ git_branch: 'test', git_commit: 'test', mode: 'router', version: 'test' }),
				{ headers: { 'Content-Type': 'application/json' }, status: 200 }
			);
		}

		if (/\/v1\/models|\/models\b/.test(url)) {
			return new Response(
				JSON.stringify({
					data: [
						entry('org/alpha-8b:Q4_K_M', 8192),
						entry('org/beta-8b:Q4_K_M', 131072),
						entry('org/gamma-8b:Q4_K_M', 32768)
					],
					object: 'list'
				}),
				{ headers: { 'Content-Type': 'application/json' }, status: 200 }
			);
		}

		throw new Error(`unexpected fetch in the test: ${url}`);
	});

	const screen = render(ModelsManagerWrapper);

	// the fetch maps the listing into options, the meta context fills in
	await modelsStore.fetch(true);

	await expect.element(screen.getByText(/gamma\s+8B/)).toBeVisible();

	await screen.getByTitle('Sort by context, lowest first').click();

	const names = rowNames(screen.container).join(' | ');

	expect(names.indexOf('alpha')).toBeLessThan(names.indexOf('gamma'));
	expect(names.indexOf('gamma')).toBeLessThan(names.indexOf('beta'));
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
