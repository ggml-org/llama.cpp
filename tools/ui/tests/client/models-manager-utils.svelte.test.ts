// The manager table is built from the store's option list, so these run in the
// browser project: how the quants fold into rows, what context a row reports and
// how the tracked downloads become rows of their own.

import {
	downloadGroups,
	groupModelQuants,
	modelContextLength,
	splitHiddenQuants
} from '$lib/components/app/models/ModelsManager/utils';
import { LOCAL_BACKEND_ID } from '$lib/constants';
import { ModelGroupKind, ServerModelStatus } from '$lib/enums';
import { modelsStore } from '$lib/stores/models/index.svelte';
import type { ApiModelDataEntry, ModelOption } from '$lib/types';
import { beforeEach, describe, expect, it } from 'vitest';

function option(model: string, id = model): ModelOption {
	return { backendId: LOCAL_BACKEND_ID, capabilities: [], id, model, name: model };
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

beforeEach(() => {
	modelsStore.routerModels = [];
});

describe('groupModelQuants', () => {
	it('folds the quants of one repo into a single entry', () => {
		const groups = groupModelQuants([
			option('org/Qwen3-8B:Q4_K_M'),
			option('org/Qwen3-8B:Q8_0'),
			option('org/Llama-3-8B:Q4_K_M')
		]);

		expect(groups.map((group) => [group.key, group.kind, group.quants.length])).toEqual([
			[`${LOCAL_BACKEND_ID}::org/Qwen3-8B`, ModelGroupKind.QUANTS, 2],
			[`${LOCAL_BACKEND_ID}::org/Llama-3-8B`, ModelGroupKind.QUANTS, 1]
		]);
	});

	it('keeps the first quant of a repo as the row it renders', () => {
		const groups = groupModelQuants([option('org/Qwen3-8B:Q4_K_M'), option('org/Qwen3-8B:Q8_0')]);

		expect(groups[0].base.model).toBe('org/Qwen3-8B:Q4_K_M');
	});

	it('keeps rows of the same id apart as variants', () => {
		const groups = groupModelQuants([
			option('org/Qwen3-8B:Q4_K_M', 'first'),
			option('org/Qwen3-8B:Q4_K_M', 'second')
		]);

		expect(groups.map((group) => [group.key, group.kind])).toEqual([
			[`${LOCAL_BACKEND_ID}::org/Qwen3-8B::first`, ModelGroupKind.VARIANTS],
			[`${LOCAL_BACKEND_ID}::org/Qwen3-8B::second`, ModelGroupKind.VARIANTS]
		]);
	});
});

describe('modelContextLength', () => {
	it('reports the context of the listing', () => {
		expect(modelContextLength({ ...option('org/Qwen3-8B:Q4_K_M'), contextLength: 8192 })).toBe(
			8192
		);
	});

	it('reports nothing while no source knows the context', () => {
		expect(modelContextLength(option('org/Qwen3-8B:Q4_K_M'))).toBeNull();
	});
});

describe('downloadGroups', () => {
	it('reuses the row of a download the router already lists', () => {
		const listed = option('org/Qwen3-8B:Q4_K_M');
		const groups = downloadGroups(
			[{ isPaused: false, progress: null, repoWithTag: 'org/Qwen3-8B:Q4_K_M' }],
			[listed]
		);

		expect(groups[0].key).toBe('download::org/Qwen3-8B:Q4_K_M');
		expect(groups[0].base).toBe(listed);
	});

	it('stands the download in for a row of its own when nothing is listed', () => {
		const groups = downloadGroups(
			[{ isPaused: true, progress: null, repoWithTag: 'org/Qwen3-8B:Q4_K_M' }],
			[]
		);

		expect(groups[0].base).toMatchObject({
			backendId: LOCAL_BACKEND_ID,
			model: 'org/Qwen3-8B:Q4_K_M'
		});
	});
});

describe('modelsStore.isModelRunning', () => {
	it('counts a loaded or sleeping model as running', () => {
		modelsStore.routerModels = [
			routerEntry('org/loaded', ServerModelStatus.LOADED),
			routerEntry('org/sleeping', ServerModelStatus.SLEEPING)
		];

		expect(modelsStore.isModelRunning('org/loaded')).toBe(true);
		expect(modelsStore.isModelRunning('org/sleeping')).toBe(true);
	});

	it('does not count a loading, failed or unknown model as running', () => {
		modelsStore.routerModels = [
			routerEntry('org/loading', ServerModelStatus.LOADING),
			routerEntry('org/failed', ServerModelStatus.FAILED)
		];

		expect(modelsStore.isModelRunning('org/loading')).toBe(false);
		expect(modelsStore.isModelRunning('org/failed')).toBe(false);
		expect(modelsStore.isModelRunning('org/unknown')).toBe(false);
	});
});

describe('splitHiddenQuants', () => {
	const entry = (repo: string, quants: string[]) => ({
		base: option(quants[0]),
		key: repo,
		kind: ModelGroupKind.QUANTS,
		quants: quants.map((model) => option(model))
	});

	it('keeps the visible quants of a partly hidden repo in the local block', () => {
		const repo = entry('org/Qwen3-8B', ['org/Qwen3-8B:Q4_K_M', 'org/Qwen3-8B:Q8_0']);
		const { hidden, local } = splitHiddenQuants([repo], (picked) => picked.model.endsWith('Q8_0'));

		expect(hidden.map((group) => [group.key, group.quants.length])).toEqual([
			['org/Qwen3-8B::hidden', 1]
		]);
		expect(local.map((group) => [group.key, group.quants.length])).toEqual([['org/Qwen3-8B', 1]]);
		expect(local[0].base.model).toBe('org/Qwen3-8B:Q4_K_M');
	});

	it('moves a repo whose quants are all hidden', () => {
		const repo = entry('org/Qwen3-8B', ['org/Qwen3-8B:Q4_K_M', 'org/Qwen3-8B:Q8_0']);
		const { hidden, local } = splitHiddenQuants([repo], () => true);

		expect(hidden[0].quants).toHaveLength(2);
		expect(local).toEqual([]);
	});

	it('leaves a repo with no hidden quant alone', () => {
		const repo = entry('org/Qwen3-8B', ['org/Qwen3-8B:Q4_K_M']);
		const { hidden, local } = splitHiddenQuants([repo], () => false);

		expect(hidden).toEqual([]);
		expect(local[0].key).toBe('org/Qwen3-8B');
	});
});
