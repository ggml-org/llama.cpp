import type { ModelOption } from '$lib/types/models';
import {
	filterModelOptions,
	type GroupedModelOptions,
	groupModelOptions,
	windowLocalGroups
} from '$lib/utils/model-list';
import { describe, expect, it } from 'vitest';

function option(model: string, orgName: string | null = null): ModelOption {
	return {
		capabilities: [],
		id: model,
		model,
		name: model,
		parsedId: {
			activatedParams: null,
			modelName: model,
			orgName,
			params: null,
			quantization: null,
			raw: model,
			sidecar: null,
			tags: []
		}
	};
}

const row = (model: string) => ({ option: option(model) });

function groups(): GroupedModelOptions {
	return {
		available: [
			{ items: [row('org/a'), row('org/b'), row('org/c')], orgName: 'org' },
			{ items: [row('other/d')], orgName: 'other' }
		],
		providers: []
	};
}

describe('windowLocalGroups', () => {
	it('fills the budget in listed order', () => {
		const windowed = windowLocalGroups(groups(), 2);

		expect(windowed.available.flatMap((group) => group.items).map((item) => item.option.model)) //
			.toEqual(['org/a', 'org/b']);
		expect(windowed.shown).toBe(2);
	});

	it('drops an org whose whole budget is spent', () => {
		const windowed = windowLocalGroups(groups(), 3);

		expect(windowed.available.map((group) => group.orgName)).toEqual(['org']);
		expect(windowed.shown).toBe(3);
	});

	it('keeps every group when the budget covers the list', () => {
		const windowed = windowLocalGroups(groups(), 99);

		expect(windowed.available.map((group) => group.orgName)).toEqual(['org', 'other']);
		expect(windowed.shown).toBe(4);
	});

	it('shows nothing on a zero budget', () => {
		const windowed = windowLocalGroups(groups(), 0);

		expect(windowed.available).toEqual([]);
		expect(windowed.shown).toBe(0);
	});
});

describe('filterModelOptions', () => {
	const options = [
		{ ...option('org/Qwen3-8B'), aliases: ['qwen'], tags: ['vision'] },
		option('org/Llama-3-8B')
	];

	it('keeps every option without a term', () => {
		expect(filterModelOptions(options, '  ')).toEqual(options);
	});

	it('matches the model id and the name', () => {
		expect(filterModelOptions(options, 'llama').map((item) => item.model)).toEqual([
			'org/Llama-3-8B'
		]);
	});

	it('matches aliases and tags', () => {
		expect(filterModelOptions(options, 'qwen').map((item) => item.model)).toEqual(['org/Qwen3-8B']);
	});
});

describe('groupModelOptions', () => {
	it('groups the options by org, unnamed last', () => {
		const grouped = groupModelOptions([
			option('org/a', 'org'),
			option('org/b', 'org'),
			option('other/c', 'other'),
			option('plain')
		]);

		expect(grouped.available.map((group) => group.orgName)).toEqual(['org', 'other', null]);
		expect(grouped.available[0].items.map((item) => item.option.model)).toEqual(['org/a', 'org/b']);
	});
});
