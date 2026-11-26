import type { ModelOption } from '#lib/types/models.js';
import { pickQuickModels } from '#lib/utils/index.ts';
import { describe, expect, it } from 'vitest';

const option = (id: string) => ({ id, model: id, name: id }) as ModelOption;
const options = ['a', 'b', 'c', 'd', 'e'].map(option);
const ids = (items: { option: ModelOption }[]) => items.map((item) => item.option.id);

describe('pickQuickModels', () => {
	it('lists loaded, then favorite, then recent models, each once', () => {
		expect(
			ids(
				pickQuickModels({
					favoriteIds: new Set(['d', 'b']),
					limit: 6,
					loadedIds: new Set(['b']),
					options,
					recentIds: ['e', 'b', 'missing']
				})
			)
		).toEqual(['b', 'd', 'e']);
	});

	it("leads with the chat's model and the dictation model", () => {
		expect(
			ids(
				pickQuickModels({
					activeId: 'c',
					dictationModel: 'e',
					favoriteIds: new Set(['d']),
					limit: 6,
					loadedIds: new Set(['b', 'c']),
					options,
					recentIds: ['a']
				})
			)
		).toEqual(['c', 'e', 'b', 'd', 'a']);
	});

	it('is empty when nothing is in reach', () => {
		expect(
			pickQuickModels({
				favoriteIds: new Set(),
				limit: 6,
				loadedIds: new Set(),
				options,
				recentIds: []
			})
		).toEqual([]);
	});

	it('stops at the limit', () => {
		expect(
			ids(
				pickQuickModels({
					favoriteIds: new Set(),
					limit: 2,
					loadedIds: new Set(),
					options,
					recentIds: ['c', 'a', 'b']
				})
			)
		).toEqual(['c', 'a']);
	});
});
