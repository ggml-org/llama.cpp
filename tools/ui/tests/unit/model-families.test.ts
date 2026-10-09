import { groupModelFamilies, modelFamilyKey } from '$lib/utils/model-families';
import { describe, expect, it } from 'vitest';

describe('modelFamilyKey', () => {
	it('folds the leading letters of the model name', () => {
		expect(modelFamilyKey('Qwen3.5-27B')).toBe('Qwen');
		expect(modelFamilyKey('Qwen3.8-27B-GGUF')).toBe('Qwen');
		expect(modelFamilyKey('Llama-3.2-3B-Instruct')).toBe('Llama');
	});

	it('uses the model and not the org', () => {
		expect(modelFamilyKey('ggml-org/Qwen3.8-27B-GGUF')).toBe('Qwen');
	});

	it('falls back to the first segment when the name starts with a digit', () => {
		expect(modelFamilyKey('3.5-MoE-instruct')).toBe('3');
	});

	it('keeps a name with no letters whole', () => {
		expect(modelFamilyKey('_internal')).toBe('_internal');
		expect(modelFamilyKey('-leading-dash')).toBe('-leading-dash');
	});
});

describe('groupModelFamilies', () => {
	const rows = [
		{ name: 'ggml-org/Qwen3.5-9B-GGUF' },
		{ name: 'unsloth/Qwen3.8-27B-GGUF' },
		{ name: 'meta-llama/Llama-3.2-3B-Instruct' },
		{ name: 'org/tinyllama-1.1B' }
	];
	const modelOf = (row: { name: string }) => row.name;

	it('collects the entries of one family', () => {
		const families = groupModelFamilies(rows, modelOf);

		expect(families.map((family) => [family.key, family.entries.length])).toEqual([
			['Qwen', 2],
			['Llama', 1],
			['tinyllama', 1]
		]);
	});

	it('labels a family with its key', () => {
		expect(groupModelFamilies(rows, modelOf)[0].label).toBe('Qwen');
	});

	it('keeps the input order of the entries', () => {
		expect(groupModelFamilies(rows, modelOf)[0].entries.map(modelOf)).toEqual([
			'ggml-org/Qwen3.5-9B-GGUF',
			'unsloth/Qwen3.8-27B-GGUF'
		]);
	});
});
