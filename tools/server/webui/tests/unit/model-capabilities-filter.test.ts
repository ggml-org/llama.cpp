import { describe, expect, it } from 'vitest';
import type { ModelOption } from '$lib/types/models';

/**
 * Tests for capability-based model filtering in the chat model selector.
 *
 * The ModelsSelector component filters models based on their `capabilities` array:
 * - Models with capabilities that include "completion" are shown
 * - Models with capabilities that do NOT include "completion" are hidden
 * - Models with empty capabilities (legacy/backwards compat) are shown
 *
 * This mirrors the filtering logic in ModelsSelector.svelte (lines 45-58).
 */

function makeModel(overrides: Partial<ModelOption> & { id: string; model: string }): ModelOption {
	return {
		name: overrides.model.split('/').pop() || overrides.model,
		capabilities: [],
		...overrides,
	};
}

/**
 * Pure implementation of the capability filter from ModelsSelector.svelte.
 * We test this logic in isolation since Svelte component tests require
 * the browser environment and full store setup.
 */
function filterByCapabilities(
	options: ModelOption[],
	getModelProps?: (model: string) => { webui?: boolean } | undefined
): ModelOption[] {
	return options.filter((option) => {
		const modelProps = getModelProps?.(option.model);
		if (modelProps?.webui === false) return false;

		// Hide embedding/rerank-only models from chat selector
		// If capabilities are present, require "completion" capability
		// If capabilities are absent (legacy), show the model (backwards compat)
		if (option.capabilities.length > 0 && !option.capabilities.includes('completion')) {
			return false;
		}

		return true;
	});
}

describe('model capability filtering', () => {
	describe('happy paths', () => {
		it('shows a model with completion capability', () => {
			const models = [
				makeModel({ id: 'chat-model', model: 'org/chat-model', capabilities: ['completion'] }),
			];
			const result = filterByCapabilities(models);
			expect(result).toHaveLength(1);
			expect(result[0].id).toBe('chat-model');
		});

		it('hides a model with only embedding capability', () => {
			const models = [
				makeModel({ id: 'embed-model', model: 'org/embed-model', capabilities: ['embedding'] }),
			];
			const result = filterByCapabilities(models);
			expect(result).toHaveLength(0);
		});

		it('hides a model with embedding and rerank capabilities (no completion)', () => {
			const models = [
				makeModel({
					id: 'rerank-model',
					model: 'org/rerank-model',
					capabilities: ['embedding', 'rerank'],
				}),
			];
			const result = filterByCapabilities(models);
			expect(result).toHaveLength(0);
		});

		it('shows a model with empty capabilities (legacy/backwards compat)', () => {
			const models = [
				makeModel({ id: 'legacy-model', model: 'org/legacy-model', capabilities: [] }),
			];
			const result = filterByCapabilities(models);
			expect(result).toHaveLength(1);
			expect(result[0].id).toBe('legacy-model');
		});

		it('filters a mixed list of models correctly', () => {
			const models = [
				makeModel({ id: 'chat1', model: 'org/llama-3', capabilities: ['completion'] }),
				makeModel({ id: 'embed1', model: 'org/bge-small', capabilities: ['embedding'] }),
				makeModel({ id: 'chat2', model: 'org/gemma-3', capabilities: ['completion'] }),
				makeModel({
					id: 'rerank1',
					model: 'org/jina-reranker',
					capabilities: ['embedding', 'rerank'],
				}),
				makeModel({ id: 'legacy1', model: 'org/old-model', capabilities: [] }),
			];
			const result = filterByCapabilities(models);
			expect(result).toHaveLength(3);
			expect(result.map((m) => m.id)).toEqual(['chat1', 'chat2', 'legacy1']);
		});
	});

	describe('edge cases', () => {
		it('returns empty array when given empty input', () => {
			const result = filterByCapabilities([]);
			expect(result).toHaveLength(0);
		});

		it('returns empty array when all models are embedding-only', () => {
			const models = [
				makeModel({ id: 'embed1', model: 'org/embed-1', capabilities: ['embedding'] }),
				makeModel({ id: 'embed2', model: 'org/embed-2', capabilities: ['embedding'] }),
			];
			const result = filterByCapabilities(models);
			expect(result).toHaveLength(0);
		});

		it('shows model when capabilities is empty (no capabilities from server)', () => {
			const models = [
				makeModel({ id: 'no-cap', model: 'org/model', capabilities: [] }),
			];
			const result = filterByCapabilities(models);
			expect(result).toHaveLength(1);
		});

		it('hides model with only rerank capability (no embedding)', () => {
			// Edge case: rerank without embedding listed
			const models = [
				makeModel({ id: 'rerank-only', model: 'org/reranker', capabilities: ['rerank'] }),
			];
			const result = filterByCapabilities(models);
			expect(result).toHaveLength(0);
		});

		it('shows model with completion among other capabilities', () => {
			// Future-proofing: model has multiple capabilities including completion
			const models = [
				makeModel({
					id: 'multi',
					model: 'org/multi-model',
					capabilities: ['completion', 'embedding'],
				}),
			];
			const result = filterByCapabilities(models);
			expect(result).toHaveLength(1);
			expect(result[0].id).toBe('multi');
		});

		it('handles unknown capability strings gracefully', () => {
			// Model with unrecognized capability but no "completion"
			const models = [
				makeModel({
					id: 'unknown-cap',
					model: 'org/experimental',
					capabilities: ['transcription'],
				}),
			];
			const result = filterByCapabilities(models);
			expect(result).toHaveLength(0);
		});

		it('treats capabilities check as case-sensitive', () => {
			// "Completion" (capitalized) should NOT match "completion"
			const models = [
				makeModel({
					id: 'wrong-case',
					model: 'org/model',
					capabilities: ['Completion'],
				}),
			];
			const result = filterByCapabilities(models);
			expect(result).toHaveLength(0);
		});
	});

	describe('webui property interaction', () => {
		it('hides model when webui is false even with completion capability', () => {
			const models = [
				makeModel({ id: 'hidden', model: 'org/hidden', capabilities: ['completion'] }),
			];
			const getProps = () => ({ webui: false });
			const result = filterByCapabilities(models, getProps);
			expect(result).toHaveLength(0);
		});

		it('shows model when webui is true and has completion capability', () => {
			const models = [
				makeModel({ id: 'visible', model: 'org/visible', capabilities: ['completion'] }),
			];
			const getProps = () => ({ webui: true });
			const result = filterByCapabilities(models, getProps);
			expect(result).toHaveLength(1);
		});

		it('hides embedding model even when webui is not set', () => {
			const models = [
				makeModel({ id: 'embed', model: 'org/embed', capabilities: ['embedding'] }),
			];
			const getProps = () => undefined;
			const result = filterByCapabilities(models, getProps);
			expect(result).toHaveLength(0);
		});

		it('hides model with webui=false and no capabilities (webui takes precedence)', () => {
			const models = [
				makeModel({ id: 'no-webui', model: 'org/internal', capabilities: [] }),
			];
			const getProps = () => ({ webui: false });
			const result = filterByCapabilities(models, getProps);
			expect(result).toHaveLength(0);
		});

		it('applies both webui and capabilities filters together', () => {
			const models = [
				makeModel({ id: 'chat', model: 'org/chat', capabilities: ['completion'] }),
				makeModel({ id: 'embed', model: 'org/embed', capabilities: ['embedding'] }),
				makeModel({ id: 'hidden', model: 'org/hidden', capabilities: ['completion'] }),
				makeModel({ id: 'legacy', model: 'org/legacy', capabilities: [] }),
			];
			const getProps = (model: string) => {
				if (model === 'org/hidden') return { webui: false };
				return undefined;
			};
			const result = filterByCapabilities(models, getProps);
			expect(result).toHaveLength(2);
			expect(result.map((m) => m.id)).toEqual(['chat', 'legacy']);
		});
	});

	describe('capability parsing from API response', () => {
		it('filters non-string values from raw capabilities', () => {
			// The store filters capabilities with: rawCapabilities.filter(v => Boolean(v))
			// Simulating what the store produces after filtering
			const rawCapabilities: unknown[] = ['embedding', null, undefined, '', 0, false];
			const filtered = rawCapabilities.filter(
				(value: unknown): value is string => Boolean(value)
			);
			expect(filtered).toEqual(['embedding']);
		});

		it('produces empty capabilities when API returns no capabilities field', () => {
			const details = undefined;
			const rawCapabilities = Array.isArray(details) ? details : [];
			expect(rawCapabilities).toEqual([]);
		});

		it('produces empty capabilities when API returns null capabilities', () => {
			const details = { capabilities: null };
			const rawCapabilities = Array.isArray(details.capabilities)
				? details.capabilities
				: [];
			expect(rawCapabilities).toEqual([]);
		});

		it('correctly extracts capabilities when API returns valid array', () => {
			const details = { capabilities: ['completion'] };
			const rawCapabilities = Array.isArray(details.capabilities)
				? details.capabilities
				: [];
			const capabilities = rawCapabilities.filter(
				(value: unknown): value is string => Boolean(value)
			);
			expect(capabilities).toEqual(['completion']);
		});

		it('correctly extracts multiple capabilities', () => {
			const details = { capabilities: ['embedding', 'rerank'] };
			const rawCapabilities = Array.isArray(details.capabilities)
				? details.capabilities
				: [];
			const capabilities = rawCapabilities.filter(
				(value: unknown): value is string => Boolean(value)
			);
			expect(capabilities).toEqual(['embedding', 'rerank']);
		});
	});

	describe('ordering preservation', () => {
		it('preserves original order of models after filtering', () => {
			const models = [
				makeModel({ id: 'z-model', model: 'org/z-chat', capabilities: ['completion'] }),
				makeModel({ id: 'embed', model: 'org/embed', capabilities: ['embedding'] }),
				makeModel({ id: 'a-model', model: 'org/a-chat', capabilities: ['completion'] }),
				makeModel({ id: 'rerank', model: 'org/rerank', capabilities: ['embedding', 'rerank'] }),
				makeModel({ id: 'm-model', model: 'org/m-legacy', capabilities: [] }),
			];
			const result = filterByCapabilities(models);
			expect(result.map((m) => m.id)).toEqual(['z-model', 'a-model', 'm-model']);
		});
	});

	describe('single model remaining after filter', () => {
		it('returns only the completion model when mixed with embedding models', () => {
			const models = [
				makeModel({ id: 'embed1', model: 'org/embed-1', capabilities: ['embedding'] }),
				makeModel({ id: 'embed2', model: 'org/embed-2', capabilities: ['embedding'] }),
				makeModel({ id: 'chat', model: 'org/chat', capabilities: ['completion'] }),
				makeModel({
					id: 'rerank',
					model: 'org/reranker',
					capabilities: ['embedding', 'rerank'],
				}),
			];
			const result = filterByCapabilities(models);
			expect(result).toHaveLength(1);
			expect(result[0].id).toBe('chat');
		});
	});
});
