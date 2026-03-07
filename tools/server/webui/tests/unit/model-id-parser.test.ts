import { describe, expect, it } from 'vitest';
import { ModelsService } from '$lib/services/models.service';

const { parseModelId } = ModelsService;

describe('parseModelId', () => {
	it('handles unknown patterns correctly', () => {
		expect(parseModelId('model-name-1')).toStrictEqual({
			activatedParams: null,
			format: null,
			modelName: 'model-name-1',
			orgName: null,
			params: null,
			quantization: null,
			raw: 'model-name-1',
			tags: []
		});

		expect(parseModelId('org/model-name-2')).toStrictEqual({
			activatedParams: null,
			format: null,
			modelName: 'model-name-2',
			orgName: 'org',
			params: null,
			quantization: null,
			raw: 'org/model-name-2',
			tags: []
		});
	});

	it('extracts model format correctly', () => {
		expect(parseModelId('model-100B-Q4_K_M')).toMatchObject({ format: null });
		expect(parseModelId('model-100B-F16')).toMatchObject({ format: null });
		expect(parseModelId('model-100B-BF16')).toMatchObject({ format: 'BF16' });
		expect(parseModelId('model-100B-MXFP4')).toMatchObject({ format: 'MXFP4' });
	});

	// TODO: fix parser
	it.skip('extracts model format correctly in lowercase', () => {
		expect(parseModelId('model-100b-mxfp4')).toMatchObject({ format: 'MXFP4' });
	});

	it('extracts model parameters correctly', () => {
		expect(parseModelId('model-100B-BF16')).toMatchObject({ params: '100B' });
		expect(parseModelId('model-100B:Q4_K_M')).toMatchObject({ params: '100B' });
	});

	it('extracts model parameters correctly in lowercase', () => {
		expect(parseModelId('model-100b-bf16')).toMatchObject({ params: '100B' });
		expect(parseModelId('model-100b:q4_k_m')).toMatchObject({ params: '100B' });
	});

	it('extracts activated parameters correctly', () => {
		expect(parseModelId('model-100B-A10B-BF16')).toMatchObject({ activatedParams: 'A10B' });
		expect(parseModelId('model-100B-A10B:Q4_K_M')).toMatchObject({ activatedParams: 'A10B' });
	});

	// TODO: fix parser
	it.skip('extracts activated parameters correctly in lowercase', () => {
		expect(parseModelId('model-100b-a10b-bf16')).toMatchObject({ activatedParams: 'A10B' });
		expect(parseModelId('model-100b-a10b:q4_k_m')).toMatchObject({ activatedParams: 'A10B' });
	});

	it('extracts quantization correctly', () => {
		// TODO: fix parser
		// expect(parseModelId('model-100B-UD-IQ1_S')).toMatchObject({ quantization: 'UD-IQ1_S' });
		// expect(parseModelId('model-100B-IQ4_XS')).toMatchObject({ quantization: 'IQ4_XS' });
		// expect(parseModelId('model-100B-Q4_K_M')).toMatchObject({ quantization: 'Q4_K_M' });
		// expect(parseModelId('model-100B-Q8_0')).toMatchObject({ quantization: 'Q8_0' });
		// expect(parseModelId('model-100B-UD-Q8_K_XL')).toMatchObject({ quantization: 'UD-Q8_K_XL' });
		// expect(parseModelId('model-100B-F16')).toMatchObject({ quantization: 'F16' });
		// expect(parseModelId('model-100B-BF16')).toMatchObject({ quantization: 'BF16' });
		// expect(parseModelId('model-100B-MXFP4')).toMatchObject({ quantization: 'MXFP4' });
		expect(parseModelId('model-100B:UD-IQ1_S')).toMatchObject({ quantization: 'UD-IQ1_S' });
		expect(parseModelId('model-100B:IQ4_XS')).toMatchObject({ quantization: 'IQ4_XS' });
		expect(parseModelId('model-100B:Q4_K_M')).toMatchObject({ quantization: 'Q4_K_M' });
		expect(parseModelId('model-100B:Q8_0')).toMatchObject({ quantization: 'Q8_0' });
		expect(parseModelId('model-100B:UD-Q8_K_XL')).toMatchObject({ quantization: 'UD-Q8_K_XL' });
		expect(parseModelId('model-100B:F16')).toMatchObject({ quantization: 'F16' });
		expect(parseModelId('model-100B:BF16')).toMatchObject({ quantization: 'BF16' });
		expect(parseModelId('model-100B:MXFP4')).toMatchObject({ quantization: 'MXFP4' });
	});

	it('extracts additional tags correctly', () => {
		// TODO: fix parser
		// expect(parseModelId('model-100B-foobar-Q4_K_M')).toMatchObject({ tags: ['foobar'] });
		expect(parseModelId('model-100B-A10B-foobar-1M-BF16')).toMatchObject({
			tags: ['foobar', '1M']
		});
		expect(parseModelId('model-100B-1M-foobar:UD-Q8_K_XL')).toMatchObject({
			tags: ['1M', 'foobar']
		});
	});

	it('handles real-world examples correctly', () => {
		expect(parseModelId('meta-llama/Llama-3.1-8B')).toStrictEqual({
			activatedParams: null,
			format: null,
			modelName: 'Llama-3.1',
			orgName: 'meta-llama',
			params: '8B',
			quantization: null,
			raw: 'meta-llama/Llama-3.1-8B',
			tags: []
		});

		expect(parseModelId('openai/gpt-oss-120b-MXFP4')).toStrictEqual({
			activatedParams: null,
			format: 'MXFP4',
			modelName: 'gpt-oss',
			orgName: 'openai',
			params: '120B',
			quantization: null,
			raw: 'openai/gpt-oss-120b-MXFP4',
			tags: []
		});

		expect(parseModelId('openai/gpt-oss-20b:Q4_K_M')).toStrictEqual({
			activatedParams: null,
			format: null,
			modelName: 'gpt-oss',
			orgName: 'openai',
			params: '20B',
			quantization: 'Q4_K_M',
			raw: 'openai/gpt-oss-20b:Q4_K_M',
			tags: []
		});

		expect(parseModelId('Qwen/Qwen3-Coder-30B-A3B-Instruct-1M-BF16')).toStrictEqual({
			activatedParams: 'A3B',
			format: 'BF16',
			modelName: 'Qwen3-Coder',
			orgName: 'Qwen',
			params: '30B',
			// purpose: ['coder', 'instruct'],
			quantization: null,
			raw: 'Qwen/Qwen3-Coder-30B-A3B-Instruct-1M-BF16',
			tags: ['Instruct', '1M']
		});
	});

	// TODO: fix and merge working tests into one block
	it.skip('handles real-world examples correctly', () => {
		expect(parseModelId('meta-llama/Llama-4-Scout-17B-16E-Instruct-Q4_K_M')).toStrictEqual({
			activatedParams: null,
			format: null,
			modelName: 'Llama-4-Scout',
			orgName: 'meta-llama',
			params: '17B',
			quantization: 'Q4_K_M',
			raw: 'meta-llama/Llama-4-Scout-17B-16E-Instruct-Q4_K_M',
			tags: ['16E', 'Instruct']
		});
	});

	// TODO: fix and merge working tests into one block
	it.skip('handles real-world examples correctly', () => {
		expect(parseModelId('MiniMaxAI/MiniMax-M2-IQ4_XS')).toStrictEqual({
			activatedParams: null,
			format: null,
			modelName: 'MiniMax-M2',
			orgName: 'MiniMaxAI',
			params: null,
			quantization: 'IQ4_XS',
			raw: 'MiniMaxAI/MiniMax-M2-IQ4_XS',
			tags: []
		});
	});

	// TODO: fix and merge working tests into one block
	it.skip('handles real-world examples correctly', () => {
		expect(parseModelId('MiniMaxAI/MiniMax-M2-UD-Q3_K_XL')).toStrictEqual({
			activatedParams: null,
			format: null,
			modelName: 'MiniMax-M2',
			orgName: 'MiniMaxAI',
			params: null,
			quantization: 'UD-Q3_K_XL',
			raw: 'MiniMaxAI/MiniMax-M2-UD-Q3_K_XL',
			tags: []
		});
	});

	// TODO: fix and merge working tests into one block
	it.skip('handles real-world examples correctly', () => {
		expect(parseModelId('mistralai/Devstral-2-123B-Instruct-2512-Q4_K_M')).toStrictEqual({
			activatedParams: null,
			format: null,
			modelName: 'Devstral-2',
			orgName: 'mistralai',
			params: '123B',
			// purpose: 'instruct',
			quantization: 'Q4_K_M',
			raw: 'mistralai/Devstral-2-123B-Instruct-2512-Q4_K_M',
			tags: ['Instruct', '2512']
		});
	});

	// TODO: fix and merge working tests into one block
	it.skip('handles real-world examples correctly', () => {
		expect(parseModelId('mistralai/Devstral-Small-2-24B-Instruct-2512-Q8_0')).toStrictEqual({
			activatedParams: null,
			format: null,
			modelName: 'Devstral-Small-2',
			orgName: 'mistralai',
			params: '24B',
			quantization: 'Q8_0',
			raw: 'mistralai/Devstral-Small-2-24B-Instruct-2512-Q8_0',
			tags: ['Instruct', '2512']
		});
	});

	// TODO: fix and merge working tests into one block
	it.skip('handles real-world examples correctly', () => {
		expect(parseModelId('noctrex/GLM-4.7-Flash-MXFP4_MOE')).toStrictEqual({
			activatedParams: null,
			format: 'MXFP4_MOE', // TODO: does this make sense?
			modelName: 'GLM-4.7-Flash',
			orgName: 'noctrex',
			params: null,
			quantization: null,
			raw: 'noctrex/GLM-4.7-Flash-MXFP4_MOE',
			tags: []
		});
	});

	// TODO: fix and merge working tests into one block
	it.skip('handles real-world examples correctly', () => {
		expect(parseModelId('Qwen/Qwen3-Coder-Next-Q4_K_M')).toStrictEqual({
			activatedParams: null,
			format: null,
			modelName: 'Qwen3-Coder-Next',
			orgName: 'Qwen',
			params: null,
			// purpose: 'coder',
			quantization: 'Q4_K_M',
			raw: 'Qwen/Qwen3-Coder-Next-Q4_K_M',
			tags: []
		});
	});

	// TODO: fix and merge working tests into one block
	it.skip('handles real-world examples correctly', () => {
		expect(parseModelId('openai/gpt-oss-120b-Q4_K_M')).toStrictEqual({
			activatedParams: null,
			format: null,
			modelName: 'gpt-oss',
			orgName: 'openai',
			params: '120B',
			quantization: 'Q4_K_M',
			raw: 'openai/gpt-oss-120b-Q4_K_M',
			tags: []
		});
	});

	// TODO: fix and merge working tests into one block
	it.skip('handles real-world examples correctly', () => {
		expect(parseModelId('openai/gpt-oss-20b-F16')).toStrictEqual({
			activatedParams: null,
			format: null,
			modelName: 'gpt-oss',
			orgName: 'openai',
			params: '20B',
			quantization: 'F16',
			raw: 'openai/gpt-oss-20b-F16',
			tags: []
		});
	});

	// TODO: fix and merge working tests into one block
	it.skip('handles real-world examples correctly', () => {
		expect(parseModelId('nomic-embed-text-v2-moe.Q4_K_M')).toStrictEqual({
			activatedParams: null,
			format: null,
			modelName: 'nomic-embed-text-v2-moe',
			orgName: null,
			params: null,
			// purpose: 'embed',
			quantization: 'Q4_K_M',
			raw: 'nomic-embed-text-v2-moe.Q4_K_M',
			tags: ['MOE']
		});
	});
});
