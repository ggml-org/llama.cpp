import { describe, expect, it } from 'vitest';
import { ModelsService } from '$lib/services/models.service';

const { parseModelId } = ModelsService;

describe('parseModelId', () => {
	it('handles unknown patterns correctly', () => {
		expect(parseModelId('model-name-1')).toStrictEqual({
			activatedParams: null,
			modelName: 'model-name-1',
			orgName: null,
			params: null,
			quantization: null,
			raw: 'model-name-1',
			tags: []
		});

		expect(parseModelId('org/model-name-2')).toStrictEqual({
			activatedParams: null,
			modelName: 'model-name-2',
			orgName: 'org',
			params: null,
			quantization: null,
			raw: 'org/model-name-2',
			tags: []
		});
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

	it('extracts activated parameters correctly in lowercase', () => {
		expect(parseModelId('model-100b-a10b-bf16')).toMatchObject({ activatedParams: 'A10B' });
		expect(parseModelId('model-100b-a10b:q4_k_m')).toMatchObject({ activatedParams: 'A10B' });
	});

	it('extracts quantization correctly', () => {
		// Dash-separated quantization
		expect(parseModelId('model-100B-UD-IQ1_S')).toMatchObject({ quantization: 'UD-IQ1_S' });
		expect(parseModelId('model-100B-IQ4_XS')).toMatchObject({ quantization: 'IQ4_XS' });
		expect(parseModelId('model-100B-Q4_K_M')).toMatchObject({ quantization: 'Q4_K_M' });
		expect(parseModelId('model-100B-Q8_0')).toMatchObject({ quantization: 'Q8_0' });
		expect(parseModelId('model-100B-UD-Q8_K_XL')).toMatchObject({ quantization: 'UD-Q8_K_XL' });
		expect(parseModelId('model-100B-F16')).toMatchObject({ quantization: 'F16' });
		expect(parseModelId('model-100B-BF16')).toMatchObject({ quantization: 'BF16' });
		expect(parseModelId('model-100B-MXFP4')).toMatchObject({ quantization: 'MXFP4' });

		// Colon-separated quantization
		expect(parseModelId('model-100B:UD-IQ1_S')).toMatchObject({ quantization: 'UD-IQ1_S' });
		expect(parseModelId('model-100B:IQ4_XS')).toMatchObject({ quantization: 'IQ4_XS' });
		expect(parseModelId('model-100B:Q4_K_M')).toMatchObject({ quantization: 'Q4_K_M' });
		expect(parseModelId('model-100B:Q8_0')).toMatchObject({ quantization: 'Q8_0' });
		expect(parseModelId('model-100B:UD-Q8_K_XL')).toMatchObject({ quantization: 'UD-Q8_K_XL' });
		expect(parseModelId('model-100B:F16')).toMatchObject({ quantization: 'F16' });
		expect(parseModelId('model-100B:BF16')).toMatchObject({ quantization: 'BF16' });
		expect(parseModelId('model-100B:MXFP4')).toMatchObject({ quantization: 'MXFP4' });

		// Dot-separated quantization
		expect(parseModelId('nomic-embed-text-v2-moe.Q4_K_M')).toMatchObject({
			quantization: 'Q4_K_M'
		});
	});

	it('extracts additional tags correctly', () => {
		expect(parseModelId('model-100B-foobar-Q4_K_M')).toMatchObject({ tags: ['foobar'] });
		expect(parseModelId('model-100B-A10B-foobar-1M-BF16')).toMatchObject({
			tags: ['foobar', '1M']
		});
		expect(parseModelId('model-100B-1M-foobar:UD-Q8_K_XL')).toMatchObject({
			tags: ['1M', 'foobar']
		});
	});

	it('filters out container format segments from tags', () => {
		expect(parseModelId('model-100B-GGUF-Instruct-BF16')).toMatchObject({
			tags: ['Instruct']
		});
		expect(parseModelId('model-100B-GGML-Instruct:Q4_K_M')).toMatchObject({
			tags: ['Instruct']
		});
	});

	it('handles real-world examples correctly', () => {
		expect(parseModelId('meta-llama/Llama-3.1-8B')).toStrictEqual({
			activatedParams: null,
			modelName: 'Llama-3.1',
			orgName: 'meta-llama',
			params: '8B',
			quantization: null,
			raw: 'meta-llama/Llama-3.1-8B',
			tags: []
		});

		expect(parseModelId('openai/gpt-oss-120b-MXFP4')).toStrictEqual({
			activatedParams: null,
			modelName: 'gpt-oss',
			orgName: 'openai',
			params: '120B',
			quantization: 'MXFP4',
			raw: 'openai/gpt-oss-120b-MXFP4',
			tags: []
		});

		expect(parseModelId('openai/gpt-oss-20b:Q4_K_M')).toStrictEqual({
			activatedParams: null,
			modelName: 'gpt-oss',
			orgName: 'openai',
			params: '20B',
			quantization: 'Q4_K_M',
			raw: 'openai/gpt-oss-20b:Q4_K_M',
			tags: []
		});

		expect(parseModelId('Qwen/Qwen3-Coder-30B-A3B-Instruct-1M-BF16')).toStrictEqual({
			activatedParams: 'A3B',
			modelName: 'Qwen3-Coder',
			orgName: 'Qwen',
			params: '30B',
			quantization: 'BF16',
			raw: 'Qwen/Qwen3-Coder-30B-A3B-Instruct-1M-BF16',
			tags: ['Instruct', '1M']
		});
	});

	it('handles real-world examples with quantization in segments', () => {
		expect(parseModelId('meta-llama/Llama-4-Scout-17B-16E-Instruct-Q4_K_M')).toStrictEqual({
			activatedParams: null,
			modelName: 'Llama-4-Scout',
			orgName: 'meta-llama',
			params: '17B',
			quantization: 'Q4_K_M',
			raw: 'meta-llama/Llama-4-Scout-17B-16E-Instruct-Q4_K_M',
			tags: ['16E', 'Instruct']
		});

		expect(parseModelId('MiniMaxAI/MiniMax-M2-IQ4_XS')).toStrictEqual({
			activatedParams: null,
			modelName: 'MiniMax-M2',
			orgName: 'MiniMaxAI',
			params: null,
			quantization: 'IQ4_XS',
			raw: 'MiniMaxAI/MiniMax-M2-IQ4_XS',
			tags: []
		});

		expect(parseModelId('MiniMaxAI/MiniMax-M2-UD-Q3_K_XL')).toStrictEqual({
			activatedParams: null,
			modelName: 'MiniMax-M2',
			orgName: 'MiniMaxAI',
			params: null,
			quantization: 'UD-Q3_K_XL',
			raw: 'MiniMaxAI/MiniMax-M2-UD-Q3_K_XL',
			tags: []
		});

		expect(parseModelId('mistralai/Devstral-2-123B-Instruct-2512-Q4_K_M')).toStrictEqual({
			activatedParams: null,
			modelName: 'Devstral-2',
			orgName: 'mistralai',
			params: '123B',
			quantization: 'Q4_K_M',
			raw: 'mistralai/Devstral-2-123B-Instruct-2512-Q4_K_M',
			tags: ['Instruct', '2512']
		});

		expect(parseModelId('mistralai/Devstral-Small-2-24B-Instruct-2512-Q8_0')).toStrictEqual({
			activatedParams: null,
			modelName: 'Devstral-Small-2',
			orgName: 'mistralai',
			params: '24B',
			quantization: 'Q8_0',
			raw: 'mistralai/Devstral-Small-2-24B-Instruct-2512-Q8_0',
			tags: ['Instruct', '2512']
		});

		expect(parseModelId('noctrex/GLM-4.7-Flash-MXFP4_MOE')).toStrictEqual({
			activatedParams: null,
			modelName: 'GLM-4.7-Flash',
			orgName: 'noctrex',
			params: null,
			quantization: 'MXFP4_MOE',
			raw: 'noctrex/GLM-4.7-Flash-MXFP4_MOE',
			tags: []
		});

		expect(parseModelId('Qwen/Qwen3-Coder-Next-Q4_K_M')).toStrictEqual({
			activatedParams: null,
			modelName: 'Qwen3-Coder-Next',
			orgName: 'Qwen',
			params: null,
			quantization: 'Q4_K_M',
			raw: 'Qwen/Qwen3-Coder-Next-Q4_K_M',
			tags: []
		});

		expect(parseModelId('openai/gpt-oss-120b-Q4_K_M')).toStrictEqual({
			activatedParams: null,
			modelName: 'gpt-oss',
			orgName: 'openai',
			params: '120B',
			quantization: 'Q4_K_M',
			raw: 'openai/gpt-oss-120b-Q4_K_M',
			tags: []
		});

		expect(parseModelId('openai/gpt-oss-20b-F16')).toStrictEqual({
			activatedParams: null,
			modelName: 'gpt-oss',
			orgName: 'openai',
			params: '20B',
			quantization: 'F16',
			raw: 'openai/gpt-oss-20b-F16',
			tags: []
		});

		expect(parseModelId('nomic-embed-text-v2-moe.Q4_K_M')).toStrictEqual({
			activatedParams: null,
			modelName: 'nomic-embed-text-v2-moe',
			orgName: null,
			params: null,
			quantization: 'Q4_K_M',
			raw: 'nomic-embed-text-v2-moe.Q4_K_M',
			tags: []
		});
	});

	it('handles ambiguous model names', () => {
		// Qwen3.5 Instruct vs Thinking — tags should distinguish them
		expect(parseModelId('Qwen/Qwen3.5-30B-A3B-Instruct')).toMatchObject({
			modelName: 'Qwen3.5',
			params: '30B',
			activatedParams: 'A3B',
			tags: ['Instruct']
		});

		expect(parseModelId('Qwen/Qwen3.5-30B-A3B-Thinking')).toMatchObject({
			modelName: 'Qwen3.5',
			params: '30B',
			activatedParams: 'A3B',
			tags: ['Thinking']
		});

		// Dot-separated quantization with variant suffixes
		expect(parseModelId('gemma-3-27b-it-heretic-v2.Q8_0')).toMatchObject({
			modelName: 'gemma-3',
			params: '27B',
			quantization: 'Q8_0',
			tags: ['it', 'heretic', 'v2']
		});

		expect(parseModelId('gemma-3-27b-it.Q8_0')).toMatchObject({
			modelName: 'gemma-3',
			params: '27B',
			quantization: 'Q8_0',
			tags: ['it']
		});
	});
});
