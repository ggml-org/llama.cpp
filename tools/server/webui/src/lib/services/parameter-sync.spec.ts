import { describe, it, expect } from 'vitest';
import { ParameterSyncService } from './parameter-sync';

describe('ParameterSyncService', () => {
	describe('roundFloatingPoint', () => {
		it('should fix JavaScript floating-point precision issues', () => {
			// Test the specific values from the screenshot
			const mockServerParams = {
				top_p: 0.949999988079071,
				min_p: 0.009999999776482582,
				temperature: 0.800000011920929,
				top_k: 40,
				samplers: ['top_k', 'typ_p', 'top_p', 'min_p', 'temperature']
			};

			const result = ParameterSyncService.extractServerDefaults({
				...mockServerParams,
				// Add other required fields to match the API type
				n_predict: 512,
				seed: -1,
				dynatemp_range: 0.0,
				dynatemp_exponent: 1.0,
				xtc_probability: 0.0,
				xtc_threshold: 0.1,
				typ_p: 1.0,
				repeat_last_n: 64,
				repeat_penalty: 1.0,
				presence_penalty: 0.0,
				frequency_penalty: 0.0,
				dry_multiplier: 0.0,
				dry_base: 1.75,
				dry_allowed_length: 2,
				dry_penalty_last_n: -1,
				mirostat: 0,
				mirostat_tau: 5.0,
				mirostat_eta: 0.1,
				stop: [],
				max_tokens: -1,
				n_keep: 0,
				n_discard: 0,
				ignore_eos: false,
				stream: true,
				logit_bias: [],
				n_probs: 0,
				min_keep: 0,
				grammar: '',
				grammar_lazy: false,
				grammar_triggers: [],
				preserved_tokens: [],
				chat_format: '',
				reasoning_format: '',
				reasoning_in_content: false,
				thinking_forced_open: false,
				'speculative.n_max': 0,
				'speculative.n_min': 0,
				'speculative.p_min': 0.0,
				timings_per_token: false,
				post_sampling_probs: false,
				lora: [],
				top_n_sigma: 0.0,
				dry_sequence_breakers: []
			} as ApiLlamaCppServerProps['default_generation_settings']['params']);

			// Check that the problematic floating-point values are rounded correctly
			expect(result.top_p).toBe(0.95);
			expect(result.min_p).toBe(0.01);
			expect(result.temperature).toBe(0.8);
			expect(result.top_k).toBe(40); // Integer should remain unchanged
			expect(result.samplers).toBe('top_k;typ_p;top_p;min_p;temperature');
		});

		it('should preserve non-numeric values', () => {
			const mockServerParams = {
				samplers: ['top_k', 'temperature'],
				max_tokens: -1,
				temperature: 0.7
			};

			const result = ParameterSyncService.extractServerDefaults({
				...mockServerParams,
				// Minimal required fields
				n_predict: 512,
				seed: -1,
				dynatemp_range: 0.0,
				dynatemp_exponent: 1.0,
				top_k: 40,
				top_p: 0.95,
				min_p: 0.05,
				xtc_probability: 0.0,
				xtc_threshold: 0.1,
				typ_p: 1.0,
				repeat_last_n: 64,
				repeat_penalty: 1.0,
				presence_penalty: 0.0,
				frequency_penalty: 0.0,
				dry_multiplier: 0.0,
				dry_base: 1.75,
				dry_allowed_length: 2,
				dry_penalty_last_n: -1,
				mirostat: 0,
				mirostat_tau: 5.0,
				mirostat_eta: 0.1,
				stop: [],
				n_keep: 0,
				n_discard: 0,
				ignore_eos: false,
				stream: true,
				logit_bias: [],
				n_probs: 0,
				min_keep: 0,
				grammar: '',
				grammar_lazy: false,
				grammar_triggers: [],
				preserved_tokens: [],
				chat_format: '',
				reasoning_format: '',
				reasoning_in_content: false,
				thinking_forced_open: false,
				'speculative.n_max': 0,
				'speculative.n_min': 0,
				'speculative.p_min': 0.0,
				timings_per_token: false,
				post_sampling_probs: false,
				lora: [],
				top_n_sigma: 0.0,
				dry_sequence_breakers: []
			} as ApiLlamaCppServerProps['default_generation_settings']['params']);

			expect(result.samplers).toBe('top_k;temperature');
			expect(result.max_tokens).toBe(-1);
			expect(result.temperature).toBe(0.7);
		});

		it('should merge webui settings from props when provided', () => {
			const result = ParameterSyncService.extractServerDefaults(null, {
				pasteLongTextToFileLen: 0,
				pdfAsImage: true,
				renderUserContentAsMarkdown: false,
				theme: 'dark'
			});

			expect(result.pasteLongTextToFileLen).toBe(0);
			expect(result.pdfAsImage).toBe(true);
			expect(result.renderUserContentAsMarkdown).toBe(false);
			expect(result.theme).toBeUndefined();
		});
	});
});
