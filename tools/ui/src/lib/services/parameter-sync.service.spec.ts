import { ParameterSyncService } from './parameter-sync.service';
import { describe, expect, it } from 'vitest';

describe('ParameterSyncService', () => {
	describe('roundFloatingPoint', () => {
		it('should fix JavaScript floating-point precision issues', () => {
			// Test the specific values from the screenshot
			const mockServerParams = {
				min_p: 0.009999999776482582,
				samplers: ['top_k', 'typ_p', 'top_p', 'min_p', 'temperature'],
				temperature: 0.800000011920929,
				top_k: 40,
				top_p: 0.949999988079071
			};
			const result = ParameterSyncService.extractServerDefaults({
				...mockServerParams,
				chat_format: '',
				dry_allowed_length: 2,
				dry_base: 1.75,
				dry_multiplier: 0.0,
				dry_penalty_last_n: 64,
				dry_sequence_breakers: [],
				dynatemp_exponent: 1.0,
				dynatemp_range: 0.0,
				frequency_penalty: 0.0,
				generation_prompt: '',
				grammar: '',
				grammar_lazy: false,
				grammar_triggers: [],
				ignore_eos: false,
				logit_bias: [],
				lora: [],
				max_tokens: -1,
				min_keep: 0,
				mirostat: 0,
				mirostat_eta: 0.1,
				mirostat_tau: 5.0,
				n_discard: 0,
				n_keep: 0,
				// Add other required fields to match the API type
				n_predict: 512,
				n_probs: 0,
				post_sampling_probs: false,
				presence_penalty: 0.0,
				preserved_tokens: [],
				reasoning_format: '',
				reasoning_in_content: false,
				repeat_last_n: 64,
				repeat_penalty: 1.0,
				seed: -1,
				'speculative.n_max': 0,
				'speculative.n_min': 0,
				'speculative.p_min': 0.0,
				stop: [],
				stream: true,
				timings_per_token: false,
				top_n_sigma: 0.0,
				typ_p: 1.0,
				xtc_probability: 0.0,
				xtc_threshold: 0.1
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
				max_tokens: -1,
				samplers: ['top_k', 'temperature'],
				temperature: 0.7
			};
			const result = ParameterSyncService.extractServerDefaults({
				...mockServerParams,
				chat_format: '',
				dry_allowed_length: 2,
				dry_base: 1.75,
				dry_multiplier: 0.0,
				dry_penalty_last_n: 64,
				dry_sequence_breakers: [],
				dynatemp_exponent: 1.0,
				dynatemp_range: 0.0,
				frequency_penalty: 0.0,
				generation_prompt: '',
				grammar: '',
				grammar_lazy: false,
				grammar_triggers: [],
				ignore_eos: false,
				logit_bias: [],
				lora: [],
				min_keep: 0,
				min_p: 0.05,
				mirostat: 0,
				mirostat_eta: 0.1,
				mirostat_tau: 5.0,
				n_discard: 0,
				n_keep: 0,
				// Minimal required fields
				n_predict: 512,
				n_probs: 0,
				post_sampling_probs: false,
				presence_penalty: 0.0,
				preserved_tokens: [],
				reasoning_format: '',
				reasoning_in_content: false,
				repeat_last_n: 64,
				repeat_penalty: 1.0,
				seed: -1,
				'speculative.n_max': 0,
				'speculative.n_min': 0,
				'speculative.p_min': 0.0,
				stop: [],
				stream: true,
				timings_per_token: false,
				top_k: 40,
				top_n_sigma: 0.0,
				top_p: 0.95,
				typ_p: 1.0,
				xtc_probability: 0.0,
				xtc_threshold: 0.1
			} as ApiLlamaCppServerProps['default_generation_settings']['params']);

			expect(result.samplers).toBe('top_k;temperature');
			expect(result.max_tokens).toBe(-1);
			expect(result.temperature).toBe(0.7);
		});

		it('extracts sampling twins only, never ui settings', () => {
			const result = ParameterSyncService.extractServerDefaults(null);

			expect(result).toEqual({});
			expect(ParameterSyncService.canSyncParameter('theme')).toBe(false);
			expect(ParameterSyncService.canSyncParameter('pdfAsImage')).toBe(false);
			expect(ParameterSyncService.canSyncParameter('temperature')).toBe(true);
		});
	});
});
