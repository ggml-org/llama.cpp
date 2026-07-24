import { describe, expect, it } from 'vitest';
import { RouterService } from '$lib/services/router.service';

describe('RouterService.fromModelId', () => {
	it('preserves simple repo ids', () => {
		expect(RouterService.fromModelId('ggml-org/gemma-3-4b-it-GGUF')).toBe(
			'#/model-hub/ggml-org/gemma-3-4b-it-GGUF'
		);
	});

	it('strips a single :tag suffix', () => {
		expect(RouterService.fromModelId('ggml-org/gemma-3-4b-it-GGUF:Q4_K_M')).toBe(
			'#/model-hub/ggml-org/gemma-3-4b-it-GGUF'
		);
	});

	it('strips a quant+variant :tag suffix', () => {
		expect(RouterService.fromModelId('ggml-org/gemma-3-4b-it-GGUF:IQ1_M-MTP')).toBe(
			'#/model-hub/ggml-org/gemma-3-4b-it-GGUF'
		);
	});

	it('only splits on the FIRST colon (defensive)', () => {
		// Most ids have one colon; if a malformed id ever carries multiple,
		// we still keep the prefix intact rather than nuking the middle.
		expect(RouterService.fromModelId('repo/with:more:colons')).toBe('#/model-hub/repo/with');
	});

	it('returns the bare URL for empty input', () => {
		expect(RouterService.fromModelId('')).toBe('#/model-hub/');
	});
});
