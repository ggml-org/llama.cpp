import { describe, expect, it } from 'vitest';
import { ModelsService } from '$lib/services/models.service';

const { buildDownloadTag } = ModelsService;

describe('buildDownloadTag', () => {
	it('returns the bare repo id when no tag is provided', () => {
		expect(buildDownloadTag('ggml-org/Qwen3-GGUF', null)).toBe('ggml-org/Qwen3-GGUF');
	});

	it('appends bare quantization as :<quant>', () => {
		expect(buildDownloadTag('ggml-org/Qwen3-GGUF', { quant: 'Q4_K_M', variant: null })).toBe(
			'ggml-org/Qwen3-GGUF:Q4_K_M'
		);
	});

	it('upcases the variant and appends as :<quant>-<VARIANT>', () => {
		expect(buildDownloadTag('ggml-org/Qwen3-GGUF', { quant: 'IQ1_M', variant: 'mtp' })).toBe(
			'ggml-org/Qwen3-GGUF:IQ1_M-MTP'
		);
	});

	it('upcases dflash variant', () => {
		expect(buildDownloadTag('ggml-org/Qwen3-GGUF', { quant: 'BF16', variant: 'dflash' })).toBe(
			'ggml-org/Qwen3-GGUF:BF16-DFLASH'
		);
	});

	it('upcases mmproj variant (for documentation, not currently called with mmproj)', () => {
		expect(buildDownloadTag('ggml-org/Qwen3-GGUF', { quant: 'F16', variant: 'mmproj' })).toBe(
			'ggml-org/Qwen3-GGUF:F16-MMPROJ'
		);
	});
});
