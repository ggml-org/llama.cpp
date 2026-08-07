import { describe, expect, it } from 'vitest';
import { HuggingFaceService } from '$lib/services/huggingface.service';

const { extractQuantMeta, getBitDepth } = HuggingFaceService;

describe('extractQuantMeta', () => {
	describe('main model files (no variant)', () => {
		it('parses main model with trailing quant', () => {
			expect(extractQuantMeta('Llama-3-8B-Instruct-Q4_K_M.gguf')).toStrictEqual({
				quant: 'Q4_K_M',
				variant: null,
				variantForm: null
			});
		});

		it('parses bare quant filename', () => {
			expect(extractQuantMeta('Q4_K_M.gguf')).toStrictEqual({
				quant: 'Q4_K_M',
				variant: null,
				variantForm: null
			});
		});

		it('parses BF16 main file', () => {
			expect(extractQuantMeta('qp-Hy3-GGUF-BF16.gguf')).toStrictEqual({
				quant: 'BF16',
				variant: null,
				variantForm: null
			});
		});

		it('returns null quant when no recognized quant segment exists', () => {
			expect(extractQuantMeta('qp-Hy3-GGUF.gguf')).toStrictEqual({
				quant: null,
				variant: null,
				variantForm: null
			});
		});

		it('returns null for non-weight files', () => {
			expect(extractQuantMeta('README.md')).toBeNull();
			expect(extractQuantMeta('config.json')).toBeNull();
		});
	});

	describe('embedded draft (suffix variant)', () => {
		it('parses -mtp suffix with preceding quant', () => {
			expect(extractQuantMeta('Hy3-IQ1_M-mtp.gguf')).toStrictEqual({
				quant: 'IQ1_M',
				variant: 'mtp',
				variantForm: 'suffix'
			});
		});

		it('parses -MTP suffix (uppercase) with preceding quant', () => {
			expect(extractQuantMeta('Hy3-IQ1_M-MTP.gguf')).toStrictEqual({
				quant: 'IQ1_M',
				variant: 'mtp',
				variantForm: 'suffix'
			});
		});

		it('parses -mtp suffix on bare quant filename', () => {
			expect(extractQuantMeta('Q2_K_XL-mtp.gguf')).toStrictEqual({
				quant: 'Q2_K_XL',
				variant: 'mtp',
				variantForm: 'suffix'
			});
		});

		it('does not flag -mtp suffix unless the head segment is a real quant', () => {
			// a model literally named `MyModel-mtp` should remain un-variant-flagged
			expect(extractQuantMeta('MyModel-mtp.gguf')).toStrictEqual({
				quant: null,
				variant: null,
				variantForm: null
			});
		});
	});

	describe('sidecar draft (prefix variant)', () => {
		it('parses mtp-<quant>.gguf prefix', () => {
			expect(extractQuantMeta('mtp-Q4_0.gguf')).toStrictEqual({
				quant: 'Q4_0',
				variant: 'mtp',
				variantForm: 'prefix'
			});
		});

		it('parses mtp-<quant>-<size>.gguf with trailing size hint', () => {
			expect(extractQuantMeta('mtp-Q4_0-180MB.gguf')).toStrictEqual({
				quant: 'Q4_0',
				variant: 'mtp',
				variantForm: 'prefix'
			});
		});

		it('parses dflash-<quant>.gguf prefix', () => {
			expect(extractQuantMeta('dflash-BF16.gguf')).toStrictEqual({
				quant: 'BF16',
				variant: 'dflash',
				variantForm: 'prefix'
			});
		});

		it('parses dflash-<quant>.gguf with size suffix', () => {
			expect(extractQuantMeta('dflash-BF16-3GB.gguf')).toStrictEqual({
				quant: 'BF16',
				variant: 'dflash',
				variantForm: 'prefix'
			});
		});

		it('parses mmproj-<model>.gguf prefix', () => {
			expect(extractQuantMeta('mmproj-Llama-3-8B-F16.gguf')).toStrictEqual({
				quant: 'F16',
				variant: 'mmproj',
				variantForm: 'prefix'
			});
		});
	});
});

describe('getBitDepth', () => {
	it('returns bits per weight for known quantization tokens', () => {
		expect(getBitDepth('IQ1_M')).toBe(1);
		expect(getBitDepth('IQ2_XS')).toBe(2);
		expect(getBitDepth('Q4_K_M')).toBe(4);
		expect(getBitDepth('Q8_0')).toBe(8);
		expect(getBitDepth('BF16')).toBe(16);
		expect(getBitDepth('F16')).toBe(16);
	});

	it('returns null for unknown tokens', () => {
		expect(getBitDepth('UNKNOWN_QUANT')).toBeNull();
		expect(getBitDepth('X4K_M')).toBeNull();
	});
});
