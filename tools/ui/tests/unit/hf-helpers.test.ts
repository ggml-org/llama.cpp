import { HuggingFaceService } from '$lib/services/huggingface.service';
import { describe, expect, it } from 'vitest';

const {
	collapseGgufShards,
	formatSizeRange,
	getBitDepth,
	parseCachePath,
	parseParamCount,
	parseSizeBytes
} = HuggingFaceService;

describe('collapseGgufShards', () => {
	it('passes non-sharded files through', () => {
		const files = [
			{ path: 'Model-Q4_K_M.gguf', size: 100 },
			{ path: 'mmproj-F16.gguf', size: 10 }
		];

		expect(collapseGgufShards(files)).toStrictEqual(files);
	});

	it('collapses a shard set to its first shard with the summed size', () => {
		const files = [
			{ path: 'Model-00001-of-00003.gguf', size: 10 },
			{ path: 'Model-00002-of-00003.gguf', size: 20 },
			{ path: 'Model-00003-of-00003.gguf', size: 30 }
		];

		expect(collapseGgufShards(files)).toStrictEqual([
			{ path: 'Model-00001-of-00003.gguf', size: 60 }
		]);
	});

	it('treats a missing shard as zero bytes', () => {
		const files = [
			{ path: 'Model-00001-of-00002.gguf', size: 10 },
			{ path: 'Model-Q8_0.gguf', size: 5 }
		];

		expect(collapseGgufShards(files)).toStrictEqual([
			{ path: 'Model-00001-of-00002.gguf', size: 10 },
			{ path: 'Model-Q8_0.gguf', size: 5 }
		]);
	});
});

describe('getBitDepth', () => {
	it('resolves known tokens', () => {
		expect(getBitDepth('Q4_K_M')).toBe(4);
		expect(getBitDepth('BF16')).toBe(16);
		expect(getBitDepth('IQ2_XXS')).toBe(2);
	});

	it('strips the UD prefix', () => {
		expect(getBitDepth('UD-Q4_K_XL')).toBe(4);
	});

	it('falls back to the leading precision digits', () => {
		expect(getBitDepth('TQ1_0')).toBe(1);
		expect(getBitDepth('MXFP4_MOE')).toBe(4);
	});

	it('returns null for unrecognized tokens', () => {
		expect(getBitDepth('xyz')).toBeNull();
		expect(getBitDepth('QUANT')).toBeNull();
	});
});

describe('formatSizeRange', () => {
	it('formats a gigabyte range without spaces around the dash', () => {
		expect(formatSizeRange(19e9, 28.6e9)).toBe('19.0-28.6 GB');
	});

	it('downgrades the unit to the smaller bound when the max is small', () => {
		expect(formatSizeRange(1e6, 2.5e6)).toBe('1.0-2.5 MB');
	});

	it('formats sub-kilobyte sizes in bytes', () => {
		expect(formatSizeRange(100, 900)).toBe('100-900 B');
	});
});

describe('parseCachePath', () => {
	it('parses a posix cache path into repo and file', () => {
		expect(
			parseCachePath(
				'/home/u/.cache/llama.cpp/models--ggml-org--Qwen3-8B-GGUF/snapshots/abc123/Q4_K_M.gguf'
			)
		).toStrictEqual({ file: 'Q4_K_M.gguf', repo: 'ggml-org/Qwen3-8B-GGUF' });
	});

	it('accepts windows separators', () => {
		expect(
			parseCachePath('C:\\cache\\models--org--Model\\snapshots\\sha\\sub\\file.gguf')
		).toStrictEqual({ file: 'sub/file.gguf', repo: 'org/Model' });
	});

	it('returns null for non-cache paths', () => {
		expect(parseCachePath('/models/foo.gguf')).toBeNull();
	});
});

describe('parseParamCount', () => {
	it('extracts billions and millions', () => {
		expect(parseParamCount('Qwen3.8-27B-GGUF')).toBe('27B');
		expect(parseParamCount('embeddinggemma-300M-GGUF')).toBe('300M');
		expect(parseParamCount('Model-0.6B-Q4_K_M')).toBe('0.6B');
	});

	it('returns null when no size token is present', () => {
		expect(parseParamCount('ggml-org/Laguna-S-GGUF')).toBeNull();
	});
});

describe('parseSizeBytes', () => {
	it('parses single-letter catalog size suffixes', () => {
		expect(parseSizeBytes('177g')).toBe(177e9);
		expect(parseSizeBytes('1.2 t')).toBe(1.2e12);
		expect(parseSizeBytes('500m')).toBe(500e6);
	});

	it('returns null for malformed input', () => {
		expect(parseSizeBytes('unknown')).toBeNull();
		expect(parseSizeBytes('12 parsecs')).toBeNull();
		expect(parseSizeBytes('')).toBeNull();
	});
});
