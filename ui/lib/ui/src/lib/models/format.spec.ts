import {
	formatFileSize,
	formatParameters,
	formatSizeRange,
	getAvatarUrl,
	minMemoryTierGb,
	summarizeSizes
} from './format.js';
import { describe, expect, it } from 'vite-plus/test';

describe('Model UI formatting', () => {
	it('formats file sizes and ranges in decimal units', () => {
		expect(formatFileSize(12_109_566_560)).toBe('12.1 GB');
		expect(formatFileSize(740_000_000)).toBe('740.0 MB');
		expect(formatSizeRange(16_400_000_000, 28_600_000_000)).toBe('16.4-28.6 GB');
	});

	it('formats parameter and token counts', () => {
		expect(formatParameters(8_000_000_000)).toBe('8B');
		expect(formatParameters(262_144)).toBe('262.14K');
		expect(formatParameters('n/a')).toBe('Unknown');
	});

	it('builds avatar URLs without the provider tilde', () => {
		expect(getAvatarUrl('~openai')).toBe('https://huggingface.co/api/avatars/openai');
	});
});

describe('minMemoryTierGb', () => {
	it('picks the smallest tier the weights fit in', () => {
		// gpt-oss-20b, 12.1 GB: 24 GB at 75% minus 2 GB overhead is the first budget that holds it
		expect(minMemoryTierGb(12_109_566_560)).toBe(24);
		expect(minMemoryTierGb(0)).toBeNull();
	});

	it("takes a platform's own tiers", () => {
		expect(minMemoryTierGb(12_109_566_560, [8, 16, 18, 24])).toBe(24);
		expect(minMemoryTierGb(4_000_000_000, [8, 16])).toBe(8);
		expect(minMemoryTierGb(900_000_000_000, [8, 16])).toBeNull();
	});
});

describe('summarizeSizes', () => {
	it('spans the smallest to the largest and counts every size', () => {
		expect(summarizeSizes(['E2B', 'E4B', '12B', '26B-A4B', '31B'])).toBe('E2B–31B · 5 sizes');
		expect(summarizeSizes(['3B', '3B', '8B', '8B', '14B', '14B'])).toBe('3B–14B · 6 sizes');
	});

	it('names a single size alone', () => {
		expect(summarizeSizes(['740M'])).toBe('740M');
		expect(summarizeSizes([])).toBe('');
	});
});
