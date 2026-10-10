import { detectToolUseSupport } from '$lib/utils/chat-template-tool-detector';
import { minMemoryTierGb } from '$lib/utils/model-compatibility';
import { describe, expect, it } from 'vitest';

describe('minMemoryTierGb', () => {
	it('picks the smallest tier whose budget fits the file', () => {
		// budget(16) = 16 * 1024 * 0.75 - 2048 = 10240 MiB; ~9.2 GiB file fits
		expect(minMemoryTierGb(9.5 * 1024 * 1024 * 1024)).toBe(16);
		// budget(12) = 7168 MiB; the same file does not fit
		expect(minMemoryTierGb(9.5 * 1024 * 1024 * 1024)).not.toBe(12);
	});

	it('applies the quant headroom to the file size', () => {
		// exactly the tier-32 budget before the 1.05 headroom; with it the file
		// spills into the next tier
		const budget32Mb = 32 * 1024 * 0.75 - 2048;
		const bytes = (budget32Mb / 1.05) * 1024 * 1024;

		expect(minMemoryTierGb(bytes)).toBe(32);
		expect(minMemoryTierGb(bytes + 1)).toBe(48);
	});

	it('returns null for empty sizes and over-budget files', () => {
		expect(minMemoryTierGb(0)).toBeNull();
		expect(minMemoryTierGb(4 * 1024 * 1024 * 1024 * 1024)).toBeNull();
	});
});

describe('detectToolUseSupport', () => {
	it('detects the jinja tools variable', () => {
		expect(detectToolUseSupport('{% for tool in tools %}')).toBe(true);
		expect(detectToolUseSupport('{{ tools | tojson }}')).toBe(true);
	});

	it('detects tool-call tokens case-insensitively', () => {
		expect(detectToolUseSupport('WRITES <tool_call> BLOCKS')).toBe(true);
		expect(detectToolUseSupport('emits Tool_Call sections')).toBe(true);
	});

	it('rejects templates without tool references', () => {
		expect(detectToolUseSupport('')).toBe(false);
		expect(detectToolUseSupport('{{ prompt }}')).toBe(false);
	});
});
