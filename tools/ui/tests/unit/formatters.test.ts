import { formatContextLength, formatParameters } from '$lib/utils/formatters';
import { describe, expect, it } from 'vitest';

describe('formatContextLength', () => {
	it('shows a dash when no value is known', () => {
		expect(formatContextLength([])).toBe('-');
	});

	it('prints the value with thousands separators', () => {
		expect(formatContextLength([8192])).toBe('8,192 tokens');
	});

	it('pairs the configured context with the supported one', () => {
		expect(formatContextLength([8192, 131072])).toBe('8,192 / 131,072 tokens');
	});
});

describe('formatParameters', () => {
	it('shortens large parameter counts', () => {
		expect(formatParameters(35_000_000_000)).toBe('35B');
		expect(formatParameters(3_000_000)).toBe('3.00M');
		expect(formatParameters(512)).toBe('512');
	});
});
