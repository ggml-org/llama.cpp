import { describe, expect, it } from 'vitest';
import { redactValue } from '$lib/utils/redact';

describe('redactValue', () => {
	it('returns [redacted] by default', () => {
		expect(redactValue('secret-token')).toBe('[redacted]');
	});

	it('shows last N characters when showLastChars is provided', () => {
		expect(redactValue('session-abc12', 5)).toBe('....abc12');
	});

	it('handles value shorter than showLastChars', () => {
		expect(redactValue('ab', 5)).toBe('....ab');
	});

	it('returns [redacted] when showLastChars is 0', () => {
		expect(redactValue('secret', 0)).toBe('[redacted]');
	});
});
